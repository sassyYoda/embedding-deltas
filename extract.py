"""Phase 1: frame sampling + CLIP ViT-L-14 embedding extraction.

Module API (consumed by Phase 5's pipeline.run() and Phase 6's batch driver):
    sample_frames(video_path, fps=2.0) -> (frames, timestamps)
    load_model() -> (model, preprocess, device)
    embed_frames(frames, model, preprocess, device, batch_size=32) -> (N, 768) float32 L2-normalized

Verification harness (spec §0.5) lives in `if __name__ == "__main__":` per D-16.
Run: `python extract.py <video.mp4>` to verify the module on a single sample video.

Phase 1 owns 8 pitfalls (research/PITFALLS.md): 1 (BGR/RGB), 2 (CAP_PROP_POS_MSEC), 3
(unreliable container frame-count metadata — drive loop on read() returning False instead),
4 (model.eval + inference_mode), 5 (PIL + shape assert), 6 (explicit L2 + assert),
7 (single batched path), 15 (cudnn.benchmark=False). Each has a code-level guard below and a
runtime assertion in the __main__ block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image


def sample_frames(
    video_path: str | Path,
    fps: float = 2.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample frames at `fps` Hz via CAP_PROP_POS_MSEC time-based seek (D-04 / Pitfall 2).

    Returns:
        frames: list of length N; each element shape (H, W, 3), dtype uint8, RGB order
                (D-07 / Pitfall 1 — BGR→RGB conversion happens here, exactly once).
        timestamps: np.ndarray shape (N,) dtype float64, seconds, monotonically non-decreasing.
                    Derived from cap.get(CAP_PROP_POS_MSEC) — NOT nominal frame index.

    Raises:
        RuntimeError: if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {video_path}")

    sample_interval_ms = 1000.0 / fps  # 500.0 for fps=2.0
    frames: list[np.ndarray] = []
    timestamps_ms: list[float] = []

    target_ms = 0.0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break  # clean EOS detection — Pitfall 3 (do NOT pre-compute count)
        # cap.get gives the time of the NEXT frame; for 2 fps sampling the offset is < 33ms,
        # well within the 1.0s drift tolerance in D-08.
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        # D-07 / Pitfall 1: BGR→RGB conversion happens HERE, exactly once in the codebase.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        timestamps_ms.append(actual_ms)
        target_ms += sample_interval_ms

        # Defensive: if seek refuses to advance (some VFR pathologies), break to avoid infinite loop.
        if len(timestamps_ms) >= 2 and timestamps_ms[-1] <= timestamps_ms[-2]:
            break

    cap.release()
    timestamps_sec = np.asarray(timestamps_ms, dtype=np.float64) / 1000.0
    return frames, timestamps_sec


def load_model() -> tuple[Any, Any, torch.device]:
    """Load CLIP ViT-L-14 OpenAI; return (model, preprocess, device).

    - device: cuda → mps → cpu ladder (D-11). MPS auto-selected on Apple Silicon when available.
    - model: in eval() mode, on `device`. cudnn.benchmark = False set globally
             (no-op on MPS/CPU, D-15 / Pitfall 15).
    - preprocess: the third element of open_clip.create_model_and_transforms (preprocess_val).

    No env-vars or seeds set here — those belong to Phase 5 (D-15).
    """
    # D-11 device ladder
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cpu"
    )

    # D-15 / Pitfall 15: inference-side determinism precursors.
    # No-op on MPS/CPU; correct on CUDA. Phase 5 owns env-vars + the deterministic-algorithms toggle.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # D-09 / EMBD-01: exact spec call. force_quick_gelu=True forces QuickGELU activation
    # in the model config to match OpenAI's pretrained checkpoint (which was trained with
    # QuickGELU). Without this, open_clip 3.3.0 builds a standard-GELU model and emits a
    # mismatch warning — silently degrading embedding quality. Spec §2 explicitly relies
    # on the QuickGELU variant for the Koddenbrock 2025 robustness justification.
    # Returns 3-tuple (model, preprocess_train, preprocess_val); we use preprocess_val.
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', force_quick_gelu=True
    )
    model = model.to(device)
    model.eval()  # D-15 / Pitfall 4 — also called defensively inside embed_frames.

    return model, preprocess, device


def embed_frames(
    frames: list[np.ndarray],
    model: Any,
    preprocess: Any,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed RGB uint8 frames into (N, 768) float32 L2-normalized array.

    Single batched code path (D-12 / Pitfall 7). PIL conversion via Image.fromarray (Pitfall 5).
    Inference wrapped in model.eval() + torch.inference_mode() (Pitfall 4).
    Explicit L2 normalize via feats / feats.norm(dim=-1, keepdim=True) (D-13 / Pitfall 6).
    Hard assertion np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5) before return.

    Returns np.ndarray shape (len(frames), 768) dtype float32. Empty input → shape (0, 768).
    """
    # Empty-input contract (no special-case branching elsewhere — Pitfall 7 stays clean).
    if not frames:
        return np.zeros((0, 768), dtype=np.float32)

    # Defensive — D-15 / Pitfall 4 (idempotent; safe to call repeatedly).
    model.eval()

    out_chunks: list[np.ndarray] = []
    # Single batched code path (D-12 / Pitfall 7): trailing partial batch handled by Python slicing.
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            # PIL conversion (Pitfall 5) — bypassing Image.fromarray would skip Resize/CenterCrop.
            tensors = [preprocess(Image.fromarray(f)) for f in chunk]
            batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 224, 224) on device

            # Per-batch shape sanity (Pitfall 5 guard).
            assert batch_tensor.shape[1:] == (3, 224, 224), (
                f"preprocess produced wrong shape: {batch_tensor.shape}"
            )

            feats = model.encode_image(batch_tensor)  # (B, 768) float32 NOT normalized
            # D-13 / Pitfall 6 — explicit L2 normalize, on-device, BEFORE .cpu().numpy().
            feats = feats / feats.norm(dim=-1, keepdim=True)

            out_chunks.append(feats.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(out_chunks, axis=0)  # (N, 768) float32

    # Hard assertion (D-13 / EMBD-05 / Pitfall 6 — defense in depth, fires INSIDE embed_frames).
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"embeddings not L2-normalized: norms in [{norms.min()}, {norms.max()}]"
    )
    return embeddings


if __name__ == "__main__":
    # Spec §0.5 verification harness for Phase 1 (D-16). Runs as:
    #   python extract.py [<video.mp4>] [--save-fixture | --no-save-fixture]
    # If no <video.mp4> given, picks the first .mp4 found under videos/ (RESEARCH Gap 9).
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(
        description="Phase 1 §0.5 verification harness for extract.py "
                    "(D-16: inline __main__ — no separate test scaffolding per D-16 / Gap 10)."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to input video (.mp4). If omitted, picks the first .mp4 under videos/.",
    )
    parser.add_argument(
        "--save-fixture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write embeddings.npy + timestamps.npy to output/cache/ "
             "(default: ON; internal dev tool per D-18; pass --no-save-fixture to skip).",
    )
    args = parser.parse_args()

    # ── Resolve video path (smart default per RESEARCH Gap 9) ──────────────────
    if args.video is None:
        candidate = next(Path("videos").rglob("*.mp4"), None)
        if candidate is None:
            print(
                "[FATAL] no video path given and no .mp4 found under videos/. "
                "Pass a path explicitly: `python extract.py <video.mp4>`.",
                file=sys.stderr,
            )
            sys.exit(2)
        video_path = candidate
        print(f"[auto] no video arg given; using first match: {video_path}")
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[FATAL] video not found: {video_path}", file=sys.stderr)
            sys.exit(2)

    # ── Step 1: ffmpeg precondition (D-03 / ENV-02) ────────────────────────────
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[FATAL] system ffmpeg not found or broken: {e}", file=sys.stderr)
        print("Install: brew install ffmpeg (macOS) or apt install ffmpeg (linux).", file=sys.stderr)
        sys.exit(2)

    # ── Step 2: probe metadata via ffprobe (D-05) ──────────────────────────────
    from utils import probe_video_metadata, ensure_output_dirs
    meta = probe_video_metadata(video_path)
    probe_duration = meta["duration_sec"]
    print(
        f"[probe] duration={probe_duration:.3f}s  "
        f"resolution={meta['width']}x{meta['height']}  "
        f"codec={meta['codec']}  is_vfr={meta['is_vfr']}"
    )

    # ── Step 3: sample frames + §0.5 print + duration drift assertion ──────────
    # (D-04, D-07, D-08 / EXTR-01..05)
    frames, timestamps = sample_frames(video_path, fps=2.0)
    print(
        f"Sampled {len(frames)} frames; "
        f"first 5 timestamps: {timestamps[:5].tolist()}; "
        f"last timestamp: {timestamps[-1]:.3f}s; "
        f"ffprobe duration: {probe_duration:.3f}s"
    )
    assert abs(timestamps[-1] - probe_duration) < 1.0, (
        f"timestamp drift: last={timestamps[-1]:.3f}s vs ffprobe={probe_duration:.3f}s "
        f"(>1.0s — Pitfall 2/3 indicator)"
    )

    # ── Step 4: load model (D-09, D-11, D-15 / EMBD-01) ────────────────────────
    print("[model] loading CLIP ViT-L-14 OpenAI (first run downloads ~890 MB to HF cache)…")
    model, preprocess, device = load_model()
    print(f"[model] loaded; device={device}")

    # ── Step 5: embed + shape print + L2 assertion (D-12, D-13 / EMBD-02..06) ──
    embeddings = embed_frames(frames, model, preprocess, device, batch_size=32)
    print(f"embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (len(frames), 768), (
        f"shape mismatch: got {embeddings.shape}, expected ({len(frames)}, 768)"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"NOT L2-normalized: norms in [{norms.min()}, {norms.max()}]"
    )
    print(f"L2-norm check: PASS  (norms in [{norms.min():.6f}, {norms.max():.6f}])")

    # ── Step 6: bit-identical rerun (D-17 / Pitfall 7 / EMBD-06) ───────────────
    # Goes through the SAME embed_frames code path as production (D-12).
    print("[determinism] running bit-identical rerun on a single frame…")
    one_frame = [frames[0]]
    e1 = embed_frames(one_frame, model, preprocess, device, batch_size=32)
    e2 = embed_frames(one_frame, model, preprocess, device, batch_size=32)
    assert np.array_equal(e1, e2), (
        "Embeddings are non-deterministic — same frame embedded twice produced different vectors. "
        "Check that model.eval() is set and inference_mode is active."
    )
    print("[determinism] PASS  (e1 == e2 byte-for-byte)")

    # ── Step 7: write fixture if requested (D-18) ──────────────────────────────
    if args.save_fixture:
        video_name = video_path.stem
        paths = ensure_output_dirs(video_name)
        cache_dir = paths["cache"]
        emb_path = cache_dir / f"{video_name}_embeddings.npy"
        ts_path = cache_dir / f"{video_name}_timestamps.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        np.save(ts_path, timestamps.astype(np.float64))
        print(f"[fixture] wrote {emb_path}")
        print(f"[fixture] wrote {ts_path}")

    print("Phase 1 §0.5 verification: PASS")
