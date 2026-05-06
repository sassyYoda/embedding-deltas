# ── D-68 / Pitfall 14: determinism env vars MUST be set BEFORE `import torch` ──
# extract.py imports torch at module top; `from extract import ...` triggers that
# transitive import. Therefore os.environ must be configured at THIS file's first
# lines, before ANY import statement (other than `os` itself). setdefault (not =)
# so user-set values are respected. CUBLAS_WORKSPACE_CONFIG is read at torch's
# CUDA initialization — too late if set after `import torch`.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

"""Phase 5: pipeline orchestrator (user-facing CLI per spec §9).

This module is the single user-facing entry point: `python pipeline.py videos/<name>.mp4`.
It owns three pitfalls (research/PITFALLS.md):
  - Pitfall 14: determinism env vars set BEFORE `import torch` (stanza at module top
    above this docstring; torch toggles in `if __name__ == "__main__":`).
  - Pitfall 16: JSON float rounding — 3 decimal places for *_sec fields, 4 decimal
    places for mad_score / raw_cosine_delta (helpers `_round_seconds` / `_round_score`
    applied at JSON-assembly time).
  - Pitfall 17: strict three-state `coincides_with_pelt_changepoint` — null when --pelt
    is OFF, true/false when ON, never omitted (helper `_compute_pelt_coincidence` +
    runtime assertion at JSON-write time).

Composition (D-65, D-66, D-67): `run()` calls Phase 1-4 module functions in order
and prints the 6 stage banners verbatim per spec §9. JSON manifest assembly follows
spec §8 field-for-field (D-69..D-73). The §0.5 end-to-end harness lives in
`if __name__ == "__main__":` and additionally applies torch's deterministic-algorithms
toggle (warn_only=True per RESEARCH/STACK.md Apple Silicon guidance).
"""

import argparse
import json
import sys
import time
from pathlib import Path

# `import torch` here is the documented exception in D-80 — needed because the
# `use_deterministic_algorithms` toggle fires inside `if __name__ == "__main__":`.
# All env-var setup that must precede this import is in the stanza at module top.
import torch

from extract import sample_frames, load_model, embed_frames
from signal_processing import (
    compute_deltas,
    smooth_deltas,
    mad_normalize,
    detect_changepoints,
    score_index_to_timestamp,
)
from clip_selection import (
    select_peaks,
    apply_pelt_boost,
    compute_budget_seconds,
    compute_padding,
    build_clips,
    merge_clips,
    enforce_budget,
)
from export import extract_clip, concat_clips
from utils import probe_video_metadata, ensure_output_dirs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _round_seconds(v: float) -> float:
    """D-71 / Pitfall 16: 3 decimal places (millisecond precision) for *_sec fields."""
    return round(float(v), 3)


def _round_score(v: float) -> float:
    """D-71 / Pitfall 16: 4 decimal places for mad_score and raw_cosine_delta."""
    return round(float(v), 4)


def _compute_pelt_coincidence(
    peak_idx: int,
    changepoints: list[int] | None,
    window: int = 5,
) -> bool | None:
    """D-72 / Pitfall 17 — strict three-state.

    Returns:
        None  when changepoints is None (i.e. --pelt was OFF).
        True  when peak_idx is within ±window samples of any changepoint.
        False otherwise (--pelt was on but no nearby changepoint).

    NEVER returns False when --pelt is off (Pitfall 17 violation).
    NEVER omits the field — caller stores result directly under the key.
    """
    if changepoints is None:
        return None
    return any(abs(peak_idx - cp) <= window for cp in changepoints)


def _build_manifest(
    *,
    video_path: Path,
    video_duration_sec: float,
    budget_sec: float,
    final_clips: list[tuple[float, float, float, float]],
    peak_time_to_idx: dict[float, int],
    scores,             # np.ndarray (N-1,) float64 — duck-typed (D-80 no numpy import here)
    raw_deltas,         # np.ndarray (N-1,) float64
    changepoints: list[int] | None,
) -> dict:
    """Assemble the spec §8 JSON manifest dict (D-69..D-73).

    Field order matches spec §8 verbatim; sort_keys=False preserves it on dump.
    Floats rounded at assembly time (D-71 / Pitfall 16).
    coincides_with_pelt_changepoint is strict three-state (D-72 / Pitfall 17).
    """
    clips_json: list[dict] = []
    for i, (start, end, score, peak_time) in enumerate(final_clips):
        # D-70: recover peak_idx via the dict built before build_clips.
        # peak_time is a Python float emitted by score_index_to_timestamp; the dict
        # was keyed by the same float so direct lookup works for non-merged clips.
        # For MERGED clips, peak_time is the higher-scoring source's peak_time
        # (Pitfall 19 / D-46) which is also a key in the dict.
        peak_idx = peak_time_to_idx[peak_time]
        mad_score = float(scores[peak_idx])
        raw_cosine_delta = float(raw_deltas[peak_idx])
        coincides = _compute_pelt_coincidence(peak_idx, changepoints, window=5)
        clips_json.append({
            "clip_index": i,
            "start_sec": _round_seconds(start),
            "end_sec": _round_seconds(end),
            "duration_sec": _round_seconds(end - start),
            "peak_timestamp_sec": _round_seconds(peak_time),
            "mad_score": _round_score(mad_score),
            "raw_cosine_delta": _round_score(raw_cosine_delta),
            "coincides_with_pelt_changepoint": coincides,
        })

    total_reel_duration_sec = sum(end - start for start, end, _, _ in final_clips)
    manifest = {
        "video": video_path.name,
        "video_duration_sec": _round_seconds(video_duration_sec),
        "budget_sec": _round_seconds(budget_sec),
        "total_reel_duration_sec": _round_seconds(total_reel_duration_sec),
        "embedding_model": "ViT-L-14 (OpenAI / QuickGELU)",  # D-69 — locked literal
        "sampling_fps": 2.0,                                   # D-69 — locked literal
        "clips": clips_json,
    }
    return manifest


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run(
    video_path: str,
    *,
    use_pelt: bool,
    height: float,
    min_gap_sec: float,
    merge_gap_sec: float,
) -> None:
    """Orchestrate the full pipeline on one video (D-65).

    Six stages with banners matching spec §9 verbatim (D-66).
    Output: reel + JSON + intermediate clips on disk.
    """
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem  # e.g. "justin_timberlake"

    # ORCH-03 / D-66: ensure output dirs exist on demand.
    paths = ensure_output_dirs(video_name)
    clips_dir = paths["clips"]            # output/clips/{video_name}/
    reels_dir = paths["reels"]            # output/reels/
    timestamps_dir = paths["timestamps"]  # output/timestamps/

    # Probe duration up front (single source of truth — utils.probe_video_metadata).
    meta = probe_video_metadata(video_path_obj)
    video_duration_sec = float(meta["duration_sec"])

    # ── Stage 1: sample frames ─────────────────────────────────────────────
    print("[1/6] Sampling frames at 2fps...")
    frames, timestamps = sample_frames(video_path_obj, fps=2.0)

    # ── Stage 2: CLIP embeddings ───────────────────────────────────────────
    print("[2/6] Extracting CLIP embeddings...")
    model, preprocess, device = load_model()
    embeddings = embed_frames(frames, model, preprocess, device, batch_size=32)

    # ── Stage 3: deltas (KEEP raw_deltas — Phase 5 uses it for raw_cosine_delta) ─
    print("[3/6] Computing embedding deltas...")
    raw_deltas = compute_deltas(embeddings)  # (N-1,) float64 — retained for JSON

    # ── Stage 4: smooth + MAD + (opt) PELT ─────────────────────────────────
    print("[4/6] Filtering and normalizing signal...")
    smoothed = smooth_deltas(raw_deltas)
    scores = mad_normalize(smoothed, window_samples=180)
    changepoints: list[int] | None = (
        detect_changepoints(smoothed, penalty=3.0) if use_pelt else None
    )

    # ── Stage 5: peaks → boost (opt) → budget → build → merge → enforce ────
    print("[5/6] Selecting clips...")
    peak_indices, peak_scores = select_peaks(
        scores, timestamps,
        height=height, min_gap_sec=min_gap_sec, fps=2.0,
    )
    if use_pelt and changepoints is not None:
        peak_indices, peak_scores = apply_pelt_boost(
            peak_indices, peak_scores, changepoints, window=5, boost=1.2,
        )

    # D-70: build {peak_time → peak_idx} reverse map BEFORE build_clips so the JSON
    # can recover scores[peak_idx] / raw_deltas[peak_idx] per FINAL clip. Each
    # peak_time is computed via score_index_to_timestamp(idx, timestamps), which is
    # what build_clips uses internally. After enforce_budget, each surviving clip's
    # peak_time is one of these keys (Pitfall 19 / D-46 — merge keeps the higher
    # score's peak_time, which is one of the source peaks' peak_times).
    peak_time_to_idx: dict[float, int] = {
        score_index_to_timestamp(int(idx), timestamps): int(idx)
        for idx in peak_indices
    }

    budget_sec = compute_budget_seconds(video_duration_sec)
    padding_sec = compute_padding(budget_sec)
    clips = build_clips(
        peak_indices, peak_scores, timestamps,
        video_duration_sec, padding_sec,
    )
    merged = merge_clips(clips, gap_threshold_sec=merge_gap_sec)
    final_clips = enforce_budget(merged, budget_sec)

    # ── Stage 6: extract clips, concat reel, write JSON ────────────────────
    print("[6/6] Exporting highlight reel...")
    clip_paths: list[Path] = []
    for i, (start, end, _, _) in enumerate(final_clips):
        out_clip = clips_dir / f"{i:03d}.mp4"
        extract_clip(video_path_obj, out_clip, start, end)
        clip_paths.append(out_clip)

    reel_path = reels_dir / f"{video_name}_highlight.mp4"
    concat_clips(
        [str(p) for p in clip_paths],
        str(reel_path),
        str(clips_dir),
    )

    # JSON-01..04: assemble + write spec §8 manifest.
    manifest = _build_manifest(
        video_path=video_path_obj,
        video_duration_sec=video_duration_sec,
        budget_sec=budget_sec,
        final_clips=final_clips,
        peak_time_to_idx=peak_time_to_idx,
        scores=scores,
        raw_deltas=raw_deltas,
        changepoints=changepoints,
    )
    # JSON-04 alignment-invariant defense in depth: assert at write time.
    for clip in manifest["clips"]:
        assert clip["start_sec"] <= clip["peak_timestamp_sec"] <= clip["end_sec"], (
            f"alignment invariant violated in JSON: {clip}"
        )
    # D-72 / Pitfall 17 strict three-state validation at write time.
    for clip in manifest["clips"]:
        assert "coincides_with_pelt_changepoint" in clip, (
            f"coincides_with_pelt_changepoint missing in clip: {clip}"
        )
        if not use_pelt:
            assert clip["coincides_with_pelt_changepoint"] is None, (
                f"--pelt off but coincides field is not None: {clip}"
            )
        else:
            assert isinstance(clip["coincides_with_pelt_changepoint"], bool), (
                f"--pelt on but coincides field is not bool: {clip}"
            )

    # D-73: indent=2, sort_keys=False (preserve spec §8 field order), trailing newline.
    json_path = timestamps_dir / f"{video_name}.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    # D-67: final user-facing line (matches spec §9 verbatim).
    print(f"Done. Reel: output/reels/{video_name}_highlight.mp4")


if __name__ == "__main__":
    # D-68 / Pitfall 14 finalize: env vars are already set at module top (lines above
    # the `import` block). Now apply the torch-side determinism toggles. warn_only=True
    # is mandatory on MPS (RESEARCH/STACK.md Apple Silicon section) — strict mode raises.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # D-64 / ORCH-01: argparse exposes EXACTLY the spec §9 surface — no extra flags.
    parser = argparse.ArgumentParser(
        description=(
            "Body Cam Highlight Reel pipeline (spec §9). Produces a highlight reel + "
            "spec §8 JSON manifest from one input video."
        ),
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--pelt", action="store_true", default=False,
        help="Enable PELT changepoint detection as supplementary signal",
    )
    parser.add_argument(
        "--height", type=float, default=1.5,
        help="Minimum MAD score to qualify as a peak",
    )
    parser.add_argument(
        "--min-gap-sec", type=float, default=15.0,
        help="Minimum seconds between selected peaks",
    )
    parser.add_argument(
        "--merge-gap-sec", type=float, default=3.0,
        help="Maximum gap between adjacent clips to merge",
    )
    args = parser.parse_args()

    # D-65 invocation.
    t0 = time.time()
    run(
        args.video,
        use_pelt=args.pelt,
        height=args.height,
        min_gap_sec=args.min_gap_sec,
        merge_gap_sec=args.merge_gap_sec,
    )
    elapsed = time.time() - t0
    print(f"[wall-clock] {elapsed:.2f}s")
