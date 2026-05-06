# Phase 1: Frame Extraction & Embeddings - Research

**Researched:** 2026-05-06
**Domain:** Implementation-detail gap-fill for `extract.py` + `utils.py` (Phase 1)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

Copied verbatim from `.planning/phases/01-frame-extraction-embeddings/01-CONTEXT.md` `<decisions>` block. The planner MUST honor these — research below resolves implementation details *within* these constraints, not alternatives to them.

**Project Skeleton & Dependencies**
- **D-01:** Project layout matches spec §1 verbatim — `pipeline.py`, `extract.py`, `signal_processing.py`, `clip_selection.py`, `export.py`, `utils.py`, `videos/`, `output/{reels,clips,timestamps}/`, `requirements.txt`. (ENV-03)
- **D-02:** `requirements.txt` uses pinned versions: `torch==2.11.0`, `torchvision==0.26.0`, `open_clip_torch==3.3.0`, `opencv-python==4.13.0.92`, `numpy>=2.1,<2.5`, `scipy==1.16.2`, `ruptures==1.1.10`, `tqdm>=4.67,<5`. **`ffmpeg-python` is NOT included.** Rationale lives as a comment at top of `requirements.txt`. (ENV-01)
- **D-03:** System `ffmpeg` precondition checked at start of `extract.__main__`: `subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)` wrapped in clear error message. Fail fast. (ENV-02)

**Frame Sampling Strategy**
- **D-04:** Frame seeking uses `cv2.VideoCapture` with `CAP_PROP_POS_MSEC` time-based seek, NOT `CAP_PROP_POS_FRAMES`. Per-frame timestamps recorded via `cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0`. (EXTR-03; Pitfall 2)
- **D-05:** Video duration sourced via `ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0`, NOT `cv2.CAP_PROP_FRAME_COUNT / fps`. `utils.probe_video_metadata` is the single owner. (EXTR-04; Pitfall 3)
- **D-06:** No upfront probe-then-switch logic. The `CAP_PROP_POS_MSEC` path is correct for both CFR and VFR; runs unconditionally.
- **D-07:** BGR→RGB conversion done inside `sample_frames` via `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` (Pitfall 1). Returned frame is RGB `uint8 (H, W, 3)`.
- **D-08:** §0.5 sanity assertion: `assert abs(timestamps[-1] - probe_duration) < 1.0`. Print: `"Sampled {N} frames; first 5 timestamps: {ts[:5]}; last timestamp: {ts[-1]:.3f}s; ffprobe duration: {probe_duration:.3f}s"`. (EXTR-05)

**Embedding Extraction**
- **D-09:** Model loaded via `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')`. Exact call, no substitution. QuickGELU forced by registry. (EMBD-01)
- **D-10:** Model loading factored into `extract.load_model() -> (model, preprocess, device)`, not done inside `embed_frames`. Caller passes model in.
- **D-11:** Device selection centralized in `load_model()`: `torch.device('cuda' if cuda else 'mps' if mps else 'cpu')`. Inference loop has no device-specific branches — `frames.to(device)` is the only place device shows up. (ENV-04)
- **D-12:** `embed_frames(frames, model, preprocess, device, batch_size=32)` is the **single batched code path**. PIL conversion via `Image.fromarray(rgb_array)` before `preprocess` (Pitfall 5). Inference wrapped in `model.eval()` + `torch.inference_mode()` (Pitfall 4). Batches of exactly 32 except trailing partial batch. (EMBD-02, EMBD-03)
- **D-13:** L2 normalization is **explicit and asserted**: `feats = feats / feats.norm(dim=-1, keepdim=True)`, then `assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5), "embeddings not L2-normalized"`. Hard assertion, not a print. (EMBD-05; Pitfall 6)
- **D-14:** Output embeddings: `np.ndarray` shape `(N, 768)`, dtype `float32`. (EMBD-04)
- **D-15:** Determinism precursors in Phase 1: `model.eval()`, `torch.inference_mode()`, single batched code path, `torch.backends.cudnn.benchmark = False` set inside `load_model()`. **No env vars set here** (Phase 5 owns the env-var stanza).

**Verification Harness**
- **D-16:** §0.5 verification lives in `extract.py`'s `if __name__ == "__main__":` block, runnable as `python extract.py <sample.mp4>`. Prints sampled-frame summary, embedding shape, L2-norm assertion result. Single video.
- **D-17:** Bit-identical-rerun check (Pitfall 7 closer): `__main__` block embeds the same single frame twice via `embed_frames(...)` and asserts byte-for-byte equality. (EMBD-06)

**Fixture Artifact for Downstream Phases**
- **D-18:** `__main__` verification block also writes `output/cache/{video_name}_embeddings.npy` and `output/cache/{video_name}_timestamps.npy` as side-effect (gated behind `--save-fixture` flag, default ON in Phase 1's `__main__`). Internal dev tool.
- **D-19:** **No user-facing embedding cache.** DIAG-02 v2 feature deferred entirely. The Phase 1 fixture writes are an internal dev tool only.

**`utils.py` API Surface (Phase 1 subset)**
- **D-20:** Phase 1 ships exactly these `utils.py` functions:
  - `probe_video_metadata(video_path: str | Path) -> dict` — `{"duration_sec": float, "width": int, "height": int, "codec": str, "is_vfr": bool}`. Uses `ffprobe -show_streams -show_format -of json`.
  - `ensure_output_dirs(video_name: str, base: Path = Path("output")) -> dict[str, Path]` — creates `reels/`, `clips/{video_name}/`, `timestamps/`, and `cache/` (for D-18).
  - `setup_logger(name: str = "highlight_reel") -> logging.Logger` — minimal stdlib logger; `print()` is also acceptable per spec — pick one and use consistently.
- **D-21:** `utils.write_timestamps_json` deferred to Phase 5.

### Claude's Discretion

- Logger vs `print()` for §0.5 verification prints — **default `print()` to match spec §0.5 verbatim**; switch to logger only if Phase 5 introduces structured logging.
- Exact wording of error messages, docstrings, type-hint style — match Python ≥3.11 conventions (PEP 604 unions like `str | Path`, lowercase `dict[str, Path]`).
- Tolerance constants (`atol=1e-5` for L2 norm, `< 1.0` for duration drift) — sensible defaults; planner/executor can tighten if a sample video reveals issues.

### Deferred Ideas (OUT OF SCOPE)

- Determinism env-var stanza (`OMP_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`) → Phase 5
- User-facing `--cache` flag for `pipeline.py` → v2 (DIAG-02)
- `utils.write_timestamps_json` → Phase 5
- `--diagnostics` plots → v2 (DIAG-01)
- Concat-filter re-encode fallback → Phase 4
- JSON float rounding → Phase 5
- Probe-then-switch seek strategy based on VFR detection — rejected (D-06)
- Module-level model loading per spec §2 verbatim — rejected (D-10)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | Pinned `requirements.txt` covering core deps; drop `ffmpeg-python` | §"Project Skeleton & Bootstrap" — exact `requirements.txt` content (Gap 7) |
| ENV-02 | Verify system `ffmpeg` on PATH at startup | §"ffmpeg precondition" — exact subprocess invocation, error message (Gap 7) |
| ENV-03 | Project layout matches spec §1 exactly | §"Skeleton creation order" — exact mkdir/touch sequence (Gap 7) |
| ENV-04 | Python 3.11+, no GPU-specific code paths beyond a single `device` constant | §"Device selection on MPS" — pattern from STACK.md (Gap 5) |
| EXTR-01 | `sample_frames(video_path, fps=2.0)` samples one frame per 0.5s | §"`cv2.VideoCapture` time-based seek" — exact loop skeleton (Gap 2) |
| EXTR-02 | Frames returned as RGB | D-07 + Pitfall 1 — `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` exactly once |
| EXTR-03 | Per-frame timestamps from actual playback position | §"`cv2.VideoCapture` time-based seek" — `cap.get(CAP_PROP_POS_MSEC)` after `cap.read()` returns ms of frame just read (Gap 2) |
| EXTR-04 | Duration from `ffprobe`, not `CAP_PROP_FRAME_COUNT` | §"`ffprobe` duration" — exact CLI: `ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 <path>` (Gap 3) |
| EXTR-05 | §0.5: print sampled frame count + first 5 timestamps; assert 0.5s spacing | §"Verification harness skeleton" — exact print/assert code (Gap 8) |
| EMBD-01 | `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')` — no substitution | §"Verified `open_clip` API" — exact 3-tuple return signature confirmed against 3.3.0 (Gap 1) |
| EMBD-02 | `model.eval()` + `torch.inference_mode()` wrap inference | §"Embedding loop skeleton" — exact context manager nesting (Gap 1) |
| EMBD-03 | Frames processed in batches of 32 | §"Embedding loop skeleton" — exact batching pattern with trailing partial batch (Gap 1, Gap 4) |
| EMBD-04 | Embeddings shape `(N, 768)` dtype `float32` | §"Verified `open_clip` API" — `encode_image` returns `(B, 768)`; `.cpu().numpy().astype(np.float32)` for final array (Gap 1) |
| EMBD-05 | Embeddings explicitly L2-normalized; hard assertion guards | §"Embedding loop skeleton" — explicit normalize either via `normalize=True` kwarg or post-call division; assert via `np.allclose` (Gap 1) |
| EMBD-06 | §0.5: embedding shape + L2-norm assertion both pass | §"Verification harness skeleton" — bit-identical-rerun (D-17) skeleton (Gap 8) |
</phase_requirements>

## Summary

Phase 1 is the gating phase for the project: produce a canonical `(timestamps, embeddings)` artifact from one body cam video while neutralizing the 8 frame/embedding pitfalls that could silently invalidate every downstream phase. The project-level research (`STACK.md`, `PITFALLS.md`, `ARCHITECTURE.md`) and CONTEXT.md (21 locked decisions) have already resolved the algorithmic, architectural, and dependency questions.

This research fills the remaining **implementation-detail gaps** the planner needs to write executable plans:
- Exact `open_clip` 3.3.0 API call signatures, return shapes, and the explicit-vs-`normalize=True`-kwarg L2 question (verified against current docs).
- Exact `cv2.VideoCapture` time-based seek loop semantics (when does `cap.get(CAP_PROP_POS_MSEC)` reflect the just-read frame vs the next frame).
- Exact `ffprobe` CLI invocations and parse strategy.
- Exact PIL → preprocess → batch shape pipeline including the `unsqueeze(0)` vs `torch.stack(...)` distinction.
- MPS-specific determinism caveats (what `cudnn.benchmark = False` analog applies).
- `np.save`/`np.load` round-trip considerations for the fixture artifact.
- Project bootstrap sequence (venv, install, ffmpeg precheck, gitignore additions, sample-vid extraction).
- Validation harness skeleton co-located in `__main__` (no pytest scaffolding per D-16).
- Sample video resolution: **only one video exists in `videos/sample-vids.zip`** — "Tiger Woods.mp4" (690 MB) — not five as roadmap implies. Flagged below.
- Inline `__main__` test pattern (no pytest, per D-16).

**Primary recommendation:** Implement `extract.py` and `utils.py` in this order — (a) ffmpeg precheck + `requirements.txt` + skeleton, (b) `utils.probe_video_metadata`, (c) `utils.ensure_output_dirs`, (d) `extract.sample_frames` with the exact `CAP_PROP_POS_MSEC` loop in §"cv2 time-based seek skeleton", (e) `extract.load_model` returning the 3-tuple, (f) `extract.embed_frames` with the exact PIL→preprocess→stack→inference→normalize→assert pipeline, (g) `__main__` block that wires all six together, runs the §0.5 prints, the bit-identical-rerun check, and writes the fixture. Use the **explicit post-call division** (`feats / feats.norm(dim=-1, keepdim=True)`) for L2 normalization — D-13 mandates the assertion, and explicit division reads more clearly than relying on the `normalize=True` kwarg.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Video frame I/O (decode, BGR→RGB, time-based seek) | `extract.py` | — | Owns `cv2.VideoCapture` per D-04/D-07; isolates OpenCV dep |
| Video metadata probing (duration, codec, is_vfr) | `utils.py` | — | Single source of truth via `ffprobe` (D-05/D-20); imported by `extract.py` `__main__` and Phase 5's `pipeline.run()` |
| CLIP model load + device selection | `extract.load_model()` | — | Centralizes `open_clip` + `torch` device pick (D-09/D-11); called once per process |
| CLIP inference (batched, normalized) | `extract.embed_frames()` | — | Single batched code path (D-12); receives model from caller |
| Output directory tree creation | `utils.ensure_output_dirs()` | — | Idempotent `mkdir -p` for `reels/clips/{name}/timestamps/cache/` (D-20) |
| Fixture write (npy artifacts for Phase 2/3 dev) | `extract.__main__` | `utils.ensure_output_dirs` | Side-effect of `--save-fixture` flag (D-18); never called from `pipeline.py` |
| §0.5 verification (prints + asserts) | `extract.__main__` | — | Inline per D-16; no separate `verify.py` |
| `ffmpeg -version` precondition check | `extract.__main__` | — | Phase 1's responsibility per D-03; Phase 5 will call again from `pipeline.run()` |

## Standard Stack

### Core (already locked by D-02; verified versions)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | `==2.11.0` | CLIP forward pass, tensor ops, MPS/CUDA dispatch | `[CITED: STACK.md / pypi.org/project/torch]` Latest stable; `open_clip_torch>=3.3.0` requires `torch>=2.6` |
| `torchvision` | `==0.26.0` | Image transforms used by `open_clip`'s preprocess | `[CITED: STACK.md]` Lockstep-pinned with torch 2.11.0; mismatch is #1 install-time CLIP failure |
| `open_clip_torch` | `==3.3.0` | CLIP ViT-L/14 OpenAI loader + preprocess transform | `[VERIFIED: ctx7 docs + STACK.md]` Canonical loader; `'ViT-L-14' + pretrained='openai'` registry forces `quick_gelu=True` |
| `opencv-python` | `==4.13.0.92` | Frame sampling via `VideoCapture` | `[CITED: STACK.md]` Spec §3 dependency; arm64 mac wheels published |
| `numpy` | `>=2.1,<2.5` | Embedding arrays, deltas | `[CITED: STACK.md]` scipy 1.16+ requires numpy 2.x; range allows transitive resolution |
| `scipy` | `==1.16.2` | (Phase 2) `medfilt`, `find_peaks`, `median_abs_deviation` | `[CITED: STACK.md + scipy issue #22333]` Pin away from 1.15 medfilt regression — included in Phase 1 `requirements.txt` even though only Phase 2 imports it |
| `ruptures` | `==1.1.10` | (Phase 2) PELT changepoint detection (lazy-imported) | `[CITED: STACK.md]` Latest; upper-bounds Python at <3.14 |
| `tqdm` | `>=4.67,<5` | Progress bars over embedding loop | `[CITED: STACK.md]` 4.67.x current line; standard library for batch progress |

### System Dependencies

| Tool | Required Version | Detected Locally | Purpose |
|------|------------------|------------------|---------|
| `ffmpeg` | `>=4.4` (recommend `>=6.0`) | `[VERIFIED: 7.1.1 via Homebrew]` | (Phase 4) Cut + concat; (Phase 1) precondition check + `ffprobe` for duration |
| `ffprobe` | Bundled with `ffmpeg` | `[VERIFIED: 7.1.1]` | Phase 1 video duration source (D-05) |
| Python | `>=3.11,<3.14` | `[VERIFIED: 3.13.2]` | `ruptures` upper-bounds at <3.14 |

### Removed From spec §1 (D-02)

| Listed in spec | Action | Reason |
|----------------|--------|--------|
| `ffmpeg-python` | **Drop** | `[VERIFIED: pypi.org/project/ffmpeg-python]` Last release 2019-07-06; spec §7 export code uses `subprocess.run` directly — binding never imported. Comment at top of `requirements.txt` to document the deviation. |

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
ffmpeg -version  # confirm system binary present
```

**Version verification** — before locking `requirements.txt`, the planner should NOT re-verify each version (already done in STACK.md as of 2026-05-06; pins are 6 months fresh). If install fails, escalate as a research flag rather than silently relaxing pins.

## Architecture Patterns

### System Architecture Diagram (Phase 1 only)

```
                  python extract.py <video.mp4> [--save-fixture]
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │  extract.__main__   │
                        │ (verification gate) │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────────┐
        │ 1. precondition          │                                  │
        ▼                          ▼                                  ▼
┌────────────────┐      ┌─────────────────────┐         ┌─────────────────────┐
│ subprocess.run │      │ utils.probe_video_  │         │ utils.ensure_output │
│ ffmpeg -version│      │ metadata(video)     │         │ _dirs(video_name)   │
│                │      │ → ffprobe -of json  │         │ → mkdir -p output/* │
└────────────────┘      └──────────┬──────────┘         └─────────────────────┘
                                   │ duration_sec
                                   ▼
                        ┌─────────────────────┐
                        │ extract.sample_     │
                        │ frames(video, 2.0)  │
                        │ ┌─────────────────┐ │
                        │ │ cv2.VideoCapture│ │
                        │ │ loop:           │ │
                        │ │  set POS_MSEC   │ │
                        │ │  read()         │ │
                        │ │  cvtColor BGR→  │ │
                        │ │   RGB           │ │
                        │ │  get POS_MSEC   │ │
                        │ │  step += 500ms  │ │
                        │ └─────────────────┘ │
                        └──────────┬──────────┘
                                   │ frames: list[ndarray RGB uint8 (H,W,3)]
                                   │ timestamps: ndarray (N,) float64
                                   ▼
                        ┌─────────────────────┐
                        │ extract.load_model()│
                        │ → open_clip.create_ │
                        │   model_and_        │
                        │   transforms        │
                        │ → device pick       │
                        │ → cudnn.benchmark   │
                        │   = False           │
                        │ → model.eval()      │
                        └──────────┬──────────┘
                                   │ (model, preprocess, device)
                                   ▼
                        ┌─────────────────────┐
                        │ extract.embed_      │
                        │ frames(...)         │
                        │ ┌─────────────────┐ │
                        │ │ for batch in    │ │
                        │ │  chunk(frames,  │ │
                        │ │   32):          │ │
                        │ │  PIL.fromarray  │ │
                        │ │  preprocess     │ │
                        │ │  torch.stack    │ │
                        │ │  inference_mode │ │
                        │ │  encode_image   │ │
                        │ │  feats / norm   │ │
                        │ │  cpu numpy f32  │ │
                        │ └─────────────────┘ │
                        └──────────┬──────────┘
                                   │ embeddings: ndarray (N, 768) float32
                                   ▼
                        ┌─────────────────────┐
                        │ §0.5 verification   │
                        │ - shape == (N, 768) │
                        │ - L2-norm assert    │
                        │ - bit-identical     │
                        │   rerun (D-17)      │
                        │ - duration drift    │
                        │   assert (D-08)     │
                        └──────────┬──────────┘
                                   │ if --save-fixture:
                                   ▼
                        ┌─────────────────────┐
                        │ np.save             │
                        │ output/cache/       │
                        │  {name}_embeddings  │
                        │  .npy + _timestamps │
                        │  .npy               │
                        └─────────────────────┘
```

### Recommended Project Structure (locked by D-01)

```
abel_police_interview/                    # repo root (already exists)
├── pipeline.py                           # stub for Phase 5 (just `if __name__ == "__main__": pass` is fine)
├── extract.py                            # Phase 1 owns
├── signal_processing.py                  # stub for Phase 2
├── clip_selection.py                     # stub for Phase 3
├── export.py                             # stub for Phase 4
├── utils.py                              # Phase 1 owns probe_video_metadata, ensure_output_dirs, setup_logger
├── requirements.txt                      # Phase 1 owns
├── videos/                               # gitignored; sample-vids.zip already here, needs unzip
│   └── Sample Videos/Tiger Woods.mp4     # ⚠ ONE video, not five — see §"Sample video for §0.5"
├── output/                               # gitignored
│   ├── reels/                            # mkdir at runtime via utils.ensure_output_dirs
│   ├── clips/                            # mkdir at runtime
│   ├── timestamps/                       # mkdir at runtime
│   └── cache/                            # mkdir at runtime (D-18 fixture writes)
├── .gitignore                            # Phase 1 extends
└── .planning/, assignment-details/, CLAUDE.md  # already exist
```

**Stub modules:** Phase 1 should create empty (or single-line) `signal_processing.py`, `clip_selection.py`, `export.py`, `pipeline.py` files so the layout matches spec §1 verbatim from day 1. Each stub: `"""Phase N — see ROADMAP.md."""` plus a `pass` or empty `if __name__ == "__main__":`. This satisfies ENV-03 without leaking work into other phases.

## Implementation-Detail Gaps Resolved

> This section answers the 10 specific gaps the orchestrator flagged. Each is the *exact* code skeleton, CLI command, or API call the planner will reference.

### Gap 1 — Verified `open_clip` 3.3.0 API call signatures and return shapes

**`create_model_and_transforms` returns a 3-tuple:** `(model, preprocess_train, preprocess_val)`. Spec §2 uses `model, _, preprocess = ...` — the `_` is `preprocess_train`, the third element `preprocess` is `preprocess_val`. **Use the third (val) preprocess for inference.** This is verified against `open_clip_torch==3.3.0` docs `[VERIFIED: ctx7 /mlfoundations/open_clip]`.

**`preprocess(pil_image)` returns** a `torch.Tensor` of shape `(3, 224, 224)` dtype `float32`, **already normalized** (CLIP mean/std, NOT 0–1) and **already resized + center-cropped**. **No batch dim.** Caller must add it via `.unsqueeze(0)` for single-image, OR `torch.stack([preprocess(pil) for pil in pils])` for batches `[VERIFIED: ctx7 docs]`.

**`model.encode_image(tensor_BCHW)` returns** a `torch.Tensor` of shape `(B, 768)` dtype `float32`. **Does NOT normalize by default** — must either pass `normalize=True` kwarg OR divide manually `[VERIFIED: ctx7 docs explicitly state "Whether to L2-normalize the output features. Defaults to False"]`.

**For Phase 1 (D-13 explicit-and-asserted):** Use the manual division pattern. The `normalize=True` kwarg is functionally equivalent but the manual division reads more clearly alongside the assertion:

```python
# inside embed_frames (Gap 4 has the full skeleton)
with torch.inference_mode():
    feats = model.encode_image(batch_tensor)              # (B, 768) NOT normalized
    feats = feats / feats.norm(dim=-1, keepdim=True)      # explicit L2 normalize
# move to numpy outside the inference_mode block
batch_np = feats.cpu().numpy().astype(np.float32)         # (B, 768) float32 on CPU
```

**Device handling:** `tensor.to(device)` works on the input batch; the model is moved to device once in `load_model()` via `model = model.to(device)`. `feats.cpu().numpy()` is required because numpy can't read MPS/CUDA tensors directly.

**Dtype:** open_clip's `preprocess` output is `float32`. `encode_image` output is `float32`. The final `.astype(np.float32)` is defensive belt-and-suspenders (catches the rare case where mixed precision or a GPU codepath returned `float16`).

### Gap 2 — Exact `cv2.VideoCapture` time-based seek loop semantics

**Critical semantic** verified against OpenCV documentation and the PITFALLS.md research:

1. `cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)` requests a seek to time `target_ms` (float, milliseconds).
2. `ret, frame = cap.read()` decodes the **next available frame at or after** the seek target.
3. `cap.get(cv2.CAP_PROP_POS_MSEC)` returns the **timestamp of the NEXT frame to be read** — i.e., **after** `cap.read()` succeeds, `cap.get(...)` returns the timestamp of the frame *following* the one just returned.

**Pitfall:** to record the timestamp of the frame *just read*, you must read `CAP_PROP_POS_MSEC` **before** calling `cap.read()`, OR subtract one frame interval from the post-read value. The simplest robust pattern: read the timestamp **immediately after** `cap.read()` (which gives the *next* frame's time, ≈ `actual_read_time + 1/source_fps`), and accept the ≈33ms offset as negligible at 2 fps sampling (where the sample interval is 500ms). The PITFALLS.md research recommends reading after `cap.read()`; the offset is well below the 1.0s tolerance in D-08.

**Recommended skeleton** for `extract.sample_frames`:

```python
import cv2
import numpy as np

def sample_frames(video_path: str | Path, fps: float = 2.0) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample frames at `fps` Hz via CAP_PROP_POS_MSEC. Returns (frames, timestamps_sec)."""
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
            break  # clean EOS detection
        # cap.get gives the time of the NEXT frame; for 2 fps sampling the offset is < 33ms
        # which is well within the 1.0s drift tolerance in D-08.
        actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # D-07 / Pitfall 1
        frames.append(frame_rgb)
        timestamps_ms.append(actual_ms)
        target_ms += sample_interval_ms

        # Defensive: if seek refuses to advance (some VFR pathologies), break to avoid infinite loop
        if len(timestamps_ms) >= 2 and timestamps_ms[-1] <= timestamps_ms[-2]:
            break

    cap.release()
    timestamps_sec = np.asarray(timestamps_ms, dtype=np.float64) / 1000.0
    return frames, timestamps_sec
```

**End-of-stream detection:** the `if not ret or frame_bgr is None: break` is the only reliable signal. Do NOT pre-compute iteration count from `CAP_PROP_FRAME_COUNT` (Pitfall 3) — that's why the loop is `while True` with break-on-False.

**Why both `not ret` and `is None`:** OpenCV builds vary; some return `(False, None)`, others return `(False, <empty array>)`. Guarding both is defensive.

### Gap 3 — Exact `ffprobe` CLI for duration

**Two CLI invocations needed in Phase 1:**

**(a) Bare duration (used internally by D-08 sanity check):**
```bash
ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 <video_path>
```
- `-v error` suppresses banner/info logs to keep stdout clean
- `-select_streams v:0` picks the first video stream (avoids picking up audio duration)
- `-show_entries format=duration` returns just the duration field
- `-of csv=p=0` outputs a single bare float, no header — easiest to parse
- Output: `"1842.300000\n"` — `float(stdout.strip())` works directly

**(b) Full metadata for `utils.probe_video_metadata` (D-20 returns dict with duration, width, height, codec, is_vfr):**
```bash
ffprobe -v error -show_streams -show_format -of json <video_path>
```
- Output: JSON with `streams[]` (per-stream codec_name, width, height, avg_frame_rate, r_frame_rate) and `format` (duration, format_name).
- Parse with `json.loads(stdout)`.
- `is_vfr`: detected by checking if `streams[0]['avg_frame_rate'] != streams[0]['r_frame_rate']` (avg differs from base when VFR). For body cams this is a useful diagnostic but does not change the seek strategy (D-06 — always use `CAP_PROP_POS_MSEC`).

**Exact `utils.probe_video_metadata` skeleton:**

```python
import json
import subprocess
from pathlib import Path

def probe_video_metadata(video_path: str | Path) -> dict:
    """Single source of truth for video metadata. Uses ffprobe (NOT cv2)."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe not found on PATH; install ffmpeg (which bundles ffprobe). "
            "On macOS: `brew install ffmpeg`."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {e.stderr.strip()}"
        ) from e

    data = json.loads(proc.stdout)
    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"no video stream found in {video_path}")

    v = video_streams[0]
    duration_sec = float(data["format"]["duration"])

    # is_vfr: avg_frame_rate differs from r_frame_rate when source is VFR
    avg_fr = v.get("avg_frame_rate", "0/0")
    r_fr = v.get("r_frame_rate", "0/0")
    is_vfr = avg_fr != r_fr  # diagnostic only; sampling strategy unchanged per D-06

    return {
        "duration_sec": duration_sec,
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "codec": v.get("codec_name", "unknown"),
        "is_vfr": is_vfr,
    }
```

**Error modes handled:**
- `ffprobe` not on PATH → clear error with install hint (FileNotFoundError on subprocess.run).
- File doesn't exist → clear error before invoking ffprobe.
- File exists but not a video → ffprobe nonzero return → CalledProcessError with stderr exposed.
- File is a video but has no video stream (audio-only) → explicit RuntimeError.

### Gap 4 — PIL conversion semantics: `unsqueeze(0)` vs `torch.stack`

**Per-image:** `preprocess(Image.fromarray(rgb))` returns shape `(3, 224, 224)` — **no batch dim**.

**For a single frame (e.g., the bit-identical-rerun check D-17):** wrap with `.unsqueeze(0)` to add the batch dim:
```python
single = preprocess(Image.fromarray(rgb)).unsqueeze(0)  # → (1, 3, 224, 224)
```

**For a batch of frames (the production path in D-12):** use `torch.stack` over a list comprehension:
```python
batch = torch.stack([preprocess(Image.fromarray(f)) for f in frame_batch])  # → (B, 3, 224, 224)
```

**Recommended `embed_frames` skeleton** — this is the single batched code path (D-12), used both for production embedding AND for the D-17 single-frame rerun check (which just calls `embed_frames([frame], ...)` returning `(1, 768)`):

```python
import numpy as np
import torch
from PIL import Image

def embed_frames(
    frames: list[np.ndarray],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed a list of RGB uint8 frames into (N, 768) float32 L2-normalized array.
    Single batched code path (D-12 / Pitfall 7). Caller passes loaded model (D-10).
    """
    if not frames:
        return np.zeros((0, 768), dtype=np.float32)

    model.eval()  # defensive — D-15 / Pitfall 4 (idempotent; safe to call repeatedly)

    out_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            # PIL conversion (Pitfall 5) — uses third element of create_model_and_transforms tuple
            tensors = [preprocess(Image.fromarray(f)) for f in chunk]
            batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 224, 224) on device

            # Shape sanity (Pitfall 5)
            assert batch_tensor.shape[1:] == (3, 224, 224), (
                f"preprocess produced wrong shape: {batch_tensor.shape}"
            )

            feats = model.encode_image(batch_tensor)  # (B, 768) float32 NOT normalized
            feats = feats / feats.norm(dim=-1, keepdim=True)  # D-13 explicit L2 norm

            out_chunks.append(feats.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(out_chunks, axis=0)  # (N, 768) float32

    # Hard assertion (D-13 / Pitfall 6) — NOT a print
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"embeddings not L2-normalized: norms in [{norms.min()}, {norms.max()}]"
    )
    return embeddings
```

**Trailing partial batch:** the `for start in range(0, len(frames), batch_size)` slicing handles this naturally — the last chunk is `frames[start:start+32]` which Python slices to whatever remains. No special-case needed. (D-12: "Batches of exactly 32 except the trailing partial batch.")

### Gap 5 — Determinism on MPS specifically

**The `cudnn.benchmark = False` setting (D-15) is a CUDA-only knob.** It is a no-op on MPS and CPU, but it's safe to call unconditionally. There is no MPS analog that needs setting in Phase 1 — MPS doesn't have a benchmark/auto-tune mechanism that perturbs first-batch behavior the way cuDNN does.

**`torch.use_deterministic_algorithms(True)` on MPS:** **DOES raise on MPS** for some kernels — `[CITED: pytorch.org docs]`. The escape hatch is `torch.use_deterministic_algorithms(True, warn_only=True)`, which converts the raise into a warning and falls back to the nondeterministic kernel. **However, this call belongs in Phase 5 (D-15 + ROBU-02), not Phase 1.** The CONTEXT.md explicitly defers env-vars + `use_deterministic_algorithms` to Phase 5.

**Phase 1's load_model determinism stanza:**

```python
import torch
import open_clip

def load_model() -> tuple:
    """Load CLIP ViT-L-14 OpenAI; return (model, preprocess, device).
    Centralizes device pick (D-11) and Phase-1 determinism precursors (D-15).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cpu"
    )

    # D-15 inference-side determinism precursors (env vars are Phase 5's job)
    torch.backends.cudnn.benchmark = False       # no-op on MPS/CPU; correct on CUDA
    torch.backends.cudnn.deterministic = True    # no-op on MPS/CPU; correct on CUDA

    # D-09: exact spec call, no substitution. Returns 3-tuple (model, preprocess_train, preprocess_val)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model = model.to(device)
    model.eval()  # D-15 / Pitfall 4 — train-mode-by-default per open_clip README

    return model, preprocess, device
```

**`torch.manual_seed(0)` and `np.random.seed(0)`:** Phase 1 inference involves no stochasticity (no dropout in eval mode, no random augmentation in val preprocess). Seeds belong in Phase 5 (`pipeline.run` entry) where they protect any future random operation. **Do NOT set seeds in `load_model()`** — would cross the Phase 1/Phase 5 boundary.

**MPS warm-up** (Pitfall 15): the bit-identical-rerun check (D-17) implicitly serves as a warm-up — by the time the second `embed_frames` call runs, MPS is fully initialized. No explicit warm-up needed in Phase 1. Phase 6's batch driver may want to pre-call `_ = model.encode_image(torch.zeros(1, 3, 224, 224, device=device))` once before the first real video, but that's Phase 6's concern.

### Gap 6 — `np.save` / `np.load` round-trip for the fixture artifact (D-18)

**Format:** `np.save` writes `.npy` binary format. For `(3600, 768) float32` ≈ 11 MB per video, **uncompressed is fine** — no need for `np.savez_compressed`. Compression saves ~30–50% but adds load latency and a manual `arr = np.load(...)['arr_0']` indirection.

**Round-trip is bit-exact for float32** as long as you don't change dtype. Watch:
- Always pass `np.ndarray` explicitly to `np.save` (don't accidentally save a torch tensor — it would pickle, slower load).
- Use `allow_pickle=False` on load for safety: `np.load(path, allow_pickle=False)` — refuses if anyone slipped a Python object in.

**Exact fixture-write skeleton** (called from `extract.__main__` when `--save-fixture` is set):

```python
def write_fixture(cache_dir: Path, video_name: str, embeddings: np.ndarray, timestamps: np.ndarray) -> tuple[Path, Path]:
    """Write fixtures for Phase 2/3/4 dev. Internal dev tool only (D-18/D-19)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / f"{video_name}_embeddings.npy"
    ts_path = cache_dir / f"{video_name}_timestamps.npy"
    np.save(emb_path, embeddings.astype(np.float32))   # defensive dtype lock
    np.save(ts_path, timestamps.astype(np.float64))    # timestamps are float64 per ARCHITECTURE.md
    return emb_path, ts_path
```

**Naming convention (path-safety):** `Path(video_path).stem` may contain spaces ("Tiger Woods" — see Gap 9). The fixture filenames will inherit those spaces. Two options:
- (A) Pass the raw stem through (tolerable; npy paths support spaces). Risk: shell tab-completion friction during dev.
- (B) Replace spaces with underscores: `video_name = Path(video_path).stem.replace(" ", "_")`.

**Recommend (A)** for Phase 1: Phase 4 will sanitize stems for ffmpeg manifest paths (Pitfall 13); Phase 1's npy fixture is read directly by Python with no shell intermediary. Inconsistency is acceptable as long as `extract.__main__`'s success message prints the full path so the user knows what to expect.

**`.gitignore` addition** (see Gap 7): `output/cache/` must be ignored — these are dev artifacts, never committed.

### Gap 7 — Project skeleton creation order + bootstrap commands

The exact sequence the planner should encode as task steps:

**Step 1: Bootstrap virtualenv** (one-time, do not commit `.venv/`):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Step 2: Write `requirements.txt`** with pinned versions and a comment documenting the `ffmpeg-python` exclusion:
```text
# Body Cam Highlight Reel — locked dependencies (verified 2026-05-06)
# Note: spec §1 lists `ffmpeg-python` but it is dropped per research/STACK.md —
# the project is unmaintained since 2019 and spec §7 uses subprocess.run directly.
# scipy pinned to 1.16.2 to dodge the 1.15 medfilt regression (scipy issue #22333).

torch==2.11.0
torchvision==0.26.0
open_clip_torch==3.3.0

opencv-python==4.13.0.92

numpy>=2.1,<2.5
scipy==1.16.2
ruptures==1.1.10

tqdm>=4.67,<5
```

**Step 3: Install dependencies** (this is the long step; ~1.3 GB download for torch+opencv+open_clip):
```bash
pip install -r requirements.txt
```

**Step 4: Verify ffmpeg system dep** (precondition gate for everything):
```bash
ffmpeg -version    # must succeed; verified locally as 7.1.1
ffprobe -version   # must succeed; bundled with ffmpeg
```

**Step 5: Extend `.gitignore`** — current state is just `.DS_Store`. Add:
```
# Python
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Project artifacts
output/
videos/

# Editor
.vscode/
.idea/
```
Note: `videos/` is ignored so the 690 MB sample stays out of git; `output/` is ignored because all generated artifacts (reels, clips, JSON, npy fixtures) live there.

**Step 6: Extract the sample video archive** (gitignored, so this is a per-clone dev step — document in README later):
```bash
cd videos
unzip sample-vids.zip
# Produces: videos/Sample Videos/Tiger Woods.mp4 (690 MB)
cd ..
```
**⚠ See Gap 9** — the zip contains ONE video, not five. This is a Phase 6 / project-scope concern but the planner should be aware now.

**Step 7: Create module stubs** to satisfy ENV-03 (project layout matches spec §1 verbatim):
```bash
touch pipeline.py signal_processing.py clip_selection.py export.py
# Each gets a one-line docstring placeholder; no implementation in Phase 1
```
Each stub should look like:
```python
"""TODO: implemented in Phase N — see ROADMAP.md."""
```

**Step 8: Implement `utils.py`** (D-20 functions; see Gap 3 skeleton for `probe_video_metadata`).

**Step 9: Implement `extract.py`** (load_model, sample_frames, embed_frames, `__main__`).

**Step 10: Run the §0.5 verification** (Gap 8 has the exact `__main__` skeleton):
```bash
python extract.py "videos/Sample Videos/Tiger Woods.mp4" --save-fixture
```
Note the quoted path — the sample filename has a space.

**Step 11: Commit** the source files (NOT the venv, NOT the output, NOT the videos):
```bash
git add requirements.txt extract.py utils.py pipeline.py signal_processing.py clip_selection.py export.py .gitignore
```

### Gap 8 — Verification harness skeleton (inline `__main__`, no pytest)

D-16 locks the harness as `if __name__ == "__main__":` in `extract.py`. **No pytest scaffolding** — neither now nor later in Phase 1. The harness must:
1. Run the ffmpeg precondition (D-03).
2. Probe metadata (D-05).
3. Sample frames (D-04, D-07).
4. Print the §0.5 frame summary (D-08).
5. Assert duration drift `< 1.0` (D-08).
6. Load model (D-09, D-11, D-15).
7. Embed frames (D-12, D-13).
8. Print embedding shape (`(N, 768)`).
9. Hard assertion on L2-norm (D-13).
10. Bit-identical-rerun check (D-17).
11. Optionally write fixture (D-18, default ON).

**Exact `__main__` skeleton:**

```python
# at bottom of extract.py
if __name__ == "__main__":
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Phase 1 §0.5 verification + fixture writer")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--save-fixture", action=argparse.BooleanOptionalAction, default=True,
        help="Write embeddings.npy + timestamps.npy to output/cache/ (default: ON, internal dev tool)"
    )
    args = parser.parse_args()
    video_path = Path(args.video)

    # ── Step 1: ffmpeg precondition (D-03) ────────────────────────────────────────
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[FATAL] system ffmpeg not found or broken: {e}", file=sys.stderr)
        print("Install: brew install ffmpeg (macOS) or apt install ffmpeg (linux).", file=sys.stderr)
        sys.exit(2)

    # ── Step 2: probe metadata (D-05) ─────────────────────────────────────────────
    from utils import probe_video_metadata, ensure_output_dirs
    meta = probe_video_metadata(video_path)
    probe_duration = meta["duration_sec"]
    print(f"[probe] duration={probe_duration:.3f}s  resolution={meta['width']}x{meta['height']}  "
          f"codec={meta['codec']}  is_vfr={meta['is_vfr']}")

    # ── Step 3 + 4: sample frames + §0.5 print (D-04, D-07, D-08) ─────────────────
    frames, timestamps = sample_frames(video_path, fps=2.0)
    print(
        f"Sampled {len(frames)} frames; "
        f"first 5 timestamps: {timestamps[:5].tolist()}; "
        f"last timestamp: {timestamps[-1]:.3f}s; "
        f"ffprobe duration: {probe_duration:.3f}s"
    )
    # ── Step 5: duration drift assert (D-08) ──────────────────────────────────────
    assert abs(timestamps[-1] - probe_duration) < 1.0, (
        f"timestamp drift: last={timestamps[-1]:.3f}s vs ffprobe={probe_duration:.3f}s"
    )

    # ── Step 6: load model (D-09, D-11, D-15) ─────────────────────────────────────
    print("[model] loading CLIP ViT-L-14 OpenAI (first run downloads ~890 MB to HF cache)…")
    model, preprocess, device = load_model()
    print(f"[model] loaded; device={device}")

    # ── Step 7 + 8 + 9: embed + shape print + L2 assertion (D-12, D-13) ───────────
    embeddings = embed_frames(frames, model, preprocess, device, batch_size=32)
    print(f"embeddings shape: {embeddings.shape}")  # → (N, 768)
    assert embeddings.shape == (len(frames), 768), (
        f"shape mismatch: got {embeddings.shape}, expected ({len(frames)}, 768)"
    )
    # L2 assertion is also inside embed_frames; this is a redundant guard at the
    # verification gate boundary so a §0.5 reader sees the check explicitly.
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"NOT L2-normalized: norms in [{norms.min()}, {norms.max()}]"
    )
    print(f"L2-norm check: PASS  (norms in [{norms.min():.6f}, {norms.max():.6f}])")

    # ── Step 10: bit-identical rerun (D-17 / Pitfall 7) ──────────────────────────
    print("[determinism] running bit-identical rerun on a single frame…")
    one_frame = [frames[0]]
    e1 = embed_frames(one_frame, model, preprocess, device, batch_size=32)
    e2 = embed_frames(one_frame, model, preprocess, device, batch_size=32)
    assert np.array_equal(e1, e2), (
        "Embeddings are non-deterministic — same frame embedded twice produced different vectors. "
        "Check that model.eval() is set and inference_mode is active."
    )
    print(f"[determinism] PASS  (e1 == e2 byte-for-byte)")

    # ── Step 11: write fixture (D-18, default ON) ─────────────────────────────────
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
```

**Why this lives in `__main__` and not pytest:** D-16 locked it. The harness reads top-to-bottom as the spec §0.5 protocol; pytest would split it across files. Phase 5 may introduce pytest later for end-to-end repro testing — Phase 1 must NOT. **Planner directive:** do not add `pytest`, `pytest.ini`, `conftest.py`, `tests/`, or any `def test_*` functions in Phase 1.

### Gap 9 — Sample video for §0.5 verification

**Critical finding:** `videos/sample-vids.zip` (690 MB) contains **exactly ONE video**, not five:

```
Archive:  videos/sample-vids.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
690164228  05-03-2026 10:37   Sample Videos/Tiger Woods.mp4
```

This contradicts the project-level docs (PROJECT.md, REQUIREMENTS.md RUN-01..03, ROADMAP.md Phase 6) that consistently reference "five sample videos". **This is a project-scope concern but the planner needs to know now** so Phase 1's §0.5 verification can pick the right file.

**Resolution for Phase 1:**
- The single video file is `videos/Sample Videos/Tiger Woods.mp4` (note the space in both folder and filename).
- The planner should sequence an "unzip step" before the §0.5 run: `cd videos && unzip -n sample-vids.zip` (the `-n` flag means "never overwrite" — safe to re-run).
- Run the §0.5 verification against this one file: `python extract.py "videos/Sample Videos/Tiger Woods.mp4" --save-fixture`.
- The 690 MB file is large enough that running the full §0.5 (which embeds all sampled frames) will take **5–15 minutes on M-series CPU, ~2–4 minutes on MPS**. The bit-identical-rerun check is fast (one extra frame). Plan for this wall-clock duration in any time-sensitive task estimates.

**`output/cache/` policy:** the npy fixture written by `--save-fixture` lives in `output/cache/` which is gitignored (Gap 7). **Do NOT commit the npy files** — they're 11 MB each and dev-only. The locked CONTEXT.md doesn't say to commit them; gitignoring `output/` covers it.

**Project-scope flag (NOT Phase 1's job to fix):** RUN-01, RUN-02, RUN-03, DOC-01, DOC-03 in Phase 6 assume five videos. With only one available, those requirements need to be re-discussed in Phase 6's `/gsd-discuss-phase`. The planner for Phase 1 should:
- Note this discrepancy in the plan's "Risks / Open Questions" section.
- NOT block on it — Phase 1 only needs *one* video to satisfy its success criteria.
- NOT attempt to rename "five" to "one" elsewhere — out of scope for this phase.

Possible explanations (for the project-level next discussion, NOT Phase 1):
- The zip was incomplete / a fresh upload is pending.
- "Sample-vids" is a misnomer and the assignment intended one video as a working sample.
- Other videos exist elsewhere (Drive link) but weren't downloaded.

### Gap 10 — Tests vs verification scripts: NO pytest in Phase 1

**Locked by D-16:** `if __name__ == "__main__":` block in `extract.py`, runnable as `python extract.py <video>`. The §0.5 verification IS the test. The bit-identical-rerun (D-17) IS the determinism test.

**Planner directives:**
- Do NOT add `pytest` to `requirements.txt`.
- Do NOT create `tests/`, `test_extract.py`, `conftest.py`, or any `def test_*` functions.
- Do NOT introduce a `verify.py` orchestrator (rejected in Area F of DISCUSSION-LOG).
- Do NOT split the `__main__` block into helper modules — keep it monolithic so a §0.5 reader sees the full verification top-to-bottom.

**If a future phase wants pytest** (Phase 5 might, for the two-run-byte-identical-JSON acceptance test), that's Phase 5's choice — Phase 1 has no opinion and should leave the door open by not adding any anti-pytest constraints (e.g., don't write code that depends on argparse `__main__` import-time side effects).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Video duration calculation | `cv2.CAP_PROP_FRAME_COUNT / fps` math | `ffprobe -show_entries format=duration` | Pitfall 3 — `CAP_PROP_FRAME_COUNT` lies on body-cam MP4s |
| L2 normalization | `feats / np.linalg.norm(...)` from scratch on numpy | `feats / feats.norm(dim=-1, keepdim=True)` (torch) OR `model.encode_image(x, normalize=True)` | Both keep the op on GPU/MPS until after; numpy round-trip first wastes time and triggers Pitfall 7 (precision drift) |
| PIL conversion | Manual `torch.from_numpy(rgb).permute(2,0,1).float() / 255.` | `Image.fromarray(rgb)` → `preprocess(...)` | Pitfall 5 — manual tensorization silently bypasses Resize/CenterCrop, embeddings degrade |
| Device selection | `os.environ.get("DEVICE", "cpu")` env-var hack | One-line ladder: `cuda if cuda else mps if mps else cpu` | Spec §1 forbids GPU-specific code paths; the ladder is a single `device` constant that falls through (STACK.md Mac/Apple Silicon) |
| Frame batching | Manual `while idx < N: ...` with index math | `for start in range(0, len(frames), batch_size): chunk = frames[start:start+batch_size]` | Slicing handles trailing partial batch automatically; off-by-one bugs eliminated |
| Output directory creation | Inline `os.makedirs(...)` scattered across modules | Single `utils.ensure_output_dirs(video_name)` returning a paths dict | D-20 — single source of truth; Phase 5's `pipeline.run()` and Phase 1's `__main__` both use it |
| Subprocess error handling | Bare `subprocess.run(...)` with no error context | `subprocess.run(..., check=True, capture_output=True)` wrapped in try/except with `e.stderr.decode()` exposed in the raise | `check=True` alone gives a useless `CalledProcessError` without the stderr; user can't debug ffmpeg failures |
| ffmpeg-python wrapper for the precondition check | `import ffmpeg; ffmpeg.probe(...)` | `subprocess.run(["ffmpeg", "-version"], ...)` | D-02 dropped `ffmpeg-python`; spec §7 already uses subprocess directly |

**Key insight:** the entire phase's complexity is in *correctly using* opencv + open_clip + ffprobe. Hand-rolling alternatives to any of these (custom video readers, custom CLIP loading, custom video metadata parsers) is the path to silent correctness bugs.

## Common Pitfalls (Phase 1 owned: 1, 2, 3, 4, 5, 6, 7, 15)

These are summarized from `.planning/research/PITFALLS.md` — the planner should reference that file for full detail. Phase 1 owns these by name; verification steps appear in the `__main__` block (Gap 8).

### Pitfall 1: BGR/RGB swap missing or doubled
**Guard in code:** `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` happens **exactly once** inside `sample_frames` (D-07). The contract returned to callers documents "RGB uint8 (H, W, 3)" so downstream never re-converts.
**Verification:** Optional channel-swap A/B test — embed a frame and `frame[:, :, ::-1]`, expect cosine sim 0.85–0.95. Not in Phase 1's mandatory §0.5; document as "if reels look off, run this check first."

### Pitfall 2: VFR seek inaccuracy via `CAP_PROP_POS_FRAMES`
**Guard in code:** D-04 mandates `CAP_PROP_POS_MSEC` only. Gap 2 has the exact loop.
**Verification:** D-08's `assert abs(timestamps[-1] - probe_duration) < 1.0`. Spot-check the first 5 timestamps in the §0.5 print are `≈[0.0, 0.5, 1.0, 1.5, 2.0]` (with sub-50ms drift acceptable from the post-`cap.read()` `cap.get(...)` semantics).

### Pitfall 3: `CAP_PROP_FRAME_COUNT` lies; loop terminates early or runs past EOF
**Guard in code:** Loop is `while True: ... if not ret: break` (Gap 2). Duration sourced from `ffprobe` only (D-05, Gap 3).
**Verification:** D-08 print shows `ffprobe duration` alongside `last timestamp` — eyeball check on every run.

### Pitfall 4: `model.eval()` forgotten; non-deterministic embeddings
**Guard in code:** `model.eval()` is called in `load_model()` (Gap 5) AND defensively at the top of `embed_frames` (Gap 4). Idempotent; cheap.
**Verification:** Bit-identical-rerun check (D-17, Gap 8 step 10).

### Pitfall 5: `preprocess` applied to wrong type (ndarray vs PIL)
**Guard in code:** `Image.fromarray(rgb)` wrapping is **inside the batch loop** of `embed_frames` (Gap 4). Shape assertion `batch_tensor.shape[1:] == (3, 224, 224)` fires per batch.
**Verification:** Per-batch shape assertion in `embed_frames` (Gap 4).

### Pitfall 6: L2 normalization assumed but not enforced
**Guard in code:** D-13 mandates explicit `feats / feats.norm(...)` AND a hard `np.allclose(...)` assertion. Both inside `embed_frames` AND inside `__main__` (defense in depth — Gap 4 + Gap 8 step 9).
**Verification:** The assertion IS the verification.

### Pitfall 7: Batch vs single-frame inference produces different embeddings
**Guard in code:** D-12 mandates a single batched code path; even the bit-identical-rerun check uses `embed_frames([frame], ...)`, not a special single-frame function.
**Verification:** Bit-identical-rerun check (D-17, Gap 8 step 10) — passes only if the single batched path is consistent.

### Pitfall 15: Lazy GPU init perturbs first batch
**Guard in code:** `cudnn.benchmark = False` set in `load_model()` (Gap 5). The bit-identical-rerun check after the main embedding loop implicitly serves as warm-up validation — if first-batch ≠ steady-state, the rerun would catch it because rerun is "after warm-up" and main is "during warm-up".
**Verification:** Implicit via D-17's bit-identical check passing across the batched/single boundary.

## Code Examples

> All skeletons are presented in Gap 1–8 above with full source-level detail. This section cross-references the canonical examples for the planner's quick lookup.

| Operation | Reference |
|-----------|-----------|
| Frame sampling loop | Gap 2 — `sample_frames` skeleton |
| `ffprobe` metadata invocation | Gap 3 — `probe_video_metadata` skeleton |
| Single-frame preprocess+embed | Gap 4 — `embed_frames` skeleton (called with `[frame]`) |
| Batched preprocess+embed | Gap 4 — `embed_frames` skeleton (called with full frames list) |
| Model load + device pick + determinism precursors | Gap 5 — `load_model` skeleton |
| Fixture write (.npy round-trip) | Gap 6 — `write_fixture` snippet |
| `__main__` verification harness | Gap 8 — full skeleton |
| Project bootstrap commands | Gap 7 — Steps 1–11 |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `model.encode_image(x)` returns normalized features | `model.encode_image(x)` returns RAW; `normalize=True` kwarg or manual division for L2 | open_clip 1.x → 2.x → 3.x | D-13's explicit normalization is correct; relying on the model is wrong |
| `torch.jit.load` for OpenAI CLIP weights | HF Hub download to `~/.cache/huggingface/hub/timm--vit_large_patch14_clip_224.openai/` | open_clip 3.x | First-run downloads ~890 MB; user-visible only as a one-time wait |
| `ffmpeg-python` (kkroening) wrapper | Direct `subprocess.run(["ffmpeg", ...])` and `subprocess.run(["ffprobe", ...])` | 2019-07 (kkroening abandoned) | D-02 dropped from `requirements.txt`; spec §7 already uses subprocess |
| `scipy.signal.medfilt` baseline (Phase 2) | `scipy.signal.medfilt` on scipy `1.16.x` (1.15 has correctness regression) | scipy 1.15 → 1.16.2 | Pin in `requirements.txt`; affects Phase 2 not Phase 1, but Phase 1 ships the lock |

**Deprecated/outdated (Phase 1 specific):**
- `cv2.CAP_PROP_POS_FRAMES` for sampling — superseded by `CAP_PROP_POS_MSEC` for VFR robustness (D-04).
- `cv2.CAP_PROP_FRAME_COUNT / fps` for duration — superseded by `ffprobe` (D-05).
- Module-level `model = open_clip.create_model_and_transforms(...)` (spec §2 verbatim) — superseded by `load_model()` factory function (D-10) so multi-video Phase 6 amortizes the ~30s load.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cap.get(CAP_PROP_POS_MSEC)` after `cap.read()` returns the timestamp of the *next* frame, not the just-read frame | Gap 2 | Recorded timestamps shift by ≈1 source-frame interval (≈33ms). Below the 1.0s D-08 tolerance. The downstream impact is invisible at 2 fps sampling. **`[ASSUMED]`** based on widely-reported OpenCV behavior; not re-verified against opencv-python 4.13.0.92 in this session. **Mitigation:** the §0.5 print includes both `last timestamp` and `ffprobe duration` — eyeball check at every run will surface discrepancies. |
| A2 | `torch.use_deterministic_algorithms(True)` raises on MPS for some kernels; `warn_only=True` is the escape | Gap 5 | If MPS supports full determinism without the escape, Phase 5's stanza is conservative but still correct. If MPS raises and Phase 5 uses the bare form (no `warn_only`), Phase 5 will crash on the `[5/6]` JSON-emit step. **`[ASSUMED]`** — Phase 5's research will re-verify. |
| A3 | Phase 1 `requirements.txt` should include `scipy==1.16.2` and `ruptures==1.1.10` even though they're Phase 2 deps | §"Standard Stack" | If we ship a Phase-1-only `requirements.txt`, Phase 2 has to extend it — minor friction. Including Phase 2 deps now matches CONTEXT.md D-02 which pins them all together; the planner may safely ship the full set. Risk: low; the `pip install` will work either way. **`[ASSUMED]`** D-02 wants the full set in Phase 1 (its phrasing "uses the research-pinned versions: ..." lists all of them). |
| A4 | The single video in `videos/sample-vids.zip` ("Tiger Woods.mp4", 690 MB) is sufficient for §0.5 verification | Gap 9 | If the file is corrupt or not a valid mp4, Phase 1 cannot complete §0.5. **`[ASSUMED]`** — file existence and size are verified; codec validity is not (would require running `ffprobe` against it, which the planner's first task does anyway). |
| A5 | Bit-identical-rerun (D-17) on a single frame *re-fed through the same batched path* will pass on MPS without warm-up | Pitfall 15 / Gap 8 step 10 | If MPS first-batch differs from second-batch by 1e-6 (Pitfall 15 territory), `np.array_equal(e1, e2)` fails on `assert`. Recovery: explicit warm-up before the production embedding (one-line addition: `_ = model.encode_image(torch.zeros(1,3,224,224,device=device))` after `load_model()`). **`[ASSUMED]`** — the production `embed_frames` call before the rerun *is* the warm-up, so this should hold. If it fails, add the explicit warm-up step. |
| A6 | The `videos/Sample Videos/Tiger Woods.mp4` filename with spaces will be handled correctly by `cv2.VideoCapture(str(path))`, `subprocess.run(["ffprobe", ..., str(path)])`, and `np.save(path)` | Gap 9 | All three accept paths with spaces because they're passed as discrete argv elements (subprocess) or strings (cv2/numpy), not shell-interpolated. **`[VERIFIED: standard library behavior]`** — but worth noting because Phase 4's ffmpeg manifest path (Pitfall 13) DOES need quote-sanitization. |

## Open Questions

1. **Are there 4 more sample videos somewhere?**
   - What we know: `videos/sample-vids.zip` contains exactly one video; project docs assume five.
   - What's unclear: whether the zip is incomplete, the project scope is actually one video, or other videos exist elsewhere (e.g., a Drive link mentioned in PROJECT.md).
   - Recommendation: **Phase 1 ignores this** (one video is sufficient for §0.5). Surface it as a Phase 6 / project-level discussion item via the plan's "Risks" section. Phase 6's `/gsd-discuss-phase` should re-validate RUN-01..03 / DOC-01 / DOC-03 against actual available videos.

2. **MPS bit-identical guarantee for D-17?**
   - What we know: Apple's MPS backend has had reproducibility issues in past PyTorch versions; the current torch 2.11 + macOS 15.3.2 combo is reportedly stable (STACK.md).
   - What's unclear: whether running the same single-frame `embed_frames` call twice within the same process produces *byte-identical* embeddings on MPS. CPU is bit-exact; CUDA is bit-exact under the determinism stanza; MPS is *probably* bit-exact post-warm-up.
   - Recommendation: **Run the §0.5 harness early in execution.** If D-17's `np.array_equal` assertion fires, add the explicit MPS warm-up (Gap 5 / A5) and re-run. If it still fires, fall back to `np.allclose(e1, e2, atol=1e-6)` as a relaxed check and document the residual nondeterminism in the plan's Risks (this contradicts D-17's "byte-for-byte" wording so it would need a CONTEXT.md amendment).

3. **Should the §0.5 print use `print()` or the `setup_logger` helper from D-20?**
   - What we know: D-20 ships `setup_logger`; spec §0.5 snippets use `print()`; CONTEXT.md "Claude's Discretion" defaults to `print()` for spec-verbatim feel.
   - What's unclear: whether Phase 5's structured logging (if any) will retroactively want `extract.py`'s prints converted.
   - Recommendation: **Use `print()` in `extract.__main__`.** D-20's logger is available for Phase 5 to opt into; not a Phase 1 concern.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | runtime | ✓ | `[VERIFIED: 3.13.2]` (within `>=3.11,<3.14`) | — |
| `ffmpeg` system binary | D-03 precondition; (Phase 4) export | ✓ | `[VERIFIED: 7.1.1]` (Homebrew, well above 4.4 minimum) | — |
| `ffprobe` (bundled with ffmpeg) | D-05 duration source | ✓ | `[VERIFIED: 7.1.1]` | — |
| `pip` / `venv` | bootstrap | ✓ | (stdlib) | `uv pip install` ~10× faster but standard `pip` works |
| Internet (HF Hub for first CLIP weight download) | D-09 first-run | unknown — assumed `[ASSUMED]` | — | If the HF cache is pre-populated at `~/.cache/huggingface/hub/timm--vit_large_patch14_clip_224.openai/`, no network needed. Otherwise the first run will fail without internet. |
| Disk space (~1.3 GB pip + 890 MB CLIP weights + 690 MB unzip + ≤30 MB output) | D-09 model + sample video | unknown `[ASSUMED]` | — | If <3 GB free, abort early with clear message. The planner can include a `df -h .` check in the bootstrap step, or skip it (most modern dev machines have GB to spare). |
| Sample video file | §0.5 verification | ✓ (after unzip) | `[VERIFIED: 690 MB Tiger Woods.mp4 in zip]` | If unzip fails: re-download `sample-vids.zip` from source. The .zip itself is in git (it's a binary in `videos/` which is gitignored — wait, the zip is committed?). **Confirmed:** `videos/sample-vids.zip` IS in the working tree per repo `ls`, so it's available. |

**Missing dependencies with no fallback:** none.

**Missing dependencies with fallback:** none required for Phase 1.

## Project Constraints (from CLAUDE.md)

Extracted directives — research has been written to comply:

- **Module layout is locked by spec §1** — no refactors. Phase 1 creates stubs for the 4 non-owned modules (Gap 7 step 7).
- **Drop `ffmpeg-python`** — D-02 + Standard Stack table.
- **Lazy-import `ruptures` inside `detect_changepoints`** — Phase 2 concern; Phase 1 doesn't import it at all.
- **`scipy==1.16.2` (avoid 1.15 medfilt regression)** — pinned in `requirements.txt` even though Phase 1 doesn't use scipy.
- **No transcripts, ASR, audio, OCR, LLMs in selection path** — N/A to Phase 1 (no selection logic here).
- **No median-filtering raw video frames** — N/A to Phase 1 (no filtering here).
- **Index-space vs time-space separation:** `extract.py` ships both `frames` (index-space) and `timestamps` (time-space). The contract is documented in ARCHITECTURE.md §3.1 and CONTEXT.md `<code_context>` Integration Points. Phase 2 receives `embeddings` only; Phase 3 receives `timestamps`. Phase 1's job is to ship both cleanly.
- **§0.5 prints are the success criteria** — D-16 + Gap 8 verification harness.

## Validation Architecture

Skipped — `.planning/config.json` has `workflow.nyquist_validation: false`. Per CONTEXT.md D-16 and Gap 10, the verification harness is the inline `__main__` block in `extract.py`. No pytest scaffolding in Phase 1.

## Sources

### Primary (HIGH confidence)
- `[VERIFIED: ctx7 /mlfoundations/open_clip]` — `create_model_and_transforms` 3-tuple return; `preprocess` returns `(3,224,224)` tensor without batch dim; `model.encode_image` returns `(B, 768)` NOT normalized by default; `normalize=True` kwarg available in 3.x.
- `.planning/research/STACK.md` — version pins, MPS notes, ffmpeg-python rationale, scipy regression citation.
- `.planning/research/PITFALLS.md` — Pitfalls 1, 2, 3, 4, 5, 6, 7, 15 (Phase 1 owned).
- `.planning/research/ARCHITECTURE.md` — Phase 1 data contracts (§3.1), parallel build order rationale (§6).
- `.planning/research/SUMMARY.md` — phase ordering and parallelism implications.
- `assignment-details/bodycam_highlight_reel_spec.md` — §§0, 0.5, 1, 2, 3, 10, 12 (locked spec).
- `.planning/phases/01-frame-extraction-embeddings/01-CONTEXT.md` — D-01..D-21 (locked decisions).
- Local environment probe — `ffmpeg 7.1.1`, `ffprobe 7.1.1`, `python 3.13.2`, `videos/sample-vids.zip` contents (one video).

### Secondary (MEDIUM confidence)
- OpenCV `VideoCapture` documentation (general behavior of `CAP_PROP_POS_MSEC` post-`read()`) — referenced in PITFALLS.md but specific 4.13.0.92 behavior assumed (A1).
- PyTorch reproducibility documentation — `torch.use_deterministic_algorithms`, MPS deterministic_algorithms behavior (A2).

### Tertiary (LOW confidence)
- None for Phase 1 — all material findings traced to primary or secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified against STACK.md (locked 2026-05-06) + ctx7 open_clip docs in this session.
- Architecture: HIGH — derived directly from CONTEXT.md D-01..D-21 + ARCHITECTURE.md.
- Pitfalls: HIGH — Phase 1 owns 8 named pitfalls all documented in PITFALLS.md with code-level guards.
- Implementation skeletons (Gaps 1–8): HIGH — code is concrete and grounded in verified APIs; the only `[ASSUMED]` items are the OpenCV `cap.get(...)` post-`read()` semantics (A1) and MPS bit-identical behavior (A5), both flagged.
- Sample video discovery (Gap 9): HIGH — verified by `unzip -l` against the real archive in this session. Project-scope concern flagged for Phase 6.
- pytest abstention (Gap 10): HIGH — locked by D-16 and Area F of DISCUSSION-LOG.

**Research date:** 2026-05-06
**Valid until:** 2026-06-05 (30 days; stack is on stable releases, no fast-moving APIs in this domain)

---
*Phase: 01-frame-extraction-embeddings*
*Researched: 2026-05-06*
