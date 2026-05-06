---
phase: 01-frame-extraction-embeddings
plan: 02
subsystem: extract
tags: [extract, clip, opencv, open_clip, frame-sampling, embedding, verification, mps]
requires:
  - "utils.probe_video_metadata"
  - "utils.ensure_output_dirs"
  - "requirements.txt (torch 2.11.0, open_clip_torch 3.3.0, opencv-python 4.13.0.92, numpy 2.4.4, scipy 1.16.2)"
provides:
  - "extract.sample_frames (CAP_PROP_POS_MSEC time-based seek; RGB uint8 frames + float64 seconds)"
  - "extract.load_model ((model, preprocess, device) — D-10 caller-passes-model contract)"
  - "extract.embed_frames ((N, 768) float32 L2-normalized; single batched code path)"
  - "extract __main__ §0.5 verification harness (--save-fixture default ON)"
  - "output/cache/justin_timberlake_embeddings.npy ((2269, 768) float32) — Phase 2/3/4 dev fixture"
  - "output/cache/justin_timberlake_timestamps.npy ((2269,) float64) — Phase 2/3/4 dev fixture"
affects:
  - "Phase 2 signal_processing.compute_deltas (consumes embeddings.npy)"
  - "Phase 3 clip_selection.build_clips (consumes timestamps.npy via Phase 5 orchestrator)"
  - "Phase 5 pipeline.run (calls load_model once, sample_frames + embed_frames per video)"
  - "Phase 6 batch driver (amortizes load_model across 5 videos per D-10)"
tech-stack:
  added: []     # all deps already pinned in Plan 01-01's requirements.txt
  patterns:
    - "single-batched-code-path inference (Pitfall 7) — production path also serves the determinism rerun"
    - "ffmpeg precondition + ffprobe-as-source-of-truth for duration (Pitfalls 2, 3)"
    - "explicit-and-asserted L2 normalize (Pitfall 6) — assertion fires inside embed_frames AND in __main__"
    - "smart-default video discovery via Path.rglob('*.mp4') — no hardcoded sample filename (resilient to gdown rename)"
key-files:
  created: []   # extract.py was created by Plan 01-02 Task 1, in commit d99fd32 (this plan)
  modified:
    - "extract.py (Task 2 added __main__ block in commit 40a1e28)"
decisions:
  - "Honored D-04: CAP_PROP_POS_MSEC seek (NOT POS_FRAMES) — verified zero matches for forbidden patterns."
  - "Honored D-07: BGR→RGB conversion is the only cvtColor call in the codebase (single source)."
  - "Honored D-09 verbatim: open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')."
  - "Honored D-10: load_model is factored out; embed_frames takes (model, preprocess, device) as args."
  - "Honored D-11: cuda → mps → cpu device ladder; MPS auto-selected on Apple Silicon."
  - "Honored D-12: single batched code path; bit-identical rerun goes through embed_frames([frame], …)."
  - "Honored D-13: feats / feats.norm(...) explicit L2 + np.allclose(..., atol=1e-5) hard assertion (twice — embed_frames + __main__)."
  - "Honored D-15: cudnn.benchmark=False set in load_model; NO env vars or use_deterministic_algorithms (Phase 5 owns those)."
  - "Honored D-16: __main__ block — no pytest, no separate verify.py."
  - "Honored D-17: bit-identical rerun assertion via np.array_equal — passed on MPS without warm-up."
  - "Honored D-18: --save-fixture default ON (BooleanOptionalAction); --no-save-fixture skips."
  - "Honored D-19: no --cache flag added (Phase 1 fixture is internal dev tool only)."
  - "Honored ENV-02 / D-03: ffmpeg -version is the first runtime check after argparse."
  - "Honored EXTR-04: duration sourced via utils.probe_video_metadata (ffprobe), NOT cv2.CAP_PROP_FRAME_COUNT."
  - "Smart-default video discovery (RESEARCH Gap 9) — extract.py finds videos/*.mp4 if no path given."
metrics:
  tasks_completed: 2
  commits: 2
  duration_minutes: ~12
  completed_date: 2026-05-06
  harness_runtime_sec: 615.25
  harness_device: "mps"
---

# Phase 1 Plan 02: extract.py — sample_frames, load_model, embed_frames + §0.5 Harness Summary

Implemented the canonical `(timestamps, embeddings)` artifact producer that gates every downstream phase. Three pure functions plus a `__main__` block that runs spec §0.5 end-to-end on a single video, with a code-level guard plus a runtime assertion for each of the 8 phase-owned pitfalls (1, 2, 3, 4, 5, 6, 7, 15).

## Public API (verbatim from PLAN `<interfaces>`)

```python
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

def load_model() -> tuple[Any, Any, torch.device]:
    """Load CLIP ViT-L-14 OpenAI; return (model, preprocess, device).
    - device: cuda → mps → cpu ladder (D-11)
    - cudnn.benchmark = False (D-15 / Pitfall 15) — no-op on MPS/CPU; correct on CUDA
    - model in eval() mode, on `device`
    - preprocess: third element of open_clip.create_model_and_transforms (preprocess_val)
    """

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
    Returns shape (len(frames), 768) dtype float32. Empty input → shape (0, 768).
    """
```

## §0.5 Harness — Verbatim stdout from a successful run

Phase 2 may grep these markers as a Phase-1-complete signal. Run on `videos/justin_timberlake.mp4` (h264, 1280x720, CFR, 1134.144s) on Apple Silicon MPS, including the first-run 890 MB CLIP weight download:

```
[probe] duration=1134.144s  resolution=1280x720  codec=h264  is_vfr=False
Sampled 2269 frames; first 5 timestamps: [0.0, 0.5005000000000001, 1.0010000000000001, 1.5015, 2.0020000000000002]; last timestamp: 1134.000s; ffprobe duration: 1134.144s
[model] loading CLIP ViT-L-14 OpenAI (first run downloads ~890 MB to HF cache)…
[model] loaded; device=mps
embeddings shape: (2269, 768)
L2-norm check: PASS  (norms in [1.000000, 1.000000])
[determinism] running bit-identical rerun on a single frame…
[determinism] PASS  (e1 == e2 byte-for-byte)
[fixture] wrote output/cache/justin_timberlake_embeddings.npy
[fixture] wrote output/cache/justin_timberlake_timestamps.npy
Phase 1 §0.5 verification: PASS
```

Wall-clock from `/usr/bin/time -p`: **real 615.25s, user 542.68s, sys 52.64s** (includes the one-time CLIP weight download to `~/.cache/huggingface/hub/`).

## Pitfall guards in code

Each phase-owned pitfall has a code-level guard inside the three functions plus a runtime assertion that fires in `__main__`. Verified by static `ast.parse` checks and the live harness run.

| Pitfall | Guard location | Mechanism |
|---|---|---|
| 1 — BGR/RGB confusion | `sample_frames` (line ~63) | `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)` — `src.count('cvtColor') == 1`, single source of conversion |
| 2 — `CAP_PROP_POS_FRAMES` seek inaccuracy | `sample_frames` (line ~55) | `cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)` — `CAP_PROP_POS_FRAMES` is NOT in the source |
| 3 — `CAP_PROP_FRAME_COUNT` lies | `sample_frames` loop (line ~54-71) + `__main__` (line ~235) | `while True: …if not ret or frame_bgr is None: break`; duration source is `utils.probe_video_metadata` (ffprobe); runtime `assert abs(timestamps[-1] - probe_duration) < 1.0` (drift was 0.144s on JT) |
| 4 — `model.eval()` forgotten | `load_model` (line ~105) + `embed_frames` (line ~131) + `with torch.inference_mode():` (line ~135) | Defense in depth: eval set in load AND in embed; inference_mode wraps the entire batched loop |
| 5 — `preprocess` wrong type | `embed_frames` (line ~139, 143-145) | `preprocess(Image.fromarray(f))`; per-batch `assert batch_tensor.shape[1:] == (3, 224, 224)` |
| 6 — L2 norm not enforced | `embed_frames` (line ~149, 156-159) + `__main__` (line ~251-255) | Explicit `feats / feats.norm(dim=-1, keepdim=True)`; hard `np.allclose(..., 1.0, atol=1e-5)` assertion fires INSIDE embed_frames AND again in `__main__` (norms range was [1.000000, 1.000000] on JT) |
| 7 — batch vs single inference inconsistency | `embed_frames` (line ~136) + `__main__` (line ~261-262) | No `if len(frames) == 1` branch; `for start in range(0, len(frames), batch_size)`; rerun uses `embed_frames([frames[0]], …)` — same code path; `np.array_equal(e1, e2)` passed |
| 15 — `cudnn.benchmark` perturbs first batch | `load_model` (line ~96-97) | `torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True` (no-op on MPS/CPU; correct on CUDA) |

## Fixture artifacts (D-18)

Written to `output/cache/` on a successful `--save-fixture` run (default ON):

| Path | Shape | Dtype | Size |
|---|---|---|---|
| `output/cache/justin_timberlake_embeddings.npy` | (2269, 768) | float32 | 6,970,496 B (≈6.97 MB; 6,970,368 B payload + 128 B header) |
| `output/cache/justin_timberlake_timestamps.npy` | (2269,) | float64 | 18,280 B (≈18 KB; 18,152 B payload + 128 B header) |

Round-trip validated: re-loaded with `np.load(..., allow_pickle=False)`, L2-norm range stayed `[1.000000, 1.000000]`, timestamps monotonically non-decreasing, first 5 = `[0.0, 0.5005, 1.001, 1.5015, 2.002]`, last = `1134.000s`.

## Caveats observed

- **VFR detection (`is_vfr`):** `False` for `justin_timberlake.mp4`. The other 4 sample videos (currently downloading via `gdown` into `videos_staging/`) may differ — `sample_frames` will handle either case identically per D-06.
- **Drift between `last timestamp` (1134.000s) and `ffprobe duration` (1134.144s):** 0.144s, well under the 1.0s D-08 tolerance. Plausible explanation: the post-`cap.read()` `cap.get(POS_MSEC)` returns the *next* frame's timestamp (RESEARCH A1 / Gap 2), and the source has a ~30 fps rate (sample interval ≈ 0.5005s, matching `2 × 0.5 + 0.0005 ≈ source-frame-interval` forward-bias).
- **MPS bit-identical rerun (D-17 / Pitfall 7):** **PASSED** on the first try without an explicit warm-up batch. The documented fallback (`_ = model.encode_image(torch.zeros(1, 3, 224, 224, device=device))` per RESEARCH A5) was NOT needed — the implicit warm-up from the production embedding pass is sufficient before the rerun fires. This is the exact path Phase 5/6 will take.
- **Wall-clock 615s on MPS** for 2269 frames + 890 MB weight download. Steady-state per-frame embedding cost is ≈0.27s/frame on MPS (from harness total minus model download), or ≈9 min for a 30-min video. Phase 6's batch driver should expect ≈30–50 min total wall-time across 5 videos (model loaded ONCE per D-10, amortizing the ~30s load).

## Threat Flags

None — Phase 1 added no new network endpoints, auth paths, file-write permissions outside the documented `output/cache/` cache, or schema changes. The `--save-fixture` flag writes only inside `output/` (gitignored).

## Deviations from Plan

### Auto-fixed Issues
None — both tasks executed exactly as written.

### Open Issue — flagged for user attention (CONTEXT amendment territory)

**[Rule 4 — Architectural / D-09 contradicts open_clip 3.3.0 runtime behavior]** open_clip 3.3.0 emits a UserWarning at model load:

```
UserWarning: QuickGELU mismatch between final model config (quick_gelu=False)
and pretrained tag 'openai' (quick_gelu=True).
```

This means the *bare* call `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')` (per D-09 verbatim) loads the OpenAI weights into a model whose default activation is **standard GELU**, not the **QuickGELU** variant the OpenAI checkpoint expects. Spec §2 explicitly states: "Use open_clip with the OpenAI pretrained checkpoint specifically — this is the **QuickGELU variant** identified as most robust to domain-shift corruptions". CONTEXT D-09 says: "Exact call, no substitution. **The QuickGELU variant is forced by the registry** — verified against `open_clip_torch==3.3.0`." That second sentence is no longer true at runtime in 3.3.0.

**Why this is flagged not auto-fixed:** D-09 is a locked CONTEXT decision, and the §0.5 harness PASSES as-is (L2-norm fine, deterministic, shape correct). The change would be a one-line model-name swap to `'ViT-L-14-quickgelu'` (which `open_clip.list_pretrained_tags_by_model('ViT-L-14-quickgelu')` confirms supports `pretrained='openai'`).

**Options for the user:**
1. **Amend D-09** to use `'ViT-L-14-quickgelu'` — silences the warning, satisfies spec §2's robustness rationale, requires re-running the harness (the embedding fixture would change and Phase 2 would consume slightly different vectors).
2. **Keep D-09 as-is** — accept the warning; the model still produces valid 768-d embeddings; the spec §2 robustness justification becomes weaker (the cited paper's QuickGELU result no longer applies to our actual runtime model).
3. **Pass `force_quick_gelu=True`** to `create_model_and_transforms` (also a one-line change; preserves the model name verbatim per D-09).

Recommendation: option 3 (most conservative — preserves D-09's model-name verbatim while honoring spec §2's QuickGELU requirement). Either option 1 or 3 would re-emit the fixture, so the cleanest sequencing is to make this call BEFORE Phase 2 starts consuming the embeddings.

This deviation does NOT block Phase 1 completion — the harness passes all explicit gates. But it should be resolved before the locked fixtures propagate to Phase 2.

## Forward pointer for Phase 2

`signal_processing.py` should load the fixture for fast iteration:

```python
import numpy as np
from pathlib import Path

cache = Path("output/cache")
embeddings = np.load(cache / "justin_timberlake_embeddings.npy", allow_pickle=False)  # (2269, 768) float32
timestamps = np.load(cache / "justin_timberlake_timestamps.npy", allow_pickle=False)  # (2269,) float64
# embeddings ARE L2-normalized (asserted by Plan 01-02). compute_deltas can dot-product directly.
```

Phase 2 owns Pitfalls 8 (off-by-one alignment), 9 (medfilt edge effects), 10 (zero-MAD windows). Phase 1 has done its part; the embeddings carry the contract `np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5) is True`.

## Self-Check: PASSED

Files claimed:
- `extract.py` — FOUND (committed in 40a1e28; full 282-line module + __main__)
- `output/cache/justin_timberlake_embeddings.npy` — FOUND (6,970,496 bytes)
- `output/cache/justin_timberlake_timestamps.npy` — FOUND (18,280 bytes)

Commits claimed:
- `d99fd32` (Task 1: feat(01-02-01): implement extract.py functions) — FOUND in `git log`
- `40a1e28` (Task 2: feat(01-02-02): add §0.5 verification harness) — FOUND in `git log`

Static checks (re-run from PLAN's `<verify><automated>` blocks): all assertions pass; `ALL STATIC CHECKS PASS`.

Runtime gates (SC2 / SC3 / SC4):
- **SC2 (sample_frames):** PASS — 2269 frames, first 5 ts spaced ≈0.5005s, drift 0.144s < 1.0s tolerance.
- **SC3 (embed_frames):** PASS — shape (2269, 768), L2-norms in [1.000000, 1.000000], `np.array_equal(e1, e2) == True` on MPS without warm-up.
- **SC4 (§0.5 harness):** PASS — full stdout block emitted; exit code 0; both npy fixtures written.

Phase-1-complete signal grep:
```
grep -E "Sampled .* frames|embeddings shape: \(|L2-norm check: PASS|\[determinism\] PASS|Phase 1 §0.5 verification: PASS" /tmp/extract_run.log
```
All 5 markers present.
