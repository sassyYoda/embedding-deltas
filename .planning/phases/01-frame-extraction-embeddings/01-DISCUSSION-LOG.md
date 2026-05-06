# Phase 1: Frame Extraction & Embeddings - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `01-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-06
**Phase:** 1 — Frame Extraction & Embeddings
**Mode:** `--auto` (recommended option auto-selected for each area; no interactive AskUserQuestion calls)
**Areas discussed:** Device selection, Model load location, Seek strategy, utils.py API surface, Fixture artifacts, Verification harness, Embedding cache, Determinism precursor

---

## Area A — Device Selection for CLIP Inference

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-detect helper | `torch.device('cuda' if cuda else 'mps' if mps else 'cpu')` inside `load_model()`; inference loop has no device branches. Spec §1's "no GPU-specific code paths" intent preserved. | ✓ |
| Force CPU only | Simplest; matches spec hardware assumption literally; ~3–5× slower on Apple Silicon. | |
| `--device` CLI flag | Adds a knob; ENV-04 says no GPU-specific paths. | |

**Auto-selected:** Auto-detect helper (recommended).
**Notes:** Single-line device pick centralized in `load_model()`. Research/STACK.md confirms MPS works on macOS 15.x without the macOS 26 regression.

---

## Area B — Where the CLIP Model Loads

| Option | Description | Selected |
|--------|-------------|----------|
| `embed_frames(frames, model, preprocess, device)` | Caller loads model once via `load_model()`; passes into each call. Multi-video batch (Phase 6) amortizes the ~30 s load. | ✓ |
| Load inside `embed_frames` | Matches spec §2 snippet literally; ~30 s overhead per video × 5 = wasteful in Phase 6. | |

**Auto-selected:** Caller passes model in (recommended).
**Notes:** Research/ARCHITECTURE.md flags the spec-§2 module-level load as a 5-line architectural improvement worth doing in Phase 1. The spec snippet's intent is preserved (`open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')`); only the call-site moves up one frame.

---

## Area C — Probe-Then-Seek vs Always-Time-Based Seek

| Option | Description | Selected |
|--------|-------------|----------|
| Always `CAP_PROP_POS_MSEC` | Time-based seek + read actual `cap.get(...)` timestamps. Correct for both CFR and VFR. Costs nothing. | ✓ |
| Probe first, switch by `nb_frames` | Run `ffprobe -show_streams` upfront; use `POS_FRAMES` if CFR, `POS_MSEC` if VFR. Adds branches. | |

**Auto-selected:** Always `CAP_PROP_POS_MSEC` (recommended).
**Notes:** Research/PITFALLS.md §2 names `CAP_PROP_POS_FRAMES` as the most likely silent-correctness bug on body-cam VFR encodings. The single safe path eliminates the choice.

---

## Area D — `utils.py` API Surface in Phase 1

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 1 ships `probe_video_metadata` + `ensure_output_dirs` only | Defer `write_timestamps_json` to Phase 5 (where JSON schema lives). | ✓ |
| Ship all utils.py helpers in Phase 1 | Includes `write_timestamps_json` even though Phase 1 has no JSON to write. | |

**Auto-selected:** Phase 1 minimal surface (recommended).
**Notes:** Research/ARCHITECTURE.md flags `utils.py` contents as MEDIUM confidence ("derived from call sites, not stated in spec"). Locking only the Phase 1-needed subset keeps the API honest.

---

## Area E — Fixture Artifacts (`embeddings.npy` + `timestamps.npy`) for Phase 2/3 Dev

| Option | Description | Selected |
|--------|-------------|----------|
| `extract.py --save-fixture` writes `output/cache/{video}_embeddings.npy` + `_timestamps.npy` | Phase 2/3/4 develop against fixtures without re-running 30+ minutes of CLIP inference. Internal dev tool only. | ✓ |
| No fixture; re-run CLIP each time | Slow but matches spec verbatim. | |

**Auto-selected:** Save fixture from `extract.py.__main__` (recommended).
**Notes:** Research/SUMMARY.md "Build-order parallelism" calls this out: "once `extract.py` produces an `(embeddings.npy, timestamps.npy)` fixture from one sample video, `signal_processing.py` / `clip_selection.py` / `export.py` are independent." This is the parallelism the spec footer's serial order hides.

---

## Area F — Verification Harness Style for Spec §0.5

| Option | Description | Selected |
|--------|-------------|----------|
| `if __name__ == "__main__":` block in `extract.py` | Runs §0.5 checks against a single sample video. `python extract.py <video.mp4>`. Matches spec §0.5 protocol literally. | ✓ |
| Separate `verify.py` module | Cleaner separation but adds scaffolding for no current benefit. | |

**Auto-selected:** Inline `__main__` block (recommended).
**Notes:** Spec §0.5 is per-module ("After `extract.py`: ..."). Co-locating the checks with the module reads naturally and makes the verification gate concrete.

---

## Area G — Embedding Cache (DIAG-02 / v2)

| Option | Description | Selected |
|--------|-------------|----------|
| Defer entirely | DIAG-02 is explicitly v2 in REQUIREMENTS.md. Phase 1 ships without a `--cache` flag. The Phase 1 fixture (Area E) gives ~80% of the benefit and is internal. | ✓ |
| Scaffold a `--cache` flag now | Promotes DIAG-02 into v1; small addition. | |

**Auto-selected:** Defer entirely (recommended).
**Notes:** Honors REQUIREMENTS.md v1/v2 split. The fixture from Area E is dev-only and not exposed via `pipeline.py`.

---

## Area H — Determinism Precursors in Phase 1

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 1 owns inference-side determinism: `model.eval()`, `torch.inference_mode()`, single batched code path, `cudnn.benchmark = False` | Env-var stanza waits for Phase 5 where `pipeline.run()` is the entry point. | ✓ |
| Set the env-var stanza in Phase 1 too | Spreads the concern across phases; Phase 1 doesn't own pipeline entry. | |

**Auto-selected:** Phase 1 inference-side only (recommended).
**Notes:** The env vars (`OMP_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`) belong wherever the process entry point is. Phase 5 owns `pipeline.py`. ROBU-02 in REQUIREMENTS.md confirms Phase 5 placement.

---

## Claude's Discretion

- Logger vs `print()` for §0.5 output (default: `print()`, matches spec snippets).
- Exact wording of error messages, docstrings, type-hint style (PEP 604 unions, lowercase generics on Python 3.11+).
- Tolerance constants (`atol=1e-5` for L2 norm, `< 1.0` s for duration drift).

## Deferred Ideas

- Determinism env-var stanza → Phase 5
- User-facing `--cache` flag → v2 (DIAG-02)
- `utils.write_timestamps_json` → Phase 5
- `--diagnostics` plots → v2 (DIAG-01)
- Concat-filter re-encode fallback → Phase 4
- JSON float rounding → Phase 5
- VFR-vs-CFR probe-then-switch — rejected in Area C
- Module-level model loading per spec §2 verbatim — rejected in Area B
