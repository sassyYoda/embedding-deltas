# Requirements: Body Cam Highlight Reel

**Defined:** 2026-05-06
**Core Value:** Given a body cam video, the pipeline must produce a highlight reel where selected moments visibly correspond to high-action or significant scene changes — using only visual embedding signal — and must do so reproducibly across all five sample videos with one fixed parameter set.

> Source of truth: `assignment-details/bodycam_highlight_reel_spec.md`. Every v1 requirement traces to a numbered section in the spec or to its `§0.5 Testing Protocol` / `§12 What Not To Do` rules.

## v1 Requirements

### Environment & Project Structure

- [ ] **ENV-01**: Project provides a pinned `requirements.txt` covering `torch`, `torchvision`, `open_clip_torch`, `opencv-python`, `numpy`, `scipy`, `ruptures`, `tqdm` (per spec §1, with `ffmpeg-python` dropped per research)
- [x] **ENV-02
**: Project verifies system `ffmpeg` is installed and on PATH at startup
- [ ] **ENV-03**: Project layout matches spec §1 exactly: `pipeline.py`, `extract.py`, `signal_processing.py`, `clip_selection.py`, `export.py`, `utils.py`, `videos/`, `output/{reels,clips,timestamps}/`
- [ ] **ENV-04**: Code runs on Python 3.11+ with no GPU-specific code paths (device selection deferred to `open_clip` / torch defaults)

### Frame Extraction

- [x] **EXTR-01
**: `extract.sample_frames(video_path, fps=2.0)` samples one frame every 0.5 seconds from the input video
- [x] **EXTR-02
**: Sampled frames are returned as RGB (BGR-to-RGB conversion verified)
- [x] **EXTR-03
**: Per-frame timestamps are recorded as float seconds, derived from actual playback position (not nominal frame index, to handle VFR body-cam encodings — pitfall research)
- [x] **EXTR-04
**: Video duration is sourced via `ffprobe` (not `cv2.CAP_PROP_FRAME_COUNT`, which is unreliable on body-cam MP4s — pitfall research)
- [x] **EXTR-05
**: §0.5 verification: number of sampled frames printed; first 5 timestamps printed and asserted to be 0.5 s apart

### Embedding Extraction

- [x] **EMBD-01
**: Model is loaded via `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')` — no substitution
- [x] **EMBD-02
**: `model.eval()` and `torch.inference_mode()` (or `torch.no_grad()`) wrap all inference
- [x] **EMBD-03
**: Frames are processed in batches of 32
- [x] **EMBD-04
**: Output embeddings have shape `(N, 768)` and dtype `float32`
- [x] **EMBD-05
**: Embeddings are explicitly L2-normalized; an assertion (`np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)`) guards against silent unnormalized output
- [x] **EMBD-06
**: §0.5 verification: embeddings shape and L2-norm assertion both pass at the end of `extract.py`

### Signal Processing — Deltas

- [ ] **SIGD-01**: `signal_processing.compute_deltas(embeddings)` returns an `(N-1,)` array of cosine distances `1 - dot_product`, with `np.clip(dots, -1, 1)` for numerical safety
- [ ] **SIGD-02**: Timestamp alignment invariant is implemented and documented: `delta[i]` (and `score[i]`) corresponds to `timestamps[i+1]` — encoded as a single helper used by `clip_selection`
- [ ] **SIGD-03**: §0.5 verification: first 20 raw delta values printed and visually within 0.0–0.3 typical range; length asserted to be `N-1`

### Signal Processing — Smoothing

- [ ] **SIGS-01**: A median filter is applied to raw deltas with kernel size 5 (per spec §5 step 1)
- [ ] **SIGS-02**: §0.5 verification: raw vs smoothed deltas compared — no isolated single-sample spikes in smoothed signal; sustained high-delta periods preserved

### Signal Processing — MAD Normalization

- [ ] **SIGM-01**: `mad_normalize(smoothed, window_samples=180)` implements rolling MAD normalization over a centered 180-sample (= 90 s at 2 fps) window
- [ ] **SIGM-02**: Zero-MAD windows (static footage) are guarded against; `0.0` is emitted in that case
- [ ] **SIGM-03**: Output is post-clipped to `[0.0, 10.0]` per spec §5
- [ ] **SIGM-04**: §0.5 verification: min/max/mean of normalized scores printed; max ≥ 2.0 sanity-checked (window not too large), <90% of samples above 3.0 (window not too small)

### Signal Processing — PELT (opt-in)

- [ ] **SIGP-01**: Pipeline accepts a `--pelt` boolean flag (default `False`) per spec §5
- [ ] **SIGP-02**: When `--pelt` is passed, `detect_changepoints(smoothed, penalty=3.0)` runs via `ruptures.Pelt(model="rbf")`; `ruptures` is lazy-imported so non-PELT runs do not require it (pitfall + research recommendation)
- [ ] **SIGP-03**: When `--pelt` is not passed, `detect_changepoints` is never called and PELT-related JSON fields are emitted as `null` (not `False`, not omitted)

### Clip Selection — Peak Detection

- [ ] **SELP-01**: `select_peaks(scores, timestamps, height, min_gap_sec, fps=2.0)` uses `scipy.signal.find_peaks` with the height threshold and a `distance` parameter expressed in samples (`min_gap_sec * fps`)
- [ ] **SELP-02**: Peaks are sorted by descending score
- [ ] **SELP-03**: When PELT is active, peaks within ±5 samples of any changepoint receive a 1.2× score boost; results are re-sorted descending by boosted score
- [ ] **SELP-04**: §0.5 verification: number of peaks detected, timestamps, and scores are printed before budget enforcement

### Clip Selection — Construction, Merge, Budget

- [ ] **SELB-01**: `compute_budget_seconds(video_duration_sec)` returns `(duration / 1800) * 60` (1 minute of reel per 30 minutes of video)
- [ ] **SELB-02**: `compute_padding(budget_sec)` returns `max(3.0, min(8.0, budget_sec * 0.15))`
- [ ] **SELB-03**: `build_clips(...)` builds candidate clips chronologically with `start = max(0, peak_time - padding)` and `end = min(video_duration, peak_time + padding)`
- [ ] **SELB-04**: `merge_clips(clips, gap_threshold_sec)` merges adjacent clips within `--merge-gap-sec` (default 3.0); merged clip retains the maximum score
- [ ] **SELB-05**: `enforce_budget(merged_clips, budget_sec)` greedily selects clips by score descending until the budget is hit; if a partial clip has ≥3.0 s remainder, it is appended; final list is re-sorted chronologically
- [ ] **SELB-06**: §0.5 verification: final selected clips printed (start, end, duration, score); total duration asserted ≤ budget; no two clips overlap after merging

### Export — Cut & Concat

- [ ] **EXPC-01**: `extract_clip(input_path, output_path, start_sec, end_sec)` invokes `ffmpeg -y -ss <start> -to <end> -i <input> -c copy <output>` via `subprocess.run`
- [ ] **EXPC-02**: `concat_clips(clip_paths, output_path, temp_dir)` writes a concat manifest with absolute paths and runs `ffmpeg -y -f concat -safe 0 -i <manifest> -c copy <output>`
- [ ] **EXPC-03**: §0.5 verification: each intermediate clip file exists and is non-zero bytes; first clip is the correct segment when played; final reel is a coherent concatenation

### Orchestration & CLI

- [ ] **ORCH-01**: `pipeline.py` accepts a positional `video` argument and exposes `--pelt`, `--height` (default 1.5), `--min-gap-sec` (default 15.0), `--merge-gap-sec` (default 3.0) flags
- [ ] **ORCH-02**: Pipeline prints stage banners `[1/6]…[6/6]` matching spec §9
- [ ] **ORCH-03**: All output directories (`output/reels`, `output/clips/{video_name}`, `output/timestamps`) are created on demand
- [ ] **ORCH-04**: A reel is written to `output/reels/{video_name}_highlight.mp4`

### JSON Output

- [ ] **JSON-01**: Each video produces `output/timestamps/{video_name}.json` matching spec §8 schema field-for-field: `video`, `video_duration_sec`, `budget_sec`, `total_reel_duration_sec`, `embedding_model`, `sampling_fps`, `clips[]`
- [ ] **JSON-02**: Each clip object contains `clip_index`, `start_sec`, `end_sec`, `duration_sec`, `peak_timestamp_sec`, `mad_score`, `raw_cosine_delta`, `coincides_with_pelt_changepoint`
- [ ] **JSON-03**: `coincides_with_pelt_changepoint` is `null` when `--pelt` is off, `true`/`false` when `--pelt` is on (strict three-state — pitfall research)
- [ ] **JSON-04**: `peak_timestamp_sec` falls within `[start_sec, end_sec]` for every clip (asserted post-merge — pitfall research)

### Multi-Video Reproducibility

- [ ] **RUN-01**: All five sample videos in `videos/` can be processed by running the pipeline back-to-back with one fixed parameter set (no per-video retuning — spec §6, §12)
- [ ] **RUN-02**: A `run_all.sh` (or equivalent in-process loop) exists for batch execution
- [ ] **RUN-03**: All 5 reels and 5 JSON files are produced in `output/reels/` and `output/timestamps/` respectively

### Submission Polish

- [ ] **DOC-01**: README documents installation, runtime, parameter rationale, and per-video qualitative observations (the only credible eval given no labeled GT — spec §11)
- [ ] **DOC-02**: README acknowledges known limitations from spec §11 and explicit non-goals from §12
- [ ] **DOC-03**: One representative video has been used to tune `--height` / `--min-gap-sec` / `--merge-gap-sec`; chosen values are documented and frozen across all 5 runs

## v2 Requirements

Deferred — could improve the prototype but not required for the assignment.

### Diagnostics

- **DIAG-01**: Optional `--diagnostics` flag emits PNG plots of raw deltas, smoothed deltas, MAD scores, peaks, and selected clips
- **DIAG-02**: Embedding cache (`output/cache/{video_name}_embeddings.npy`) avoids re-running CLIP on tuning passes
- **DIAG-03**: Per-phase timing logs printed alongside the `[1/6]` stage banners

### Robustness

- **ROBU-01**: `concat_clips` automatically falls back to the concat filter (`-c:v libx264 -crf 18`) when the demuxer fails due to codec/timebase mismatch (pitfall research)
- **ROBU-02**: `pipeline.py` sets determinism env vars (`OMP_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`) at entry to make two-run JSON output byte-identical (pitfall research)
- **ROBU-03**: JSON floats are rounded (3 dp for seconds, 4 dp for scores) for byte-stable diffs

## Out of Scope

| Feature | Reason |
|---|---|
| Audio / ASR / OCR / LLM clip selection | Assignment §12 prohibits — visual-embedding-only constraint |
| Manual timestamp selection (even for verification) | Assignment §12 prohibits — must be fully algorithmic |
| Per-video parameter retuning | Spec §6 prohibits — would amount to manual selection |
| Fine-tuning or modifying the CLIP model | Spec §2 prohibits |
| Substituting a different embedding model (SigLIP, ALIGN, DINOv2, etc.) | Spec §2 locks ViT-L/14 OpenAI |
| Fixed global threshold (instead of rolling MAD) | Spec §12 prohibits — was Antonio's original failure mode |
| Median-filtering raw video frames | Spec §12 prohibits — filter the 1D delta signal only |
| Default `libx264` re-encode for clip extraction | Spec §12 prohibits — `-c copy` is faster, lossless, and acceptable |
| Web UI / Streamlit / dashboard | Out of scope per PROJECT.md — CLI prototype only |
| Quantitative precision/recall evaluation | No labeled ground truth available — spec §11 acknowledges |
| Configurable embedding model or JSON schema | Locked by spec §2 and §8; configurability adds risk for no benefit |
| GPU-specific code paths | Spec §1 forbids — let `open_clip` / torch handle device selection |
| Polishing beyond reviewer-needed signal | Spec §12 — "honest analysis beats optimistic framing" |

## Traceability

Confirmed by roadmap creation (2026-05-06). Every v1 requirement is mapped to exactly one phase.

| Requirement | Phase | Status |
|---|---|---|
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
| ENV-04 | Phase 1 | Pending |
| EXTR-01 | Phase 1 | Pending |
| EXTR-02 | Phase 1 | Pending |
| EXTR-03 | Phase 1 | Pending |
| EXTR-04 | Phase 1 | Pending |
| EXTR-05 | Phase 1 | Pending |
| EMBD-01 | Phase 1 | Pending |
| EMBD-02 | Phase 1 | Pending |
| EMBD-03 | Phase 1 | Pending |
| EMBD-04 | Phase 1 | Pending |
| EMBD-05 | Phase 1 | Pending |
| EMBD-06 | Phase 1 | Pending |
| SIGD-01 | Phase 2 | Pending |
| SIGD-02 | Phase 2 | Pending |
| SIGD-03 | Phase 2 | Pending |
| SIGS-01 | Phase 2 | Pending |
| SIGS-02 | Phase 2 | Pending |
| SIGM-01 | Phase 2 | Pending |
| SIGM-02 | Phase 2 | Pending |
| SIGM-03 | Phase 2 | Pending |
| SIGM-04 | Phase 2 | Pending |
| SIGP-01 | Phase 2 | Pending |
| SIGP-02 | Phase 2 | Pending |
| SIGP-03 | Phase 2 | Pending |
| SELP-01 | Phase 3 | Pending |
| SELP-02 | Phase 3 | Pending |
| SELP-03 | Phase 3 | Pending |
| SELP-04 | Phase 3 | Pending |
| SELB-01 | Phase 3 | Pending |
| SELB-02 | Phase 3 | Pending |
| SELB-03 | Phase 3 | Pending |
| SELB-04 | Phase 3 | Pending |
| SELB-05 | Phase 3 | Pending |
| SELB-06 | Phase 3 | Pending |
| EXPC-01 | Phase 4 | Pending |
| EXPC-02 | Phase 4 | Pending |
| EXPC-03 | Phase 4 | Pending |
| ORCH-01 | Phase 5 | Pending |
| ORCH-02 | Phase 5 | Pending |
| ORCH-03 | Phase 5 | Pending |
| ORCH-04 | Phase 5 | Pending |
| JSON-01 | Phase 5 | Pending |
| JSON-02 | Phase 5 | Pending |
| JSON-03 | Phase 5 | Pending |
| JSON-04 | Phase 5 | Pending |
| DOC-03 | Phase 5 | Pending |
| RUN-01 | Phase 6 | Pending |
| RUN-02 | Phase 6 | Pending |
| RUN-03 | Phase 6 | Pending |
| DOC-01 | Phase 6 | Pending |
| DOC-02 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 51 total
- Mapped to phases: 51 ✓
- Unmapped: 0 ✓
- Duplicates: 0 ✓

**Phase distribution:**
- Phase 1 (Frame Extraction & Embeddings): 15 requirements
- Phase 2 (Signal Processing): 12 requirements
- Phase 3 (Clip Selection): 10 requirements
- Phase 4 (Export): 3 requirements
- Phase 5 (Orchestration & First-Video End-to-End): 9 requirements
- Phase 6 (Multi-Video Run & Submission Polish): 5 requirements

**Note on DOC-03:** Tuning + freezing parameters is the gate that lets Phase 6 begin, so DOC-03 lives in Phase 5 (where the work happens). DOC-01/02 (README narrative) are written in Phase 6 and reference the values frozen in Phase 5.

---
*Requirements defined: 2026-05-06*
*Last updated: 2026-05-06 after roadmap creation — traceability confirmed*
