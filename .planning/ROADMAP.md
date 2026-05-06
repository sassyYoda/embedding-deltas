# Roadmap: Body Cam Highlight Reel — AbelPolice Take-Home

**Created:** 2026-05-06
**Granularity:** coarse
**Source:** PROJECT.md + REQUIREMENTS.md (51 v1) + research/{SUMMARY,ARCHITECTURE,PITFALLS}.md
**Spec:** `assignment-details/bodycam_highlight_reel_spec.md` (locked single source of truth)

---

## Core Value

Given a body cam video, the pipeline must produce a highlight reel where the selected moments visibly correspond to high-action or significant scene changes — using only visual embedding signal — and must do so reproducibly across all five sample videos with the same fixed parameters.

---

## Cross-Cutting Invariants

These hold across all phases and are referenced explicitly where they lock:

- **Timestamp alignment:** `scores[i] ⟺ timestamps[i+1]` (the delta "arrives at" frame `i+1`). Encoded as a single helper in `signal_processing.py` and used by `clip_selection.py` as the *only* place index→seconds conversion happens. **Locked in Phase 2; consumed in Phase 3; verified in Phase 5 JSON (`start_sec ≤ peak_timestamp_sec ≤ end_sec`).**
- **`--pelt` orthogonality:** PELT is opt-in supplementary signal, never on the critical path. `ruptures` is **lazy-imported inside `detect_changepoints()`** so non-PELT runs never touch it. JSON field `coincides_with_pelt_changepoint` is strict three-state: `null` when `--pelt` is off, `true`/`false` when on. **Detection lives in Phase 2; boost+JSON application in Phase 3+5; flag plumbing in Phase 5.**
- **Index-space vs time-space separation:** `signal_processing.py` works in index space only; `clip_selection.py` is the single seam where indices become seconds. No other module receives `timestamps`.
- **Tune once, freeze:** `--height`, `--min-gap-sec`, `--merge-gap-sec` are tuned **once** on the most representative video in Phase 5 and **frozen** across all 5 in Phase 6. Per-video retuning would amount to manual selection (spec §6, §12).
- **§0.5 verification prints are the success criteria.** Each module's spec-mandated verification step is the qualitative gate that lets the next module begin. Phase success criteria below mirror those checks.

---

## Phases

- [x] **Phase 1: Frame Extraction & Embeddings** — `extract.py` + utils video-metadata helpers; produce the canonical `(timestamps, embeddings)` fixture from one sample video, with all 8 frame/embedding pitfalls neutralized. **DONE 2026-05-06** — verified on `videos/justin_timberlake.mp4` (615s on MPS); fixture at `output/cache/justin_timberlake_{embeddings,timestamps}.npy`.
- [ ] **Phase 2: Signal Processing** — `signal_processing.py` deltas/median/MAD on Phase-1 fixture; lock the timestamp-alignment invariant with a synthetic test; lazy-import PELT as opt-in.
- [ ] **Phase 3: Clip Selection** — `clip_selection.py` peaks/padding/build/merge/budget; the single index→seconds conversion site; thread `peak_time` through merge.
- [ ] **Phase 4: Export** — `export.py` ffmpeg cut (`-c copy`) and concat (demuxer with concat-filter re-encode fallback); own all subprocess calls.
- [ ] **Phase 5: Orchestration & First-Video End-to-End** — `pipeline.py` glue + JSON §8 schema + determinism env vars; tune `--height`/`--min-gap-sec`/`--merge-gap-sec` on the most representative video and freeze; produce one watchable reel.
- [ ] **Phase 6: Multi-Video Run & Submission Polish** — run remaining 4 videos with frozen parameters; `run_all.sh`; README with design rationale, known limitations (spec §11), per-video qualitative observations.

---

## Phase Details

### Phase 1: Frame Extraction & Embeddings
**Goal:** Produce the canonical `(timestamps, embeddings)` artifact from one sample video, with frame-extraction and embedding pitfalls neutralized so every downstream phase trusts its input.
**Depends on:** Nothing (gating phase; unblocks all parallel paths).
**Requirements:** ENV-01, ENV-02, ENV-03, ENV-04, EXTR-01, EXTR-02, EXTR-03, EXTR-04, EXTR-05, EMBD-01, EMBD-02, EMBD-03, EMBD-04, EMBD-05, EMBD-06
**Success Criteria** (what must be TRUE):
  1. `pipeline.py --help` runs and the project layout (`extract.py`, `signal_processing.py`, `clip_selection.py`, `export.py`, `utils.py`, `pipeline.py`, `output/{reels,clips,timestamps}/`) matches spec §1; `requirements.txt` installs cleanly on Python 3.11+ and `ffmpeg -version` succeeds at startup (ENV-01..04).
  2. Running `extract.sample_frames(<sample.mp4>)` prints the sampled frame count and the first 5 timestamps; the timestamps are spaced ≈0.5s apart **and come from `cap.get(CAP_PROP_POS_MSEC)`, not nominal frame index** (handles VFR — Pitfall 2); video duration is sourced from `ffprobe`, not `CAP_PROP_FRAME_COUNT` (Pitfall 3); `assert abs(timestamps[-1] - ffprobe_duration) < 1.0` passes (EXTR-01..05).
  3. Running `extract.embed_frames(...)` on the same video prints embedding shape `(N, 768)` and the **hard assertion** `np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)` passes — explicit L2 normalization is in the code, not assumed from `open_clip` (Pitfall 6); `model.eval()` and `torch.inference_mode()` wrap inference (Pitfall 4); a 32-batched code path is the only path used (Pitfall 7); embedding the same frame twice in one process produces bit-identical output (EMBD-01..06).
  4. The §0.5 print-and-assert verification block at the end of `extract.py` runs successfully on at least one of the five sample videos and emits `(N, 768)` + L2-norm pass to stdout — this is the gate to start Phase 2.
**Plans:** 2 plans (both complete)
- [x] 01-01-PLAN.md — Project skeleton, pinned deps, gitignore, and utils.py (probe_video_metadata, ensure_output_dirs, setup_logger). Covers ENV-01..04. SUMMARY: 01-01-SUMMARY.md
- [x] 01-02-PLAN.md — extract.py: sample_frames (CAP_PROP_POS_MSEC), load_model, embed_frames (batched + L2-asserted), §0.5 verification harness with bit-identical rerun and --save-fixture writer. Covers EXTR-01..05, EMBD-01..06. SUMMARY: 01-02-SUMMARY.md

### Phase 2: Signal Processing
**Goal:** Convert embeddings to a clean per-sample `scores` array in index space, with the timestamp-alignment invariant locked and verified, and with PELT available as an opt-in path that does not perturb the default pipeline.
**Depends on:** Phase 1 (consumes `(timestamps.npy, embeddings.npy)` fixture; can develop against the fixture without re-running CLIP).
**Requirements:** SIGD-01, SIGD-02, SIGD-03, SIGS-01, SIGS-02, SIGM-01, SIGM-02, SIGM-03, SIGM-04, SIGP-01, SIGP-02, SIGP-03
**Success Criteria** (what must be TRUE):
  1. Running `signal_processing.compute_deltas(embeddings)` prints the first 20 raw delta values and they fall in the typical range 0.0–0.3; the returned array has length `N-1`; a `score_index_to_timestamp(i, ts) → ts[i+1]` helper exists and is the **only** site that converts score indices to seconds (SIGD-01..03).
  2. A synthetic two-color-video test (frames 0..K-1 = color A, frames K..end = color B at 2 fps) round-trips through `compute_deltas → smooth_deltas → mad_normalize` and the detected peak's timestamp matches `K × 0.5` to within 0.01s — this is the answer to "how do you actually verify `scores[i] ⟺ timestamps[i+1]`" (SIGD-02 + alignment invariant).
  3. Comparing raw vs `smooth_deltas(raw, kernel_size=5)` shows no isolated single-sample spikes survive while sustained high-delta periods remain visible; edge handling does not produce a phantom dip in the first/last 2 samples (Pitfall 9 — use `scipy.ndimage.median_filter(mode='reflect')` or pad-then-`medfilt`) (SIGS-01..02).
  4. Running `mad_normalize(smoothed, window_samples=180)` prints min/max/mean of normalized scores; max ≥ 2.0 (window not too large) and <90% of samples are above 3.0 (window not too small); the zero-MAD branch is gated at `mad > 1e-3` (not `1e-8` — Pitfall 10) and the percentage of zero-MAD samples is printed and is <5% on a typical sample video (SIGM-01..04).
  5. With `--pelt` off, `ruptures` is never imported (verified by inspecting `sys.modules` after a non-PELT run); with `--pelt` on, `detect_changepoints(smoothed, penalty=3.0)` returns a list of indices via `ruptures.Pelt(model="rbf")` and the import happens lazily inside the function (SIGP-01..03).
**Plans:** 1 plan
- [ ] 02-01-PLAN.md — signal_processing.py: compute_deltas + smooth_deltas (ndimage.median_filter mode=reflect) + mad_normalize (1e-3 MAD floor) + detect_changepoints (lazy ruptures) + score_index_to_timestamp helper, plus §0.5 verification harness with synthetic two-color alignment test and JT scores fixture writer. Covers SIGD-01..03, SIGS-01..02, SIGM-01..04, SIGP-01..03.

### Phase 3: Clip Selection
**Goal:** Convert `(scores, timestamps)` into a final ordered list of `(start_sec, end_sec, score, peak_time)` clips that fits the duration budget, with index→seconds conversion isolated to this module and `peak_time` carried through merging so JSON consumers can rely on `start ≤ peak_timestamp_sec ≤ end`.
**Depends on:** Phase 1 (timestamps), Phase 2 (scores). Can develop in parallel against a hand-authored `scores.npy` if needed (research §6).
**Requirements:** SELP-01, SELP-02, SELP-03, SELP-04, SELB-01, SELB-02, SELB-03, SELB-04, SELB-05, SELB-06
**Success Criteria** (what must be TRUE):
  1. `select_peaks(scores, timestamps, height, min_gap_sec, fps=2.0)` calls `scipy.signal.find_peaks` with `distance=int(min_gap_sec * fps)` (Pitfall 20: distance is in samples, not seconds); the post-call assertion `min(np.diff(peaks)) >= min_gap_samples` passes; peaks come back sorted by descending score and are printed with timestamps and scores before budget enforcement (SELP-01, SELP-02, SELP-04).
  2. With `--pelt` active, peaks within ±5 samples of a changepoint receive a 1.2× score boost and the result is re-sorted descending; with `--pelt` off, the boost path is never invoked (SELP-03).
  3. `compute_budget_seconds(dur)` returns `(dur/1800)*60` and `compute_padding(budget)` returns `max(3.0, min(8.0, budget*0.15))`; `build_clips` uses `peak_time = timestamps[idx + 1]` (the alignment helper, not `timestamps[idx]`) and clamps to `[0, video_duration_sec]` (SELB-01..03).
  4. `merge_clips(clips, gap_threshold_sec=3.0)` carries `peak_time` as a 4th tuple element through the merge so the merged clip retains the higher-scoring peak's timestamp (Pitfall 19); a hard assertion `start_sec ≤ peak_timestamp_sec ≤ end_sec` passes for every merged clip (SELB-04).
  5. `enforce_budget(merged, budget_sec)` greedily selects by score descending, uses partial-clip logic that **centers on `peak_time`** when the remainder ≥3s (Pitfall 18), and the final list is re-sorted chronologically; the §0.5 print shows final clips (start, end, duration, score), `total ≤ budget` is asserted, and no two clips overlap (SELB-05..06).
**Plans:** TBD

### Phase 4: Export
**Goal:** Cut individual clips losslessly with `ffmpeg -c copy` and concatenate them into a single highlight reel, with the concat demuxer's known fragility neutralized by validation and a re-encode fallback. All `subprocess` calls live here; the rest of the pipeline stays pure-Python.
**Depends on:** None for code structure (can develop against original sample video + a hand-authored `final_clips` fixture per research §6); only Phase 3's contract for integration.
**Requirements:** EXPC-01, EXPC-02, EXPC-03
**Success Criteria** (what must be TRUE):
  1. `extract_clip(input, output, start, end)` invokes `ffmpeg -y -ss <start> -to <end> -i <input> -c copy <output>` (`-ss` *before* `-i` for stream-copy correctness — Pitfall 11); after each call, `os.path.getsize(output) > 0` is asserted (EXPC-01).
  2. `concat_clips(clip_paths, output, temp_dir)` writes a concat manifest with `os.path.abspath` paths and quote-sanitized stems (Pitfall 13), invokes `ffmpeg -y -f concat -safe 0 -i <manifest> -c copy <output>`, and pre-validates clips with `ffprobe` (codec/timebase consistency check — Pitfall 12); on demuxer failure, falls back to the concat-filter re-encode path (`-c:v libx264 -crf 18`) and logs the fallback (EXPC-02).
  3. Running export against a hand-authored 3-clip list on one sample video produces 3 non-zero-byte intermediate clips and a single concatenated reel that plays end-to-end without frozen frames at clip boundaries (the §0.5 verification: "first clip is the correct segment when played; final reel is a coherent concatenation") (EXPC-03).
**Plans:** TBD

### Phase 5: Orchestration & First-Video End-to-End
**Goal:** Wire the four module phases together in `pipeline.py`, emit the JSON §8 manifest with the strict three-state PELT field, lock determinism env vars at entry, and produce **one watchable reel** on the most representative sample video — at which point parameters are tuned once and frozen for Phase 6. This phase is the spec footer's gating moment: "Get one video working end to end before touching the others."
**Depends on:** Phase 1, Phase 2, Phase 3, Phase 4 (strict serial integration; no parallelism).
**Requirements:** ORCH-01, ORCH-02, ORCH-03, ORCH-04, JSON-01, JSON-02, JSON-03, JSON-04, DOC-03
**Success Criteria** (what must be TRUE):
  1. `python pipeline.py <video.mp4>` accepts the positional `video` argument and `--pelt` / `--height` (default 1.5) / `--min-gap-sec` (default 15.0) / `--merge-gap-sec` (default 3.0) flags; prints stage banners `[1/6] … [6/6]` matching spec §9; creates `output/reels`, `output/clips/{video_name}/`, `output/timestamps/` on demand (ORCH-01..03).
  2. A reel is written to `output/reels/{video_name}_highlight.mp4` and is **watchable end-to-end** for the chosen representative video; manually scrubbing through original vs reel confirms the selected moments visibly correspond to high-action / significant scene changes rather than camera pans or walking (ORCH-04, qualitative spec §0.5 final step).
  3. `output/timestamps/{video_name}.json` matches spec §8 field-for-field (`video`, `video_duration_sec`, `budget_sec`, `total_reel_duration_sec`, `embedding_model`, `sampling_fps`, `clips[]`); each clip has all eight required fields including `peak_timestamp_sec`, `mad_score`, `raw_cosine_delta`, `coincides_with_pelt_changepoint`; the `start_sec ≤ peak_timestamp_sec ≤ end_sec` assertion passes for every clip (alignment invariant landing in JSON — JSON-01..02, JSON-04).
  4. Running with `--pelt` produces JSON where `coincides_with_pelt_changepoint` is `true` or `false` per clip; running without `--pelt` produces JSON where it is `null` (Python `None` → JSON `null`) for every clip — strict three-state, never omitted, never `false`-as-default (JSON-03; Pitfall 17).
  5. Running the pipeline twice on the same video with the same flags produces byte-identical JSON output: determinism env vars (`OMP_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)`) are set at pipeline entry; JSON floats are rounded (3 dp seconds, 4 dp scores) for byte-stable diffs (Pitfalls 14, 16; ROBU-02/03 are v2, but the env-var stanza belongs here to make the repro check possible).
  6. Final tuned values for `--height`, `--min-gap-sec`, `--merge-gap-sec` are recorded (in code defaults and a NOTES paragraph for Phase 6's README); these values are **frozen** before Phase 6 begins (DOC-03).
**Plans:** TBD

### Phase 6: Multi-Video Run & Submission Polish
**Goal:** Run the frozen-parameter pipeline on the remaining 4 sample videos, produce all 5 reels and 5 JSON manifests, and write a README that defends the design choices, acknowledges limitations honestly, and reports per-video qualitative observations — the only credible evaluation given no labeled ground truth.
**Depends on:** Phase 5 (must have one watchable reel and frozen parameters before batch-running).
**Requirements:** RUN-01, RUN-02, RUN-03, DOC-01, DOC-02
**Success Criteria** (what must be TRUE):
  1. A `run_all.sh` (or in-process loop that loads CLIP once) processes all 5 sample videos in `videos/` back-to-back with **the same fixed parameter set** chosen in Phase 5 — no per-video flag overrides (RUN-01..02; spec §6, §12).
  2. After the batch run, `output/reels/` contains 5 `_highlight.mp4` files and `output/timestamps/` contains 5 matching `.json` files; each reel plays end-to-end and each JSON validates against the spec §8 schema (RUN-03).
  3. Watching all 5 reels (qualitative — the only available eval): selected moments on at least the representative video and a majority of the others visibly correspond to high-action or significant scene changes; on videos where the reel is dominated by camera pans or lighting transitions, this is **honestly documented** in the README rather than glossed over (spec §11 framing) (DOC-01).
  4. README documents installation, runtime, the rationale for each frozen parameter (linking to spec §-numbers), per-video qualitative observations including failure modes, and explicitly acknowledges the §11 known limitations (visually-distinct ≠ important; classification→delta robustness gap; no labeled GT) and the §12 non-goals it deliberately does not address (DOC-01..02).
**Plans:** TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Frame Extraction & Embeddings | 0/2 | Planned | - |
| 2. Signal Processing | 0/1 | Planned | - |
| 3. Clip Selection | 0/0 | Not started | - |
| 4. Export | 0/0 | Not started | - |
| 5. Orchestration & First-Video End-to-End | 0/0 | Not started | - |
| 6. Multi-Video Run & Submission Polish | 0/0 | Not started | - |

---

## Coverage

**v1 requirements:** 51 mapped to 6 phases.

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1 | ENV-01..04, EXTR-01..05, EMBD-01..06 | 15 |
| 2 | SIGD-01..03, SIGS-01..02, SIGM-01..04, SIGP-01..03 | 12 |
| 3 | SELP-01..04, SELB-01..06 | 10 |
| 4 | EXPC-01..03 | 3 |
| 5 | ORCH-01..04, JSON-01..04, DOC-03 | 9 |
| 6 | RUN-01..03, DOC-01..02 | 5 |
| **Total** | | **54 line-items / 51 unique REQ-IDs** ✓ |

(DOC-03 is mapped to Phase 5 because parameter freezing is the gate that lets Phase 6 begin; it is documented in the Phase 6 README per DOC-01.)

All 51 v1 requirements are mapped to exactly one phase. No orphans. No duplicates.

---

## Phase Ordering Rationale

The spec footer locks the build order: `extract.py → signal_processing.py → clip_selection.py → export.py → pipeline.py`. Phase 1 → 2 → 3 → 4 → 5 follows that order verbatim. Phase 6 is the spec footer's "before touching the others" boundary made explicit — batch-processing comes only after one video produces a watchable reel.

The research SUMMARY notes that Phases 2/3/4 are technically parallelizable once Phase 1 ships its fixture. With a single builder (Claude), the parallelism is not exercised; we serialize per the spec footer. The phase boundaries still reflect the parallel structure so that if it ever became useful (e.g., regenerating fixtures during debugging), the seams are already there.

`--pelt` is deliberately threaded across phases as opt-in supplementary signal rather than carved into its own phase: detection in Phase 2 (lazy-imported), boost application in Phase 3, JSON three-state field in Phase 5. This keeps it off the critical path — the baseline (`--pelt` off) is the gating success criterion.

---

*Last updated: 2026-05-06 at roadmap creation*
