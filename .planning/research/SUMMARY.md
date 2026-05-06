# Project Research Summary

**Project:** Body Cam Highlight Reel — AbelPolice Take-Home
**Domain:** Visual-embedding video highlight extraction (Python CLI; CLIP ViT-L/14 → 1D MAD/PELT → ffmpeg cut+concat)
**Researched:** 2026-05-06
**Confidence:** HIGH

## Executive Summary

This is a **single-process Python CLI prototype** for a take-home assignment whose design spec at `assignment-details/bodycam_highlight_reel_spec.md` is **locked**. The spec already prescribes the model (CLIP ViT-L/14 OpenAI/QuickGELU), the algorithmic chain (2 fps sampling → cosine deltas → 5-tap median → 90s rolling MAD → `find_peaks` → adaptive padding → merge → budget → `-c copy` concat), the JSON schema (§8), and the module layout (5 modules + orchestrator). Research scope was therefore validation, version pinning, and pitfall enumeration — not redesign.

**The recommended approach is to implement the spec verbatim**, with three documentation/dependency corrections: (1) drop `ffmpeg-python` from `requirements.txt` — it is unmaintained since 2019 and the spec's export code already uses raw `subprocess.run` instead; (2) pin `scipy==1.16.2` to dodge a `medfilt` correctness regression in scipy 1.15 ([#22333](https://github.com/scipy/scipy/issues/22333)) that affects exactly the kernel-size-5 filter the spec uses, with `scipy.ndimage.median_filter(mode='reflect')` available as a behaviorally-equivalent fallback that also fixes edge-effect phantom peaks at video boundaries; (3) lazy-import `ruptures` inside `detect_changepoints()` so non-`--pelt` runs stay clean. Architecturally, the 5-module split is correct and parallelizable: once `extract.py` ships an `embeddings.npy` fixture from one sample video, `signal_processing.py` / `clip_selection.py` / `export.py` can be built independently against synthetic fixtures, then integrated by `pipeline.py`.

**The dominant risks are not algorithmic — they are integration seams** between OpenCV, open_clip, scipy, and ffmpeg. The 20 catalogued pitfalls cluster into four categories that must drive verification: (a) **frame-extraction correctness** — BGR/RGB swap, VFR seek inaccuracy via `CAP_PROP_POS_FRAMES`, lying `CAP_PROP_FRAME_COUNT`, missing `model.eval()`, missing explicit L2 normalize; (b) **timestamp alignment invariant** — `scores[i]` must map to `timestamps[i+1]`, verified by a synthetic two-color-video test, threaded carefully through the merge step which currently discards peak time; (c) **ffmpeg keyframe-alignment imprecision** with `-c copy` and concat-demuxer fragility, requiring pre-concat `ffprobe` validation and a re-encode fallback; (d) **reproducibility hygiene** — determinism env vars, JSON float rounding, and a strict three-state (`null`/`true`/`false`) rule for `coincides_with_pelt_changepoint`.

## Key Findings

### Recommended Stack

The spec's stack choices are still correct in 2026 with the three documented exceptions above. Total install footprint is ~1.3 GB plus an ~890 MB CLIP weight download to `~/.cache/huggingface/hub/` on first run. CPU-only is acceptable; on this machine (macOS 15.3.2 arm64, ffmpeg 7.1.1, Python 3.13.2) MPS is available and gives ~3–5× speedup if desired via a single optional `device` constant — without violating the spec's "no GPU-specific code paths" rule because device selection falls through to CPU when MPS/CUDA aren't built.

**Core technologies:**
- **`torch==2.11.0` + `torchvision==0.26.0`** — lockstep-pinned; mismatch is the #1 install-time CLIP failure mode.
- **`open_clip_torch==3.3.0`** — canonical loader; `('ViT-L-14', pretrained='openai')` forces `quick_gelu=True` via the registry.
- **`opencv-python==4.13.0.92`** — frame sampling. **Caveat:** prefer `CAP_PROP_POS_MSEC` time-based seek and read actual `cap.get(...)` timestamps — VFR body cams break frame-index seek.
- **`scipy==1.16.2`** — `signal.medfilt`, `signal.find_peaks`, `stats.median_abs_deviation`. **Pin away from 1.15** (medfilt regression). `scipy.ndimage.median_filter(size=5, mode='reflect')` is a behaviorally-equivalent fallback that also fixes Pitfall 9.
- **`numpy>=2.1,<2.5`**, **`ruptures==1.1.10`** (lazy-imported inside `detect_changepoints`), **`tqdm>=4.67,<5`**.
- **`ffmpeg>=4.4` system binary** (recommend `>=6.0`); local has 7.1.1.

**Removed from spec §1:** `ffmpeg-python` (kkroening) — unmaintained since 2019-07-06; spec's own export code uses `subprocess.run` directly. Drop from `requirements.txt`.

### Expected Features

This is a take-home with a single reviewer. ~20 spec-mandated table stakes (P1, all required), ~17 differentiators (P2, pick 6–10), 18 anti-features (P0, actively avoid).

**Must have (table stakes, P1):**
- 2 fps sampling + CLIP ViT-L/14 OpenAI batch-32 embeddings, L2-normalized
- Cosine deltas → 5-tap median → 180-sample rolling MAD (clipped [0,10])
- `find_peaks` with `--height` (1.5) + `--min-gap-sec` (15), adaptive padding `max(3, min(8, budget×0.15))`, budget `(dur/1800)×60`
- Build → merge (3s gap) → enforce-budget greedy by score
- `ffmpeg -c copy` per-clip + concat demuxer
- JSON output matching §8 field-for-field (incl. `coincides_with_pelt_changepoint: null` when `--pelt` off)
- Locked module layout, `pipeline.py <video> [--pelt] [--height H] [--min-gap-sec G]`
- Per-module §0.5 verification prints
- All 5 sample videos with one fixed parameter set
- README with run instructions; `requirements.txt`

**Should have (differentiators, P2 — pick 6–10):**
- README "Design Rationale" linking each parameter to spec
- README "Known Limitations" mirroring spec §11
- README "Per-Video Notes" qualitative writeup (only credible eval given no GT)
- `run_all.sh` (or in-process loop that loads CLIP once)
- Optional `--diagnostics` PNG plots
- Embedding cache `output/cache/{video}_embeddings.npy`
- Per-phase timing logs, type hints, argparse `help=` rationale strings

**Defer / actively avoid (P0 anti-features):**
- Audio / ASR / OCR / LLM / manual selection (§12 violations)
- Per-video retuning, fine-tuning CLIP, model substitution
- Global fixed threshold (Antonio's original failure mode)
- Default libx264 re-encode
- Web UI / Streamlit
- Quantitative precision/recall (no labeled GT)
- Configurable embedding model or JSON schema

### Architecture Approach

The spec's 5-module + `pipeline.py` orchestrator split is **the right granularity** and should not be refactored. Data flow is a strict linear DAG with one orthogonal branch (`--pelt`). All inter-module contracts are plain numpy arrays + lists of tuples + Python primitives — no shared mutable state, no classes. **The single most important rule: `signal_processing.py` works in index space only; `clip_selection.py` is the single place where `timestamps[idx+1]` converts indices to seconds.** This eliminates a whole class of off-by-one bugs.

**Major components:**
1. **`extract.py`** — video → `(frames, timestamps)` → `embeddings (N, 768) float32 L2-normalized`. Owns cv2 + torch + open_clip. Must accept `(model, preprocess)` as parameters so multi-video runs amortize the ~30s model load.
2. **`signal_processing.py`** — `embeddings → raw_deltas → smoothed → scores`, plus optional `changepoints`. Pure numpy/scipy, deterministic, no I/O. **Lazy-import `ruptures` inside `detect_changepoints()`.**
3. **`clip_selection.py`** — `(scores, timestamps) → peak_indices → (boosted) → clips → merged → final_clips`. Carries `peak_time` as a 4th tuple element through merge so `peak_timestamp_sec` stays inside `[start_sec, end_sec]` for merged clips. Centers partial clips on peak when budget remainder ≥3s.
4. **`export.py`** — owns all `subprocess.run(['ffmpeg', ...])` calls. `-ss`-before-`-i` `-c copy` per clip; concat demuxer with `-safe 0` and absolute paths; pre-concat `ffprobe` validation; concat-filter re-encode fallback.
5. **`utils.py`** — load-bearing. `probe_video_metadata` (`ffprobe`-based, NOT `CAP_PROP_FRAME_COUNT`), `ensure_output_dirs`, `write_timestamps_json`. Single source of truth for video duration.
6. **`pipeline.py`** — thin orchestrator: argparse + `run()` + JSON assembly. **No algorithmic logic.** Determinism env-vars at entry, JSON float rounding, strict three-state rule for `coincides_with_pelt_changepoint`.

### Critical Pitfalls

The five with highest blast radius (each capable of silently invalidating reels with no exception):

1. **VFR seek inaccuracy + lying frame counts (Pitfalls 2, 3)** — `CAP_PROP_POS_FRAMES` lands on nearest preceding keyframe on body-cam VFR MP4s; `CAP_PROP_FRAME_COUNT` is unreliable. **Fix:** drive sampling with `CAP_PROP_POS_MSEC` + `cap.get(...)` for actual timestamps; use `ffprobe -show_entries format=duration` for duration. Verify `assert abs(timestamps[-1] - ffprobe_duration) < 1.0`.
2. **L2-normalization not enforced + `model.eval()` forgotten (Pitfalls 4, 6)** — open_clip's `encode_image` does NOT normalize by default; spec's `np.clip(dots, -1, 1)` then masks the bug, flatlining deltas. **Fix:** explicit `feats / feats.norm(dim=-1, keepdim=True)` plus hard assertion (not print) on `np.allclose(norms, 1.0, atol=1e-5)`. `model.eval()` immediately before the inference loop, with `torch.inference_mode()`.
3. **Off-by-one delta-to-timestamp alignment (Pitfall 8)** — `scores[i]` aligns to `timestamps[i+1]`. Spec calls it out but doesn't say how to verify. **Fix:** single `score_index_to_timestamp(idx, ts) → ts[idx+1]` helper used everywhere, plus a synthetic two-color-video test that constructs a known transition at frame K and asserts the detected peak timestamp matches `K × 0.5` to within 0.01s.
4. **`medfilt` edge effects + MAD-zero windows (Pitfalls 9, 10)** — `scipy.signal.medfilt` zero-pads boundaries (phantom peaks in first/last 1.5s); `mad > 1e-8` floor is too lenient (static-footage windows produce ceiling-pegged scores). **Fix:** prefer `scipy.ndimage.median_filter(size=5, mode='reflect')`; raise MAD floor to `1e-3`; print percentage of zero-MAD samples in §0.5.
5. **ffmpeg keyframe imprecision + concat fragility (Pitfalls 11–13)** — `-ss` before `-i` with `-c copy` seeks to nearest preceding keyframe (always earlier); concat demuxer rejects mismatched timebases/codec params; manifests break on quotes or relative paths. **Fix:** keep `-ss` before `-i` (correct for stream copy, accept ~1s imprecision); pre-validate with `ffprobe`; sanitize stems; always `os.path.abspath` and `-safe 0`; concat-filter fallback on demuxer failure.

## Implications for Roadmap

**Cross-cutting invariant.** The timestamp alignment rule (`scores[i] ⟺ timestamps[i+1]`) crosses three modules and the JSON output. Every roadmap phase must reference it: locked in Phase 2 (helper + synthetic test), used in Phase 3 (`build_clips`, merge), reflected in Phase 5 (JSON `peak_timestamp_sec` assertion `start_sec ≤ peak_timestamp_sec ≤ end_sec`).

**Build-order parallelism.** Spec footer locks serial order `extract → signal_processing → clip_selection → export → pipeline`. **For parallel work, once `extract.py` ships `(embeddings.npy, timestamps.npy)` from one sample video, the middle three modules become independent.** This is the parallelism the spec footer hides.

### Phase 1: Frame Extraction & Embedding Pipeline (`extract.py` + `utils.py` core)
**Rationale:** Unblocks every downstream phase by producing the canonical fixture. Highest pitfall density (8 of 20). Must be solidly verified before chain extends.
**Delivers:** `sample_frames`, `embed_frames(frames, model, preprocess)`, `utils.probe_video_metadata` (ffprobe), `utils.ensure_output_dirs`. First HF Hub weight download.
**Avoids:** Pitfalls 1 (BGR/RGB), 2 (`CAP_PROP_POS_MSEC`), 3 (`ffprobe` for duration), 4 (`model.eval` + `inference_mode`), 5 (`Image.fromarray` + shape assert), 6 (explicit L2 normalize + assert), 7 (single batched path), 15 (warm-up batch).

### Phase 2: Signal Processing — Deltas, Smoothing, MAD (`signal_processing.py`)
**Rationale:** Pure numpy/scipy on the Phase 1 fixture. Locks the timestamp-alignment invariant.
**Delivers:** `compute_deltas`, `smooth_deltas`, `mad_normalize`, lazy-imported `detect_changepoints`. Establishes index-space-only convention.
**Avoids:** Pitfalls 8 (alignment helper + synthetic test), 9 (`mode='reflect'`), 10 (MAD floor `1e-3` + diagnostic count).
**Parallel with:** Phase 3, Phase 4.

### Phase 3: Clip Selection — Peaks, Padding, Merge, Budget (`clip_selection.py`)
**Rationale:** Pure logic on a `scores.npy` fixture (real or synthetic). Single place for index→seconds conversion. Carries `peak_time` through merge.
**Delivers:** `select_peaks`, `apply_pelt_boost`, `compute_padding`, `compute_budget_seconds`, `build_clips`, `merge_clips`, `enforce_budget`. Output: `final_clips: list[(start, end, score, peak_time)]`.
**Avoids:** Pitfalls 18 (partial clip centered on peak), 19 (`peak_time` propagated + `start ≤ peak_ts ≤ end` assert), 20 (`find_peaks` distance unit conversion + post-call assert).
**Parallel with:** Phase 2, Phase 4.

### Phase 4: Export — ffmpeg Cut & Concat (`export.py`)
**Rationale:** Owns all subprocess shells. Independently developable against a hand-authored `final_clips` list and the original video.
**Delivers:** `extract_clip` (`-ss`-before-`-i` `-c copy`), `concat_clips` (concat demuxer + `ffprobe` validation + concat-filter re-encode fallback).
**Avoids:** Pitfalls 11 (`-ss` placement), 12 (codec/timebase validation + re-encode fallback), 13 (absolute paths, `-safe 0`, stem sanitization).
**Parallel with:** Phase 2, Phase 3.

### Phase 5: Orchestration, JSON, Determinism, End-to-End on Video 1 (`pipeline.py` + `utils.write_timestamps_json`)
**Rationale:** Cannot start until 2–4 done. Convergence point. Spec footer: "Get one video working end-to-end before touching the others."
**Delivers:** `pipeline.run`, argparse, `[1/6]…[6/6]` progress, JSON §8 schema assembly, determinism env vars at entry.
**Avoids:** Pitfalls 14 (determinism env vars), 16 (JSON float rounding to 3/4 decimals), 17 (strict three-state for `coincides_with_pelt_changepoint`).

### Phase 6: Multi-Video Run, README, Submission Polish
**Rationale:** Only after one video produces a watchable reel. Run remaining 4 with frozen parameters; tune once on the most representative video; write README.
**Delivers:** `run_all.sh` (or in-process loop), 5 reels + 5 JSONs, README with design rationale + limitations + per-video notes, optional differentiators.

### Phase Ordering Rationale

- **Phase 1 is the bottleneck and gating phase** — no fixture, no parallelism. Pitfall density justifies extra §0.5 verification time here.
- **Phases 2–4 are independent given fixtures.** Single contributor still serializes 2 → 3 → 4, but Phase 4 can start anytime since it needs only the original video and a hand-authored clip list.
- **Phase 5 is strict serial integration** — defer `--pelt` to *after* baseline works so the JSON's `null` branch is locked before adding `true`/`false`.
- **Phase 6 separates "works" from "submission-ready"** — don't batch-process before the first reel is watched.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Frame seeking on the actual 5 sample videos in `videos/sample-vids.zip`. Need `ffprobe -show_streams` to confirm CFR vs VFR and validate the time-based seek path before locking it.
- **Phase 4:** Concat demuxer behavior on real extracted clips — fallback path is documented but untested against actual body-cam keyframe density. One-spike test on video 1 early.

Phases with standard patterns (skip research):
- **Phase 2** (numpy/scipy 1D — well-documented; pitfalls already catalogued).
- **Phase 3** (pure Python list/tuple manipulation; only subtlety is `peak_time` propagation, captured in ARCHITECTURE.md §3.3).
- **Phase 5** (glue code).
- **Phase 6** (documentation + shell).

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Versions verified against PyPI 2026-05-06; `open_clip` registry verified against `mlfoundations/open_clip` main; scipy regression verified against issue #22333; local environment probed directly. |
| Features | HIGH | Every feature traces to a numbered spec section or §12 prohibition. Spec is locked. |
| Architecture | HIGH | Module boundaries, contracts, and `--pelt` placement explicit in spec. Parallelism analysis derived mechanically from data flow. |
| Pitfalls | HIGH | 20 pitfalls drawn from documented behaviors of OpenCV, open_clip, scipy (incl. issue #22333), ffmpeg — all primary-source. |

**Overall confidence:** HIGH

### Gaps to Address

- **Sample video VFR/CFR status unknown.** Pitfalls research recommends `CAP_PROP_POS_MSEC` based on body-cam-encoder norms, but the actual 5 sample videos haven't been probed. **Handle in Phase 1:** `ffprobe -show_streams` on each; default to time-based seek either way (costs nothing, safer).
- **`utils.py` contents not in spec.** Recommended set (`probe_video_metadata`, `ensure_output_dirs`, `write_timestamps_json`) is derived from call sites. **Handle in Phase 1:** lock the API at start.
- **No labeled ground truth for any video.** Acknowledged in spec §11. **Handle in Phase 6:** lean into qualitative per-video README writeup; do not invent quantitative numbers.
- **Parameter tuning videography.** Spec doesn't say which video is most representative. **Handle in Phase 6:** after Phase 5 produces video 1's reel, watch all 5 raw videos and pick the one with broadest dynamic range as tuning target.

## Sources

### Primary (HIGH confidence)
- `assignment-details/bodycam_highlight_reel_spec.md` §§0–12 — locked design document.
- `.planning/PROJECT.md` — derived requirements + Out of Scope + locked module layout + build-order footer.
- PyPI release pages for all pinned packages (verified 2026-05-06).
- [`mlfoundations/open_clip` main `pretrained.py`](https://raw.githubusercontent.com/mlfoundations/open_clip/main/src/open_clip/pretrained.py) — `quick_gelu=True` registry entry.
- [scipy issue #22333](https://github.com/scipy/scipy/issues/22333) — `medfilt` regression in 1.15.
- OpenCV `VideoCapture` documentation — VFR/long-GOP limitations.
- ffmpeg wiki + ffmpeg-user mailing list — `-ss` semantics, concat demuxer requirements, `-safe 0`.
- PyTorch reproducibility documentation.
- Local environment probe (`uname -m`, `sw_vers`, `python3 --version`, `ffmpeg -version`).

### Secondary (MEDIUM confidence)
- Take-home submission conventions for ML/CV roles (convention rather than written rubric, but spec §11 + §12 bullet 6 endorse the framing).
- [PyTorch issue #167679](https://github.com/pytorch/pytorch/issues/167679) — MPS regression on macOS 26 / Tahoe; not relevant to current 15.3.2 machine.
- open_clip GitHub issues on inference reproducibility and `model.eval()` defaults (community-reported, consistent).

### Tertiary (LOW confidence)
- None — all findings traced to primary or secondary sources.

---
*Research completed: 2026-05-06*
*Ready for roadmap: yes*
