---
phase: 03
plan: 01
subsystem: clip_selection
tags: [phase-3, clip-selection, peak-detection, merge, budget-enforcement, pitfall-18, pitfall-19, pitfall-20]
dependency_graph:
  requires:
    - signal_processing.score_index_to_timestamp (D-24, D-45 — single index→seconds seam)
    - output/cache/{video}_scores.npy (Phase 2 fixture, float64)
    - output/cache/{video}_timestamps.npy (Phase 1 fixture, float64)
    - output/cache/{video}_changepoints.npy (optional, only with --pelt)
    - utils.probe_video_metadata (single source of truth for video duration)
    - utils.ensure_output_dirs (cache fixture write target)
  provides:
    - clip_selection.select_peaks
    - clip_selection.apply_pelt_boost
    - clip_selection.compute_budget_seconds
    - clip_selection.compute_padding
    - clip_selection.build_clips
    - clip_selection.merge_clips
    - clip_selection.enforce_budget
    - output/cache/{video}_final_clips.json (Phase 4 dev fixture)
  affects:
    - Phase 4 export.py — consumes 4-tuple list[(start, end, score, peak_time)]
    - Phase 5 pipeline.run() — orchestrates this module's seven functions
    - Phase 5 JSON §8 schema — peak_timestamp_sec field comes from peak_time, alignment invariant guaranteed
tech_stack:
  added:
    - scipy.signal.find_peaks (already pinned via scipy==1.16.2)
  patterns:
    - 4-tuple (start, end, score, peak_time) instead of spec §6's 3-tuple (D-38)
    - Partial clip centered on peak_time when remainder ≥ 3s (Pitfall 18 fix; deviation from spec §6)
    - Imported `score_index_to_timestamp` helper as the SINGLE index→seconds conversion seam (D-45 / Pitfall 8)
    - Inline `int(min_gap_sec * fps)` in find_peaks call site so distance unit is visible (Pitfall 20)
    - Hard runtime assertions for `start ≤ peak_time ≤ end` post-merge AND post-budget (Pitfall 19)
key_files:
  created: []
  modified:
    - clip_selection.py (replaced one-line stub with 311+252=563 line implementation)
    - .planning/STATE.md (phase 3 mark complete, metrics, decisions, session continuity)
    - .planning/ROADMAP.md (Phase 3 row → Complete; Plan 03-01 link → SUMMARY)
    - .planning/REQUIREMENTS.md (10 SELP/SELB checkboxes ticked, Phase 3 traceability rows → Complete)
  generated:
    - output/cache/justin_timberlake_final_clips.json (4 clips, 548 bytes; gitignored)
decisions:
  honored:
    - D-38 (4-tuple peak_time threading)
    - D-39 (raw tuples, not NamedTuple)
    - D-40 (select_peaks with int(min_gap_sec * fps) + post-call gap assertion)
    - D-41 (peak print with idx/ts/score; first 10)
    - D-42 (apply_pelt_boost pure index-space; opt-in only)
    - D-43 (compute_budget_seconds = (dur/1800)*60)
    - D-44 (compute_padding = max(3.0, min(8.0, budget*0.15)))
    - D-45 (build_clips imports score_index_to_timestamp; no inline timestamp arithmetic)
    - D-46 (merge_clips: peak_time follows higher score; tie-break = earlier peak_time; hard assert)
    - D-47 (enforce_budget: greedy by score; partial centered on peak_time; 0.95 squeeze fallback; assertions)
    - D-48 (final clip print with start/end/dur/score/peak_time + budget total + percentage)
    - D-49 (10-step __main__ harness; argparse; smart default; sys.exit codes)
    - D-50 (--save-fixture writes output/cache/{video}_final_clips.json with indent=2)
    - D-51 (single-video focus per harness invocation)
  added: []
metrics:
  duration_min: ~6
  duration_sec: 337
  tasks: 2
  commits: 2
  files_changed: 1
  fixtures_written: 1
  completed_date: "2026-05-06"
---

# Phase 3 Plan 01: Clip Selection Summary

**One-liner:** Seven pure-Python/numpy/scipy functions (peak detection → adaptive padding → 4-tuple clip build → merge with peak_time threading → greedy budget enforcement with partial-clip centered on peak_time) plus a 10-step §0.5 harness that runs in 2.27 s on the JT fixture and writes a Phase 4 dev JSON fixture.

## What Was Built

**File:** `clip_selection.py` (replaces the one-line stub)

### Module Surface

| Function | Lines | Owner Pitfall | Purpose |
|----------|-------|---------------|---------|
| `select_peaks(scores, timestamps, height=1.5, min_gap_sec=15.0, fps=2.0)` | 47 | 20 | `scipy.signal.find_peaks` with `distance=int(min_gap_sec*fps)` SAMPLES; post-call gap assertion; descending-score sort |
| `apply_pelt_boost(peak_indices, peak_scores, changepoints, window=5, boost=1.2)` | 26 | — | Multiply score by 1.2 if peak within ±5 samples of any changepoint; pure index-space; re-sort descending; opt-in only |
| `compute_budget_seconds(video_duration_sec)` | 3 | — | `(dur / 1800.0) * 60.0` verbatim |
| `compute_padding(budget_sec, max_padding=8.0, min_padding=3.0)` | 9 | — | `max(3.0, min(8.0, budget * 0.15))` verbatim |
| `build_clips(peak_indices, peak_scores, timestamps, video_duration_sec, padding_sec)` | 39 | 8 | THE single index→seconds conversion seam; uses imported `score_index_to_timestamp`; clamps to [0, dur]; emits chronological 4-tuples |
| `merge_clips(clips, gap_threshold_sec=3.0)` | 47 | 19 | Merge if `start ≤ prev.end + gap`; peak_time follows higher score; tie-break earlier peak_time; hard asserts `start ≤ peak_time ≤ end` |
| `enforce_budget(merged_clips, budget_sec)` | 60 | 18 | Greedy by score; partial CENTERED on peak_time when remainder ≥ 3s with 0.95 squeeze fallback; hard asserts total ≤ budget+1e-6, no overlaps, alignment invariant |
| `__main__` (10-step §0.5 harness) | 230 | 18, 19, 20 | argparse → load fixtures → select_peaks → optional pelt boost → compute budget+padding → build → merge → enforce → synthetic test → write JSON → PASS banner |

### Module Imports (top-level — Pitfall guard scope)

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from signal_processing import score_index_to_timestamp
```

**FORBIDDEN at module top (verified absent):**
- `import ruptures` / `from ruptures` — Phase 3 receives changepoints as `list[int]`; ruptures was loaded in Phase 2 to PRODUCE the fixture, never at Phase 3 import time. `'ruptures' not in sys.modules` after `import clip_selection`.
- Inline `timestamps[idx + 1]` / `timestamps[idx+1]` / `timestamps[i + 1]` / `timestamps[i+1]` substrings — grep-banned across the entire file. The single conversion seam is `score_index_to_timestamp(idx, timestamps)` from `signal_processing`.

## Verification Results

### §0.5 Harness Run (`python clip_selection.py justin_timberlake`)

```
[fixture] loaded justin_timberlake: scores=(2268,) timestamps=(2269,)
[meta] duration=1134.144s codec=h264 is_vfr=False
[peaks] count=40
[peaks]   #0: idx=1812 ts=906.506s score=10.0000
[peaks]   #1: idx=1188 ts=594.494s score=9.4892
[peaks]   #2: idx=689 ts=345.011s score=6.5542
[peaks]   #3: idx=1502 ts=751.484s score=5.8053
[peaks]   #4: idx=516 ts=258.492s score=5.6535
[peaks]   #5: idx=13 ts=7.007s score=5.2830
[peaks]   #6: idx=808 ts=404.504s score=4.9502
[peaks]   #7: idx=294 ts=147.514s score=4.9034
[peaks]   #8: idx=1601 ts=801.000s score=4.4608
[peaks]   #9: idx=1955 ts=978.010s score=4.3779
[budget] duration=1134.144s -> budget=37.805s padding=5.671s
[build_clips] 40 candidate clips (chronological)
[merge] 40 -> 40 clips after gap_threshold=3.0s
[budget] total=37.805s / budget=37.805s (100.0%)
[clip-0] start=339.341 end=350.682 dur=11.341 score=6.5542 peak_time=345.011
[clip-1] start=588.823 end=600.165 dur=11.341 score=9.4892 peak_time=594.494
[clip-2] start=749.594 end=753.374 dur=3.780  score=5.8053 peak_time=751.484
[clip-3] start=900.835 end=912.176 dur=11.341 score=10.0000 peak_time=906.506
[synthetic] PASS  (peaks, build, merge, budget all chained correctly)
[fixture] wrote output/cache/justin_timberlake_final_clips.json
Phase 3 §0.5 verification: PASS
```

**Wall-clock:** 2.27 s real (well under the 5 s safety margin from CONTEXT specifics; CLIP-zero / numpy-light + one ffprobe call).

### ROADMAP Phase 3 Success Criteria — all 5 PASS

| SC | Check | Result |
|----|-------|--------|
| 1 | `select_peaks` uses `distance=int(min_gap_sec * fps)` (Pitfall 20); post-call `min(np.diff(peaks)) >= min_gap_samples`; sorted by descending score; printed before budget enforcement | PASS — JT 40 peaks @ height=1.5; min_gap_samples=30; observed min gap is well above 30; top peak is `idx=1812 score=10.0000`; bottom-of-top-10 is `idx=1955 score=4.3779`; descending order verified by index of monotonicity in `[peaks]` lines |
| 2 | With `--pelt` active, peaks within ±5 samples of changepoint get 1.2× boost and re-sort descending; with `--pelt` off, boost path never invoked | PASS — `--pelt` run prints `[pelt-boost] re-sorted; top score now 10.0000` (top peak's neighborhood already has a changepoint within ±5 so boosted from 10.0 cap-clipped); default run never traverses the `if args.pelt and changepoints is not None` branch |
| 3 | `compute_budget_seconds(dur)=(dur/1800)*60`, `compute_padding(budget)=max(3, min(8, budget*0.15))`, `build_clips` uses imported helper, clamps to `[0, video_duration]` | PASS — duration=1134.144s → budget=(1134.144/1800)*60=37.805s; padding=max(3, min(8, 37.805*0.15))=max(3, min(8, 5.671))=5.671s. `[build_clips] 40 candidate clips (chronological)`; right-edge clamp test (Test P) passes |
| 4 | `merge_clips` threads peak_time as 4th element; merged clip retains higher-scoring peak's timestamp (Pitfall 19); `start ≤ peak_time ≤ end` asserted for every merged clip | PASS — `[merge] 40 -> 40 clips`; on JT no clips merged at default 3s threshold (gaps too large), but unit Tests S/T/U/X exercise the merge logic with peak_time threading + tie-break + 3-way merge; runtime hard assert passes on every output clip |
| 5 | `enforce_budget` greedy by score; partial CENTERED on peak_time when remainder ≥ 3s (Pitfall 18); chronologically re-sorted; total ≤ budget asserted; no overlaps | PASS — `total=37.805s / budget=37.805s (100.0%)`; clip-2 is the partial (3.780s wide on a clip-2's source 11.341s clip), centered on `peak_time=751.484` so `start=749.594 ≤ 751.484 ≤ 753.374=end` (centered partial = peak ± remainder/2 = 751.484 ± 1.890); chronological order in stdout `clip-0..3` is monotonically increasing in `start` |

### Owned Pitfalls — all 3 neutralized

- **Pitfall 18 (partial-clip on peak):** Test AA passes with `partial=(22.5, 27.5, 5.0, 25.0)` — peak_time=25.0 inside; spec §6's truncation `(20.0, 25.0)` would have excluded it. Test AB passes with peak near left edge clamped to `(0.0, 5.0)` and peak=0.5 inside. JT live: clip-2 partial is `(749.594, 753.374)` width 3.78s centered on `peak_time=751.484` (peak ±1.890 = 749.594..753.374, exactly the centered window).
- **Pitfall 19 (peak_time through merge):** Test T (`[(0,5,9,2.5),(6,10,3,8)] → [(0,10,9,2.5)]` — peak_time=2.5 from higher-score=9), Test S (`[(0,5,2,2.5),(6,10,3,8)] → [(0,10,3,8)]` — peak_time=8 from higher-score=3), Test U tie-break (earlier peak_time wins), Test X 3-way merge → peak_time=7.5 (the highest score=5 in the chain). Hard runtime assert in `merge_clips` body fires before return on every clip.
- **Pitfall 20 (find_peaks distance unit):** Grep verifies `int(min_gap_sec * fps)` literal at the call site. Test B (gap=25 samples < distance=30 samples → only 1 peak survives) confirms find_peaks honors the samples contract. JT live: 40 peaks survive with min_gap_sec=15.0 over a 2,268-sample (1,134s) signal. Internal post-call assertion `min(np.diff(np.sort(peaks))) >= min_gap_samples` is reachable but never fires (find_peaks correctly enforces).

### Cross-Cutting Invariants

- **Single index→seconds seam:** Grep verifies `score_index_to_timestamp` is imported and inline `timestamps[idx + 1]` / `timestamps[i + 1]` / `+1`-style substrings are absent. Conversion happens exactly once in `build_clips`.
- **--pelt orthogonality:** `'ruptures' not in sys.modules` after `import clip_selection`. The `--pelt` flag triggers loading `_changepoints.npy` and calling `apply_pelt_boost`; default path never touches changepoints.
- **Alignment invariant `start ≤ peak_time ≤ end`:** Asserted on every clip post-build, post-merge, post-budget. JSON fixture round-trip verified — all 4 final clips satisfy it (e.g. `588.823 ≤ 594.494 ≤ 600.165`).

## Final Clips JSON Fixture (Phase 4 Input)

`output/cache/justin_timberlake_final_clips.json` (548 bytes, indent=2, gitignored):

```json
[
  {"start_sec": 339.34061..., "end_sec": 350.68205..., "score": 6.554249..., "peak_time": 345.01133...},
  {"start_sec": 588.82318..., "end_sec": 600.16462..., "score": 9.489168..., "peak_time": 594.4939},
  {"start_sec": 749.59382..., "end_sec": 753.37430..., "score": 5.805292..., "peak_time": 751.48406...},
  {"start_sec": 900.83488..., "end_sec": 912.17632..., "score": 10.0,      "peak_time": 906.5056}
]
```

Total = 11.341 + 11.341 + 3.780 + 11.341 = 37.803 ≈ 37.805 (rounding artifact in the print; numerical sum from JSON gives 37.805s within float epsilon). Budget: 37.805s. **100.0% of budget consumed.**

## Deviations from Plan

### Plan-as-written friction (resolved inline)

**1. [Rule 1 - Bug] Module docstring tripped its own grep-ban**
- **Found during:** Task 1 acceptance script first run
- **Issue:** Plan template's `<action>` block included docstring text describing the grep ban using the exact forbidden substrings (`Inline timestamps[idx + 1]` and `score[i] ⟺ timestamps[i+1]`). The acceptance script's `assert forbidden not in src` fired on the docstring itself.
- **Fix:** Rephrased to use `k+1` index variable instead of `idx`/`i` in the docstring text — the ban is described accurately without containing the banned literal.
- **Files modified:** `clip_selection.py` (module docstring lines 27, 31)
- **Commit:** `ce893b0` (folded into Task 1 commit before any external acceptance run)
- **Pattern note:** Future plans wanting to grep-ban literal substrings should use a sentinel pattern that doesn't appear in normal English description (e.g. forbid `_FORBIDDEN_PATTERN_` or use `# noqa: ban-check` markers). Recorded in STATE.md decisions.

### Decisions Honored Without Modification

D-38 through D-51 — all locked decisions implemented verbatim from the CONTEXT, no negotiations or amendments needed. The 4-tuple thread, Pitfall 18 centering, Pitfall 19 hard assert, and Pitfall 20 inline-units were all uncontroversial in implementation.

## Authentication Gates

None — pure-Python module on local fixtures.

## Known Stubs

None — every function is fully implemented and exercised by either the inline acceptance script (Task 1) or the §0.5 harness (Task 2) plus the synthetic embedded test.

## Open Issues for Phase 4

1. **JSON schema confirmation:** `output/cache/justin_timberlake_final_clips.json` is the dev-time contract Phase 4 will consume. The keys are `{start_sec, end_sec, score, peak_time}` (NOT spec §8 schema — that's Phase 5's full-clip JSON with `mad_score`, `raw_cosine_delta`, `coincides_with_pelt_changepoint`, etc.). Phase 4 should iterate this list directly: one ffmpeg `-c copy` invocation per clip, then concat-demuxer.
2. **PELT changepoint coincidence (informational):** With `--pelt` active on JT, the `[pelt-boost] re-sorted; top score now 10.0000` line indicates the top peak is unchanged at the [0,10] ceiling — the boost saturates. Phase 5 can probe this for the JSON `coincides_with_pelt_changepoint` field by reusing the same ±5-sample test from `apply_pelt_boost`.
3. **No clips were merged at JT default settings.** All 40 peaks are spaced > 3s apart (their `min_gap_sec=15.0` enforcement means peaks are ≥ 15s apart in score-index space ≈ 7.5s in timestamp space at 2 fps; with padding=5.671s clips are 11.341s wide → 0–11.34s gaps possible but JT's distribution kept all 40 distinct). Phase 5 may want to tune `--merge-gap-sec` higher when running on lower-action videos to consolidate. Document in tuning NOTES.
4. **JT fills budget exactly to 100.0%** with one partial. This validates the partial-on-peak logic is exercised in real workload, not just synthetic tests. Other 4 videos may produce different behavior depending on `--height` distribution.

## Threat Flags

None — Phase 3 is pure-numpy on local fixtures + one ffprobe subprocess via `utils.probe_video_metadata` (already in scope from Phase 1; no new attack surface).

## Self-Check: PASSED

- `clip_selection.py` exists and is the full implementation (not the stub) — FOUND
- `output/cache/justin_timberlake_final_clips.json` exists, is valid JSON list[dict] — FOUND
- Commit `ce893b0` (Task 1: 7 core functions) — FOUND in `git log`
- Commit `7a05624` (Task 2: §0.5 harness) — FOUND in `git log`
- §0.5 harness terminal print `Phase 3 §0.5 verification: PASS` — FOUND in run log
- All 10 SELP/SELB requirements ticked in REQUIREMENTS.md — VERIFIED via grep
- ROADMAP Phase 3 row → Complete; Plan 03-01 → linked to SUMMARY — VERIFIED
- STATE.md current focus updated to "Phase 3 COMPLETE" with metrics row added — VERIFIED
