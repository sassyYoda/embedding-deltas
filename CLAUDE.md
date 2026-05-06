# Claude Code — Project Guide

## Project

**Body Cam Highlight Reel — AbelPolice Take-Home.** A Python CLI prototype that ingests a raw body cam video and emits a condensed highlight reel (`.mp4`) plus a JSON manifest of selected clip timestamps and scores. Clip selection is driven entirely by visual CLIP embedding deltas — no transcript, audio, OCR, or LLMs.

## Source of Truth

The locked design spec is `assignment-details/bodycam_highlight_reel_spec.md`. Treat it as authoritative; do not substitute alternatives unless a hard technical blocker is encountered, in which case flag it explicitly.

## Planning Artifacts

This project uses GSD (Get Shit Done) workflow. Read these before starting any task:

- `.planning/PROJECT.md` — project context, core value, requirements, key decisions
- `.planning/REQUIREMENTS.md` — 51 v1 requirements with REQ-IDs (ENV / EXTR / EMBD / SIGD / SIGS / SIGM / SIGP / SELP / SELB / EXPC / ORCH / JSON / RUN / DOC)
- `.planning/ROADMAP.md` — 6-phase plan with goals, requirement mappings, and success criteria
- `.planning/STATE.md` — current project memory; check this for "what phase am I on?"
- `.planning/research/SUMMARY.md` — synthesized research (also `STACK.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md`)
- `.planning/config.json` — workflow settings (granularity=coarse, parallel=true, model=balanced, all agents enabled)

## Stack

Python 3.11+. Pinned in `requirements.txt`:

- `torch==2.11.0`, `torchvision==0.26.0`, `open_clip_torch==3.3.0`
- `opencv-python==4.13.0.92`, `numpy>=2.1,<2.5`
- `scipy==1.16.2` (avoid 1.15 — `medfilt` regression #22333)
- `ruptures==1.1.10` (lazy-imported inside `detect_changepoints` so non-`--pelt` runs don't load it)
- `tqdm>=4.67,<5`
- System: `ffmpeg>=4.4` (recommend ≥6.0)
- **Drop `ffmpeg-python`** — unmaintained since 2019; spec uses `subprocess.run` directly.

## Module Layout

Spec §1 is locked:

```
pipeline.py              # CLI entry; argparse + run() orchestration only — no algorithm
extract.py               # frame sampling (cv2) + CLIP embedding (open_clip)
signal_processing.py     # cosine deltas → median filter → rolling MAD; opt-in PELT
clip_selection.py        # find_peaks → adaptive padding → merge → budget enforcement
export.py                # ffmpeg cut (-c copy) + concat demuxer + re-encode fallback
utils.py                 # probe_video_metadata (ffprobe), ensure_output_dirs, write_timestamps_json
```

Build order (spec footer): `extract → signal_processing → clip_selection → export → pipeline`. Get one video working end-to-end before touching the others.

## Cross-Cutting Invariants

- **Timestamp alignment:** `delta[i]` and `score[i]` correspond to `timestamps[i+1]`. `signal_processing.py` is index-space only; `clip_selection.py` is the single place where indices convert to seconds. Locked in Phase 2 via a single helper, asserted in Phase 5 via `start_sec ≤ peak_timestamp_sec ≤ end_sec`.
- **`--pelt` orthogonality:** lazy-imported, opt-in, never on the critical path. JSON `coincides_with_pelt_changepoint` is strict three-state: `null` when `--pelt` is off, `true`/`false` when on.
- **Tune once, freeze:** parameters (`--height`, `--min-gap-sec`, `--merge-gap-sec`) are tuned on one representative video in Phase 5 and held fixed across all 5 videos in Phase 6. Per-video retuning violates spec §6/§12.

## What Not To Do

Per spec §12:
- No transcripts, ASR, audio levels, OCR, or LLMs in the selection path
- No manual timestamp selection
- No fixed global threshold (rolling MAD only)
- No median-filtering raw video frames (filter the 1D delta signal only)
- No default `libx264` re-encode (`-c copy` is faster and acceptable)
- No over-polishing — honest analysis beats optimistic framing

## Workflow Commands

GSD-managed flow:
- `/gsd-progress` — check current phase and next action
- `/gsd-discuss-phase N` — gather context for phase N before planning
- `/gsd-plan-phase N` — produce a detailed `PLAN.md` for phase N
- `/gsd-execute-phase N` — execute all plans in phase N
- `/gsd-verify-work` — UAT a built feature
- `/gsd-help` — list all GSD commands

## Testing Protocol

The spec's §0.5 per-module verification is the success criteria. Each module ships only when its §0.5 prints/asserts pass. Phase success criteria in `ROADMAP.md` mirror those checks.

---
*Last updated: 2026-05-06 after auto-mode initialization.*
