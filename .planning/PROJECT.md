# Body Cam Highlight Reel — AbelPolice Take-Home

## What This Is

An end-to-end Python pipeline that ingests a raw body cam video and emits a condensed highlight reel (`.mp4`) plus a JSON manifest of selected clip timestamps and scores. Clip selection is driven entirely by visual embedding deltas — no transcript, audio, OCR, or LLMs. Built as a take-home prototype for AbelPolice to demonstrate a defensible, fully-algorithmic approach to extracting "interesting moments" from continuous body cam footage.

## Core Value

Given a body cam video, the pipeline must produce a highlight reel where the selected moments visibly correspond to high-action or significant scene changes — using only visual embedding signal — and must do so reproducibly across the 4 in-scope sample videos (justin_timberlake, tiger_woods, test_assault_theft, test_missing_person) with the same fixed parameters. The 5th video (marcus_jordan, 75 min) was dropped after a reproducible MPS hang during CLIP extraction; documented as a known limitation per spec §11.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Sample frames at 2 fps from arbitrary-length body cam `.mp4` input
- [ ] Extract L2-normalized embeddings via CLIP ViT-L/14 (OpenAI / QuickGELU pretrained), batch size 32, `torch.no_grad()`
- [ ] Compute cosine-distance deltas between consecutive frame embeddings (range 0–2, expected 0.0–0.5)
- [ ] Apply 5-sample median filter to raw delta signal (impulse-noise robustness)
- [ ] Apply rolling MAD normalization with 90-second (180-sample) centered window; clip output to [0, 10]
- [ ] Detect peaks via `scipy.signal.find_peaks` with `--height` (default 1.5 MAD) and `--min-gap-sec` (default 15s) thresholds
- [ ] Optional `--pelt` flag: PELT changepoint detection as supplementary signal; boost MAD peaks within ±5 samples of a changepoint by 1.2×
- [ ] Compute duration budget = `(video_duration / 1800) × 60` seconds (1 min per 30 min)
- [ ] Adaptive padding: `max(3.0, min(8.0, budget × 0.15))` seconds per side
- [ ] Build clips → merge clips within `--merge-gap-sec` (default 3s) → enforce budget greedily by score
- [ ] Export per-clip via `ffmpeg -c copy` (stream copy); concatenate into final reel via concat demuxer
- [ ] Emit JSON timestamp file at `output/timestamps/{video_name}.json` with the schema in spec §8
- [ ] Single CLI entry point `pipeline.py <video>` with flags `--pelt`, `--height`, `--min-gap-sec`, `--merge-gap-sec`
- [ ] Run the 4 in-scope sample videos (justin_timberlake, tiger_woods, test_assault_theft, test_missing_person) with one fixed parameter set (no per-video retuning) — marcus_jordan dropped due to MPS hang
- [ ] Per-module verification matching spec §0.5 testing protocol

### Out of Scope

- Transcript / ASR / audio-level signals — assignment constraint (must be visual-only)
- OCR or LLM-based clip selection — assignment constraint
- Manual timestamp selection — algorithmic-only requirement
- Fine-tuning the CLIP model — spec §2 explicitly forbids
- GPU-specific code paths — let `open_clip` auto-select device
- Per-video parameter retuning — would amount to manual selection
- Quantitative precision/recall evaluation — no labeled ground truth available
- Re-encoding clips by default — `-c copy` is faster and lossless; only fall back to `libx264` if stream copy produces broken output
- Polished UI / web frontend — CLI prototype only

## Context

- **Source spec:** `assignment-details/bodycam_highlight_reel_spec.md` is the locked design document; every implementation decision and rationale is captured there. Treat it as the single source of truth — do not substitute alternatives unless a hard technical blocker is hit.
- **Sample videos:** 5 raw body cam `.mp4` files downloaded from a Drive folder into `videos/` (gitignored). 4 are in v1 scope (justin_timberlake 18.9 min, tiger_woods 27.5 min, test_assault_theft 3.1 min, test_missing_person 2.9 min); marcus_jordan (75.2 min, 9,025 frames) was dropped after reproducible MPS deadlocks at ~50% inference — documented as a known limitation per spec §11.
- **Assignment context:** AbelPolice take-home — "Visual Embedding Highlight Reels". The reviewer cares most about defensibility of the design choices and the ability to reproduce results across the in-scope videos with one parameter set, plus an honest writeup of any limitations encountered (per spec §11 — "honest analysis beats optimistic framing").
- **Prior art referenced in spec:** Koddenbrock et al. (2025) on CLIP robustness in handheld-camera domains motivates the ViT-L/14 OpenAI choice; Antonio's k-most-distinct approach motivates the rolling-MAD improvement; Huang/Tukey on median-filter edge preservation justifies filter choice.
- **Build order:** `extract.py → signal_processing.py → clip_selection.py → export.py → pipeline.py`. Get one video working end-to-end before the other four (per spec footer).
- **Hardware:** CPU-only inference is acceptable; GPU is auto-selected by `open_clip` if present.

## Constraints

- **Tech stack**: Python 3.11+, `torch`, `torchvision`, `open_clip_torch`, `opencv-python`, `numpy`, `scipy`, `ruptures`, `ffmpeg-python`, `tqdm` — locked by spec §1
- **System dependency**: `ffmpeg` must be installed at the OS level (not just the Python binding)
- **Embedding model**: CLIP ViT-L/14 OpenAI pretrained — non-negotiable per spec §2
- **Algorithmic-only selection**: no transcript, audio, OCR, LLM, or manual timestamps in the selection path — assignment requirement
- **Reproducibility**: all 4 in-scope videos must be processed with one fixed parameter set (marcus_jordan dropped — see Context above)
- **Project structure**: `pipeline.py / extract.py / signal_processing.py / clip_selection.py / export.py / utils.py` plus `output/{reels,clips,timestamps}/` — locked by spec §1
- **Timeline**: take-home assignment — prototype quality over polish; honest analysis of limitations beats optimistic framing

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CLIP ViT-L/14 OpenAI (QuickGELU) | Most robust to MotionBlur/PerspectiveTransform in Handheld domain (Koddenbrock 2025) | — Pending |
| 2 fps frame sampling | Captures sub-second events without 30 fps redundancy; ~3,600 frames for 30 min video is feasible CPU-side | — Pending |
| Cosine distance for deltas | Semantic similarity, robust to pixel-level noise | — Pending |
| Median filter (kernel=5) before normalization | Edge-preserving for impulse-type body-cam artifacts; filtering before MAD prevents spike contamination | — Pending |
| Rolling MAD with 90s window | Robust to outliers (vs. std dev); 90s avoids both local-baseline elevation and global drift contamination | — Pending |
| PELT as opt-in supplementary signal (--pelt) | Cross-references MAD peaks with structural changepoints when desired; not required for baseline | — Pending |
| Greedy budget enforcement post-merge | Merging reduces total duration, so it must precede the budget check | — Pending |
| ffmpeg `-c copy` for clip export | Fast and lossless; ~1s keyframe-alignment imprecision is acceptable for a prototype | — Pending |
| Adaptive padding (15% of budget, [3, 8]s) | Prevents short videos from having clips longer than the entire budget | — Pending |
| Tune parameters once on the most representative video, then freeze | Per-video retuning would violate the algorithmic-only requirement | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-06 after initialization (auto mode, sourced from `assignment-details/bodycam_highlight_reel_spec.md`)*
