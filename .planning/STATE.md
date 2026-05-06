---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Phase 1 complete (extract.py + utils.py + §0.5 verification fixture); ready for `/gsd-plan-phase 2`
last_updated: "2026-05-06T20:02:30.161Z"
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 100
---

# State: Body Cam Highlight Reel — AbelPolice Take-Home

## Project Reference

**Core Value:** Given a body cam video, produce a highlight reel where selected moments visibly correspond to high-action or significant scene changes — using only visual embedding signal — reproducibly across all five sample videos with one fixed parameter set.

**Current Focus:** Phase 1 — Frame Extraction & Embeddings. Highest pitfall density (8 of 20 cataloged); must lock the canonical `(timestamps, embeddings)` fixture before any downstream work begins.

**Locked spec:** `assignment-details/bodycam_highlight_reel_spec.md` is the single source of truth. Every implementation decision traces to a numbered section there.

---

## Current Position

- **Milestone:** v1 (initial submission)
- **Phase:** 1 — Frame Extraction & Embeddings (COMPLETE)
- **Next phase:** 2 — Signal Processing
- **Status:** Phase 1 complete; canonical (timestamps, embeddings) fixture written to output/cache/. Ready for `/gsd-plan-phase 2`.
- **Progress:** 1/6 phases complete `[█▓░░░░░░░░] ~17%`

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases planned | 1 / 6 |
| Phases complete | 1 / 6 |
| Requirements mapped | 51 / 51 ✓ |
| Requirements complete | 17 / 51 (ENV-01..04, EXTR-01..05, EMBD-01..06) |
| Plans executed | 2 (01-01 + 01-02) |
| Sample videos processed end-to-end | 0 / 5 |
| Phase 01 §0.5 harness wall-clock (mps, JT 19-min video) | 615.25 s real / 542.68 s user |

### Plan Execution

| Plan | Duration (min) | Tasks | Files |
|------|----------------|-------|-------|
| Phase 01 P01 | ~5 | 2 | 7 |
| Phase 01 P02 | ~12 | 2 | 1 |

## Accumulated Context

### Cross-Cutting Invariants (must hold across all phases)

1. **Timestamp alignment:** `scores[i] ⟺ timestamps[i+1]`. Locked in Phase 2 via single helper; consumed in Phase 3; verified in Phase 5 JSON.
2. **`--pelt` orthogonality:** lazy-imported, opt-in, never on critical path. JSON field strict three-state (`null`/`true`/`false`).
3. **Index-space vs time-space separation:** `signal_processing.py` is index-only; `clip_selection.py` is the single index→seconds conversion site.
4. **Tune once, freeze:** parameters tuned on representative video in Phase 5, frozen for Phase 6.
5. **§0.5 prints are the success criteria** — qualitative module-by-module gates, not separate tests to invent.

### Key Decisions (from PROJECT.md / spec / research)

- CLIP ViT-L/14 OpenAI (QuickGELU) — non-negotiable per spec §2; chosen for MotionBlur/PerspectiveTransform robustness (Koddenbrock 2025).
- 2 fps frame sampling — captures sub-second events without 30 fps redundancy.
- Median filter (kernel=5) **before** MAD normalization — filtering before normalization prevents spike contamination.
- Rolling MAD with 90s (180-sample) centered window — robust to outliers, captures local context.
- ffmpeg `-c copy` for stream-copy export — fast, lossless, ~1s keyframe imprecision acceptable.
- Adaptive padding `max(3, min(8, budget × 0.15))` — prevents short videos having clips longer than budget.
- Drop `ffmpeg-python` from requirements (unmaintained since 2019; spec uses raw `subprocess.run` anyway — research correction).
- Pin `scipy==1.16.2` (avoid 1.15 `medfilt` regression — research correction).
- Lazy-import `ruptures` inside `detect_changepoints()` (research correction).

### Open Todos / Watch Items

- [x] Phase 1: probe sample videos with `ffprobe -show_streams` to confirm CFR vs VFR — done by `utils.probe_video_metadata`; `justin_timberlake.mp4` is_vfr=False; sampling code path is VFR-safe regardless (D-06).
- [ ] **Phase 1 follow-up (DEVIATION REQUIRED):** open_clip 3.3.0 emits `QuickGELU mismatch` UserWarning when calling `create_model_and_transforms('ViT-L-14', pretrained='openai')` per D-09 verbatim. Spec §2 mandates the QuickGELU variant. CONTEXT amendment territory: pick option 1 (rename to `'ViT-L-14-quickgelu'`), option 2 (keep D-09 as-is, accept warning), or option 3 (`force_quick_gelu=True`). Re-emit fixtures if the model changes. See 01-02-SUMMARY.md "Open Issue".
- [ ] Phase 4: one-spike concat-demuxer test on a real extracted clip from sample video 1 to confirm fallback path expectations (research flag).
- [ ] Phase 5: choose representative video for parameter tuning by watching all 5 raw videos and picking the one with broadest dynamic range.
- [ ] Phase 6: README must lean into qualitative per-video writeup; resist inventing quantitative numbers.

### Decisions (added during execution)

- **Plan 01-02:** Honored D-09 verbatim ('ViT-L-14' + pretrained='openai'); flagged QuickGELU mismatch UserWarning as a CONTEXT amendment for the user before fixtures propagate to Phase 2.
- **Plan 01-02:** MPS bit-identical rerun (D-17) PASSED on first call without explicit warm-up — RESEARCH A5's documented warm-up batch fallback is NOT needed for Phase 1.

### Blockers

None — Phase 1 §0.5 verification PASSED end-to-end on `videos/justin_timberlake.mp4`; canonical fixture is on disk under `output/cache/`. The QuickGELU question is a flagged open issue but does not block Phase 2 planning (Phase 2 develops against the fixture; if the user picks option 1 or 3, the fixture is regenerated and Phase 2 simply reloads it).

---

## Session Continuity

**Last session (2026-05-06):**

- Executed Plan 01-02 (extract.py + §0.5 harness) end-to-end. Two atomic commits: `d99fd32` (Task 1: sample_frames + load_model + embed_frames) and `40a1e28` (Task 2: __main__ harness with ffmpeg precondition + bit-identical rerun + fixture writer).
- Ran §0.5 verification on `videos/justin_timberlake.mp4` (1134.144s, h264, 1280x720, CFR): 2269 frames sampled, embeddings (2269, 768) float32 L2-normalized, drift 0.144s, [determinism] PASS on MPS without warm-up. Wall-clock 615s.
- Wrote fixtures `output/cache/justin_timberlake_{embeddings,timestamps}.npy` for Phase 2 parallel development.
- Marked complete: ENV-02, EXTR-01..05, EMBD-01..06 (12 reqs).
- Flagged QuickGELU mismatch UserWarning from open_clip 3.3.0 as CONTEXT-amendment territory before Phase 2.

**Next action:** `/gsd-plan-phase 2` to decompose Phase 2 (Signal Processing) — `signal_processing.py` deltas/medfilt/MAD/PELT against the Phase 1 fixture. **Before planning, resolve the QuickGELU open issue** so Phase 2 doesn't lock in stale embeddings.

**Recent files touched:**

- `extract.py` (created Task 1; appended Task 2)
- `output/cache/justin_timberlake_embeddings.npy` (new)
- `output/cache/justin_timberlake_timestamps.npy` (new)
- `.planning/phases/01-frame-extraction-embeddings/01-02-SUMMARY.md` (new)
- `.planning/STATE.md` (this update)
- `.planning/REQUIREMENTS.md` (12 checkboxes ticked)

---

*Last updated: 2026-05-06 after Plan 01-02 execution*
