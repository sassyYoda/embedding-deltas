# State: Body Cam Highlight Reel — AbelPolice Take-Home

## Project Reference

**Core Value:** Given a body cam video, produce a highlight reel where selected moments visibly correspond to high-action or significant scene changes — using only visual embedding signal — reproducibly across all five sample videos with one fixed parameter set.

**Current Focus:** Phase 1 — Frame Extraction & Embeddings. Highest pitfall density (8 of 20 cataloged); must lock the canonical `(timestamps, embeddings)` fixture before any downstream work begins.

**Locked spec:** `assignment-details/bodycam_highlight_reel_spec.md` is the single source of truth. Every implementation decision traces to a numbered section there.

---

## Current Position

- **Milestone:** v1 (initial submission)
- **Phase:** 1 — Frame Extraction & Embeddings
- **Plan:** TBD (Phase 1 not yet planned)
- **Status:** Roadmap created; awaiting `/gsd-plan-phase 1`
- **Progress:** 0/6 phases complete `[░░░░░░] 0%`

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases planned | 0 / 6 |
| Phases complete | 0 / 6 |
| Requirements mapped | 51 / 51 ✓ |
| Plans executed | 0 |
| Sample videos processed end-to-end | 0 / 5 |

---

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

- [ ] Phase 1: probe sample videos with `ffprobe -show_streams` to confirm CFR vs VFR before locking sampling code path (research flag).
- [ ] Phase 4: one-spike concat-demuxer test on a real extracted clip from sample video 1 to confirm fallback path expectations (research flag).
- [ ] Phase 5: choose representative video for parameter tuning by watching all 5 raw videos and picking the one with broadest dynamic range.
- [ ] Phase 6: README must lean into qualitative per-video writeup; resist inventing quantitative numbers.

### Blockers

None — roadmap is approved-shape; ready to plan Phase 1.

---

## Session Continuity

**Last session (2026-05-06):**
- Initialized PROJECT.md, REQUIREMENTS.md (51 v1 reqs), and full research bundle (SUMMARY, ARCHITECTURE, PITFALLS).
- Created ROADMAP.md with 6 coarse-grained phases mapping all 51 requirements; locked cross-cutting invariants.
- Created STATE.md (this file).

**Next action:** `/gsd-plan-phase 1` to decompose Phase 1 (Frame Extraction & Embeddings) into executable plans against the spec §0.5 verification gates and the 8 frame/embedding pitfalls.

**Recent files touched:**
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/research/SUMMARY.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- `.planning/ROADMAP.md` (this session)
- `.planning/STATE.md` (this session)

---

*Last updated: 2026-05-06 after roadmap creation*
