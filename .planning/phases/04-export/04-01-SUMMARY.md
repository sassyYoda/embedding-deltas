---
phase: 04-export
plan: 01
subsystem: export
tags: [ffmpeg, subprocess, concat, stream-copy]
requires: ["output/cache/{video}_final_clips.json (Phase 3 D-50 schema)", "videos/{video}.mp4", "utils.probe_video_metadata", "utils.ensure_output_dirs"]
provides: ["export.extract_clip", "export.validate_clips_for_concat", "export.concat_clips", "output/reels/{video}_highlight.mp4", "output/clips/{video}/{NNN}.mp4"]
affects: ["pipeline.run() — Phase 5 calls extract_clip per clip + concat_clips once per video"]
tech_stack_added: [ffmpeg-7.1.1, ffprobe-7.1.1]
patterns: [pure-subprocess-orchestration, validate-then-fallback, smart-default-fixture-resolution-by-mtime, inline-__main__-§0.5-harness]
key_files_created: []
key_files_modified:
  - export.py
key_decisions:
  - "extract_clip uses -ss BEFORE -i with -c copy (D-52 / Pitfall 11) — fast input-seek, ~1s imprecision acceptable per spec §10"
  - "validate_clips_for_concat compares codec_name + width + height + r_frame_rate + time_base + audio presence/codec across clips (D-57 / Pitfall 12)"
  - "concat_clips falls back to libx264 -crf 18 + aac -b:a 192k via -filter_complex when validate returns FAIL (D-55), logs path taken to stdout"
  - "Manifest uses os.path.abspath + -safe 0 + 4-char single-quote escape '\\\'' (D-56 / Pitfall 13)"
  - "§0.5 harness inline in __main__ per D-58 (no pytest); exit codes 0/2/3/4 differentiate success / missing-input / ffmpeg-failure / reel-verification-failure"
  - "Reel-duration drift tolerance hard-coded at < 5.0s (D-59) to absorb keyframe alignment across N clips"
  - "Import surface restricted to stdlib + utils per D-60 — NO numpy/torch/cv2/open_clip/ruptures"
metrics:
  duration_min: ~10
  completed_date: 2026-05-06
  tasks: 2
  files_modified_count: 1
  harness_wall_clock_sec: 2.0
---

# Phase 4 Plan 01: Export Summary

ffmpeg/ffprobe subprocess orchestration over the Phase 3 final-clip fixture: `extract_clip`, `validate_clips_for_concat`, `concat_clips` + a §0.5 verification harness that produces the actual JT highlight reel on disk.

## Outcome

Phase 4 ships `export.py` with three pure-subprocess functions (~270 lines) plus a 130-line `__main__` §0.5 harness. Running `python export.py` against `output/cache/justin_timberlake_final_clips.json` extracted 4 intermediate clips into `output/clips/justin_timberlake/`, validated codec consistency (PASS), concatenated via the demuxer path (lossless, no re-encode), and wrote `output/reels/justin_timberlake_highlight.mp4` (15,490,821 bytes). ffprobe-probed reel duration 38.491s vs sum-of-clip-durations 37.805s → drift 0.686s, well under the 5.0s D-59 tolerance. Wall-clock: 2 seconds (4 ffmpeg invocations + 1 concat + ffprobe checks).

The reel IS the Phase 4 deliverable — a reviewer can `open output/reels/justin_timberlake_highlight.mp4` to evaluate qualitatively.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | extract_clip + validate_clips_for_concat + concat_clips (D-52..D-57) | `a5fb4c2` | export.py |
| 2 | §0.5 verification harness (D-58, D-59); end-to-end JT run produces reel | `9d9a740` | export.py |

## §0.5 Verification — Run Output

```
[auto] using fixture: justin_timberlake
[fixture] loaded 4 clips from output/cache/justin_timberlake_final_clips.json
[clip-1/4] start=339.341 end=350.682 dur=11.341s -> 000.mp4
[clip-2/4] start=588.823 end=600.165 dur=11.341s -> 001.mp4
[clip-3/4] start=749.594 end=753.374 dur=3.780s -> 002.mp4
[clip-4/4] start=900.835 end=912.176 dur=11.341s -> 003.mp4
[extract] 4/4 clip files: [('000.mp4', 4121087), ('001.mp4', 4398491), ('002.mp4', 2340633), ('003.mp4', 4668451)]
[validate] codec_consistency=PASS reason=''
[concat] PATH=demuxer (reason: validate_clips_for_concat OK)
[concat] reel=output/reels/justin_timberlake_highlight.mp4 size=15490821 bytes
[reel] duration=38.491s expected~=37.805s diff=0.686s tolerance=5.000s
[manual-check] play output/reels/justin_timberlake_highlight.mp4 to confirm coherent concatenation (spec §0.5)
[manual-check] play output/clips/justin_timberlake/000.mp4 to confirm first clip is correct segment (spec §0.5)
Phase 4 §0.5 verification: PASS
```

Concat path taken: **demuxer** (lossless `-c copy`). Validate reason: empty string — JT clips share `h264 / 1920x1080 / 30000/1001 / 1/30000` + `aac` audio, perfectly demuxer-compatible.

Drift breakdown: requested clip durations sum to 37.805s; ffmpeg `-ss BEFORE -i` snapped each `-ss` to the nearest preceding keyframe, accumulating ~0.17s of "earlier-than-requested" lead-in per clip → 0.686s total. Within the documented ~1s/clip imprecision budget (spec §10).

## ROADMAP Phase 4 Success Criteria — All PASS

| SC | Requirement | Evidence | Status |
|----|-------------|----------|--------|
| SC1 (EXPC-01) | extract_clip with -ss before -i + post-call size assert | AST check in Task 1 verify block: `-ss` is constant#3, `-i` is constant#5 in the cmd list. 4/4 JT clip files written non-zero (sizes 4.12 / 4.40 / 2.34 / 4.67 MB). | PASS |
| SC2 (EXPC-02) | concat_clips with manifest safety + ffprobe pre-validation + concat-filter fallback | `validate_clips_for_concat` runs first (operator log: `[validate] codec_consistency=PASS reason=''`); demuxer path chosen (`[concat] PATH=demuxer`); manifest written with `os.path.abspath` + `'\''` escape + `-safe 0` flag (Pitfall 13); concat-filter fallback path with `libx264 -crf 18` exists in source (grep-verified) for the FAIL branch. | PASS |
| SC3 (EXPC-03) | §0.5 verification — 4 non-zero intermediate clips + coherent reel | 4 files at `output/clips/justin_timberlake/{000..003}.mp4` (sizes above), reel at `output/reels/justin_timberlake_highlight.mp4` = 15,490,821 bytes, ffprobe-probed reel duration 38.491s vs fixture sum 37.805s = drift 0.686s < 5.0s. | PASS |

## Pitfall Neutralization

| Pitfall | Defense | Grep-Verifiable Location |
|---------|---------|--------------------------|
| 11 (`-ss` placement) | `-ss` BEFORE `-i` in `extract_clip` cmd list, with inline `# Pitfall 11` comment on the line | `export.py extract_clip`: `"-ss", str(start_sec), "-to", str(end_sec), "-i", in_str, "-c", "copy"` |
| 12 (concat demuxer fragility) | `validate_clips_for_concat` runs ffprobe pre-flight on codec_name/width/height/r_frame_rate/time_base + audio presence/codec across all clips; on mismatch, `concat_clips` logs `[concat] FALLBACK: re-encoding via concat filter (reason: ...)` and uses `-filter_complex` + `libx264 -crf 18 / aac 192k` re-encode | `export.py validate_clips_for_concat` (3 ffprobe calls per clip in two probe shapes) and `export.py concat_clips` fallback branch |
| 13 (manifest path safety) | `os.path.abspath(os.fspath(path))` + `escaped = abs_path.replace("'", r"'\''")` + `"-safe", "0"` flag on demuxer cmd, with inline `# Pitfall 13` comment | `export.py concat_clips` demuxer branch |

All three Pitfalls have a grep-verifiable runtime guard, NOT just a comment.

## Deviations from Plan

None of substance. The plan was executed verbatim against CONTEXT D-52..D-63.

One acceptance-check refinement (not a behavior deviation):
- **Plan Task 1 acceptance** included a Pitfall-11 AST check that searched `ast.unparse()` output for the literal string `"-ss"` (double-quoted). Python's `ast.unparse` normalizes string-constant rendering to single quotes, so the literal-substring `s.find('"-ss"')` returned -1 and the assertion fired. The implementation is correct (`-ss` IS the 4th constant in the cmd list, before `-i` as the 6th); only the verification needle's quote style was wrong. Replaced with a quote-agnostic AST walk that collects `ast.Constant` strings in source order and asserts `strings.index('-ss') < strings.index('-i')`. Confirmed: `-ss` at constant#3, `-i` at constant#5. The plan-checker's contingency in `<execution_rules>` Rule 9 anticipated this exact fragility.

## Authentication Gates

None — Phase 4 invokes only local ffmpeg/ffprobe binaries. No network, no auth, no secrets.

## Notes for Phase 5

- **`concat_clips` interface signature `(clip_paths, output_path, temp_dir)` is what `pipeline.run()` will call once per video** (D-63). The `__main__` harness in `export.py` is dev-time only and is NOT invoked by `pipeline.py`. Phase 5 calls `extract_clip` per clip in a loop and `concat_clips` once at the end.
- **`temp_dir` for the manifest:** Phase 4's harness passes the per-video `clips/{video}/` directory (which contains the intermediate clips); `concat_manifest.txt` is written there. Phase 5 should do the same, OR pass a `tempfile.TemporaryDirectory()` if it wants automatic cleanup. Per D-61/D-62, Phase 4 does NOT auto-delete intermediate clips — Phase 5 inherits that behavior.
- **`extract_clip` accepts both `str` and `pathlib.Path`** via `os.fspath(...)` (Claude's Discretion in CONTEXT). Phase 5 can pass either.
- **`validate_clips_for_concat` returns `(True, "")` for empty input** (caller handles empty case). Phase 5's `concat_clips` already guards against empty list with `ValueError` — pipeline should fail-fast before calling concat with zero clips.
- **Reel-duration drift on JT was 0.686s** (well under the 5.0s tolerance). For longer videos with more clips and sparser keyframes, drift may approach but should not exceed 5.0s. If it does, the pipeline should surface the `[FATAL] reel duration drift...` line and exit 4 — Phase 5 can either propagate the exit code or catch and log.
- **`open_clip`/`numpy`/`torch` are NOT imported by `export.py`** (D-60). Phase 5's `pipeline.py` will import all of them, but importing `export` itself remains cheap (stdlib only).
- **The `[concat] FALLBACK:` path was not exercised on JT** because all clips passed validation. Phase 6 (multi-video) may exercise it if any of the 4 remaining videos has codec/timebase boundaries; the fallback is grep-verified to exist (`libx264`, `filter_complex`, `FALLBACK` substring) and structurally correct per D-55.

## Self-Check: PASSED

- `export.py` exists (modified from 1-line stub to ~410 lines)
- `output/clips/justin_timberlake/000.mp4` exists, 4,121,087 bytes
- `output/clips/justin_timberlake/001.mp4` exists, 4,398,491 bytes
- `output/clips/justin_timberlake/002.mp4` exists, 2,340,633 bytes
- `output/clips/justin_timberlake/003.mp4` exists, 4,668,451 bytes
- `output/reels/justin_timberlake_highlight.mp4` exists, 15,490,821 bytes
- Commit `a5fb4c2` exists in `git log` (Task 1)
- Commit `9d9a740` exists in `git log` (Task 2)
- Phase 4 §0.5 verification: PASS (reproduced above)
