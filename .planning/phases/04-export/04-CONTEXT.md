# Phase 4: Export - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning
**Mode:** `--auto` (recommended option auto-selected per gray area; choices logged in `04-DISCUSSION-LOG.md`)

<domain>
## Phase Boundary

Cut individual clips losslessly with `ffmpeg -c copy` and concatenate them into a single highlight reel `.mp4`, with the concat-demuxer's known fragility neutralized by ffprobe-based pre-validation and a re-encode fallback. **All `subprocess` calls in the project live in this module.** The rest of the pipeline stays pure-Python.

Specifically Phase 4 delivers (in `export.py`):
- `extract_clip(input_path, output_path, start_sec, end_sec) -> None` — invokes `ffmpeg -y -ss <start> -to <end> -i <input> -c copy <output>`. Stream copy, fast, lossless. Sub-second keyframe imprecision is acceptable per spec §10.
- `concat_clips(clip_paths, output_path, temp_dir) -> None` — writes a concat manifest with absolute paths, runs `ffmpeg -y -f concat -safe 0 -i <manifest> -c copy <output>`. Falls back to concat-filter re-encode (`-c:v libx264 -crf 18`) if the demuxer rejects the inputs (Pitfall 12).
- `validate_clips_for_concat(clip_paths) -> tuple[bool, str]` — pre-flight `ffprobe` check that all clips share codec + timebase. Returns `(True, "")` if compatible, `(False, reason)` otherwise. Caller decides whether to use demuxer or fallback.
- `__main__` block — §0.5 verification harness against the JT `final_clips.json` fixture; produces `output/reels/justin_timberlake_highlight.mp4` as the actual reel artifact.

**Not in this phase:** clip selection (Phase 3, already complete); JSON assembly (Phase 5); CLI argparse plumbing for `pipeline.py` (Phase 5); multi-video batch run (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### `extract_clip` — Per-Clip Cut

- **D-52:** `extract_clip(input_path, output_path, start_sec, end_sec)` invokes:
  ```python
  cmd = [
      "ffmpeg", "-y",
      "-ss", str(start_sec),
      "-to", str(end_sec),
      "-i", input_path,
      "-c", "copy",
      output_path,
  ]
  subprocess.run(cmd, check=True, capture_output=True)
  ```
  Note: **`-ss` BEFORE `-i`** is correct for stream copy (spec §7; Pitfall 11). It's the FAST seek path: ffmpeg seeks to nearest preceding keyframe before decoding starts. Imprecision: clip can start up to ~1s earlier than requested. Acceptable per spec §10. (Reference EXPC-01.)
- **D-53:** Post-call sanity: `assert os.path.getsize(output_path) > 0` — guard against ffmpeg producing a zero-byte file (which can happen on some edge cases without raising). If the assertion fires, raise `RuntimeError(f"clip extraction produced empty file: {output_path}")` with stderr from the subprocess for debugging.
- **D-54:** On `subprocess.CalledProcessError`: re-raise with the captured stderr decoded as UTF-8 in the message (`subprocess.run(..., capture_output=True)` captures both; `e.stderr.decode('utf-8', errors='replace')`). Don't swallow ffmpeg errors silently.

### `concat_clips` — Concatenation with Fallback

- **D-55:** `concat_clips(clip_paths, output_path, temp_dir)` first runs `validate_clips_for_concat(clip_paths)`:
  - If compatible → use **concat demuxer** (lossless, fast):
    ```python
    manifest_path = os.path.join(temp_dir, "concat_manifest.txt")
    with open(manifest_path, "w") as f:
        for path in clip_paths:
            abs_path = os.path.abspath(path)
            # Sanitize: ffmpeg concat manifest uses single-quoted paths.
            # Replace any literal single-quote in the filename with '\''.
            escaped = abs_path.replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", manifest_path,
        "-c", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    ```
  - If incompatible → fallback to **concat filter** (re-encodes; slower but reliable):
    ```python
    inputs = []
    for path in clip_paths:
        inputs.extend(["-i", os.path.abspath(path)])
    n = len(clip_paths)
    filter_str = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n)) + f"concat=n={n}:v=1:a=1[outv][outa]"
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    ```
    Log `[concat] FALLBACK: re-encoding via concat filter (reason: ...)` so the deviation is visible in stdout. (Reference EXPC-02; Pitfall 12.)
- **D-56:** Manifest path safety (Pitfall 13):
  - Always use `os.path.abspath(path)` for clip paths in the manifest.
  - Always pass `-safe 0` to ffmpeg (allows absolute paths).
  - Escape single-quotes in filenames (`'` → `'\''`) to handle pathological filenames.
  - The temp_dir for the manifest is `output/clips/{video_name}/` (already created by Phase 1's `utils.ensure_output_dirs`); pass it in from the caller.

### `validate_clips_for_concat` — Pre-flight Compatibility Check

- **D-57:** `validate_clips_for_concat(clip_paths) -> tuple[bool, str]`:
  - Probes each clip via `ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,time_base -of json <path>`.
  - Returns `(False, "codec mismatch: clip 0 h264 vs clip 1 hevc")` (or similar) on mismatch.
  - Returns `(False, "ffprobe failed for {path}: <error>")` on any ffprobe error.
  - Returns `(True, "")` if all clips share codec_name + width + height + r_frame_rate + time_base.
  - Audio stream consistency is also checked (codec_name only — different bitrates/sample-rates can survive concat demuxer in practice).
  - **Edge case:** if a clip has no audio stream, that's allowed — but ALL clips must agree (all-with-audio or all-without). Mixed → fail validation.

### `__main__` Verification Harness

- **D-58:** Pattern matches Phases 1–3: `if __name__ == "__main__":` block, runnable as `python export.py [video_name]`.
  - Default `video_name`: derive from first `*_final_clips.json` in `output/cache/` (most recent), with `_final_clips` stripped.
  - Loads `output/cache/{video}_final_clips.json` (Phase 3 fixture) — list of `{start_sec, end_sec, score, peak_time}`.
  - Resolves video path: `videos/{video}.mp4`.
  - Calls `utils.ensure_output_dirs(video)` to get `{reels: ..., clips: ...}` paths.
  - Loop: for each clip in JSON, call `extract_clip(video_path, clips_dir/{N:03d}.mp4, start, end)`. Log progress `[clip-N/total] ...` and clip duration.
  - After all clips extracted, call `concat_clips([clip-paths], reels_dir/{video}_highlight.mp4, clips_dir)`.
  - **§0.5 prints:**
    - `[extract] N/M clip files: <list of basenames + sizes>`
    - `[validate] codec_consistency=PASS|FAIL <reason>`
    - `[concat] PATH=<demuxer|filter> reel=<path> size=<bytes>`
    - `[reel] duration=<probed_duration>s expected≈<sum_of_clip_durations>s diff=<...>`
    - `Phase 4 §0.5 verification: PASS`
  - Exit codes: 0 success; 2 missing fixture; 3 ffmpeg failure; 4 reel verification failure.

### Sub-Second Cut Precision

- **D-59:** Phase 3 produces clip boundaries at sub-second precision (e.g., `start=339.341, end=350.682`). The `-ss` BEFORE `-i` stream-copy seek lands on the nearest **preceding keyframe**, so actual clip start may be up to ~1s earlier than requested. **This is acceptable per spec §10** ("~1s boundary imprecision acceptable for prototype"). The §0.5 reel duration check uses `< 5.0s` tolerance to account for keyframe-alignment drift across N clips. If the user later wants frame-precise cuts, they can switch to `-ss` AFTER `-i` + re-encode (`-c:v libx264 -crf 18`) — that's a v2 concern, not v1.

### `ruptures` / `torch` / `cv2` Are NOT Imported Here

- **D-60:** `export.py` imports ONLY: stdlib (`os`, `subprocess`, `json`, `pathlib`, `argparse`, `sys`, `logging`), `utils` (for `probe_video_metadata` + `ensure_output_dirs`). No `numpy`, no `torch`, no `cv2`, no `open_clip`, no `ruptures`. This module is purely subprocess-orchestration over ffmpeg/ffprobe. Static check in §0.5 acceptance: `assert 'import torch' not in src and 'import cv2' not in src and 'import ruptures' not in src`.

### Cleanup Behavior

- **D-61:** `concat_clips` does **NOT** auto-delete intermediate per-clip `.mp4` files in `output/clips/{video}/`. They stay on disk for spec §0.5 verification ("verify each intermediate clip file exists and is non-zero bytes" — needed at any time, not just immediately post-concat). A future `--cleanup` flag is v2.
- **D-62:** On error during `concat_clips`, intermediate clips are NOT deleted (debugging value). The user can rm them by hand if disk pressure is real.

### Multi-Video Awareness

- **D-63:** Phase 4 operates on **one video at a time**. The §0.5 harness uses the JT fixture. Phase 6 (multi-video batch) will call `extract_clip` + `concat_clips` per video via `pipeline.run()`.

### Claude's Discretion

- Type-hint style, docstring style (consistent with Phases 1–3).
- Whether `validate_clips_for_concat` returns a `(bool, str)` tuple or a `dataclass` (use tuple for v1 — matches spec §6 simplicity).
- ffmpeg's `-loglevel error` suppression of progress lines on the subprocess level vs letting them flow (we already `capture_output=True` so they're hidden — choose to NOT add `-loglevel`; on failure, the captured output has full debug info).
- Whether to support `pathlib.Path` or only `str` for `input_path` / `output_path` (support both via `os.fspath(...)` wrapping at call sites in `extract_clip` / `concat_clips`).
- Tolerance: `< 5.0s` for reel-duration drift assertion in §0.5 (loose enough to absorb keyframe alignment across multiple clips).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Locked Spec
- `assignment-details/bodycam_highlight_reel_spec.md` §7 (Export — `extract_clip`, `concat_clips`), §0.5 (testing protocol), §10 row 12 (`-c copy` rationale + ~1s imprecision acceptable), §12 (no default re-encode).

### Inherited from Prior Phases
- `.planning/phases/03-clip-selection/03-CONTEXT.md` D-50 (`output/cache/{video}_final_clips.json` shape: list of `{start_sec, end_sec, score, peak_time}` dicts — Phase 4's input).
- `.planning/phases/01-frame-extraction-embeddings/01-CONTEXT.md` D-20 (`utils.ensure_output_dirs` returns `{reels, clips, timestamps, cache}` paths — Phase 4 uses `clips` and `reels`).

### Project-Level
- `.planning/PROJECT.md` — Out of Scope: no `libx264` re-encode by default (spec §12).
- `.planning/REQUIREMENTS.md` — REQ-IDs EXPC-01, EXPC-02, EXPC-03 (3 mapped to Phase 4).
- `.planning/ROADMAP.md` Phase 4 detail — 3 success criteria.
- `.planning/STATE.md` — current memory.

### Research
- `.planning/research/SUMMARY.md` — Phase 4 owns Pitfalls 11, 12, 13.
- `.planning/research/PITFALLS.md` §11 (`-ss` placement before/after `-i`), §12 (concat demuxer fragility — codec/timebase mismatch), §13 (manifest path safety — single-quote sanitization, `-safe 0`).
- `.planning/research/ARCHITECTURE.md` — Phase 4 owns ALL subprocess calls in the project.

### External
- `https://ffmpeg.org/ffmpeg-formats.html#concat` — concat demuxer requirements.
- `https://trac.ffmpeg.org/wiki/Concatenate` — fallback via concat filter when codecs differ.
- `https://ffmpeg.org/ffmpeg.html#Stream-copy` — `-c copy` and the `-ss` placement semantics.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `output/cache/justin_timberlake_final_clips.json` — Phase 4 dev input. List of 4 clip dicts (3 full + 1 partial), total 37.805s.
- `videos/justin_timberlake.mp4` — original 19-min video, ffmpeg input.
- `utils.ensure_output_dirs(video_name)` — already creates `output/reels/`, `output/clips/{video}/`, etc.
- `utils.probe_video_metadata(path)` — already shipped; Phase 4's reel-duration check uses this.
- `export.py` is a stub (Plan 01-01). Phase 4 replaces it.

### Established Patterns

- **Module `__main__` verification harness** (D-58) — same pattern as Phases 1, 2, 3.
- **Output artifact write side-effect** — Phase 4 produces a real reel `.mp4` in `output/reels/`, not a `.npy` fixture in `output/cache/`. The reel IS the product.
- **Single-source-of-truth dependency on `utils`** — `export.py` calls `utils.ensure_output_dirs` and `utils.probe_video_metadata` (the latter for the §0.5 reel-duration check).
- **Conventional commits** — `feat(04-NN-MM): subject`.

### Integration Points

- **Phase 3 → Phase 4:** `final_clips.json` (4-tuple list of dicts) consumed by Phase 4's `__main__` harness.
- **Phase 4 → Phase 5:** `pipeline.run()` will call `extract_clip` per clip and `concat_clips` once per pipeline invocation. The `__main__` harness in Phase 4 is dev-time only; Phase 5's `pipeline.py` calls these directly without re-reading the fixture.
- **Phase 4 → reviewer/end-user:** the `output/reels/{video}_highlight.mp4` file IS the deliverable. A reviewer plays this to evaluate the project.

</code_context>

<specifics>
## Specific Ideas

- **`-ss` BEFORE `-i` (D-52)** is the locked placement. A reviewer reading `extract.py` will be sensitive to this — the spec explicitly chose stream copy with fast seek, accepting ~1s imprecision. Inline comment must say so.
- **The concat demuxer fallback (D-55) is the single most fragile path in Phase 4.** Plan must include a runtime test where we deliberately cause a demuxer failure (e.g., feed it clips with different codecs) and verify the fallback kicks in. On the JT happy path the validator will say PASS and we use the demuxer; the fallback should be runtime-tested via a synthetic check (re-encode one clip first to mismatch the others) — but only IF time permits in `__main__`. As a minimum, log `[concat] PATH=demuxer` so the path taken is visible.
- **The §0.5 print "Watch the reel"** — spec §0.5 says "Play the first clip. Verify it is the correct segment of the video. Play the final reel. Verify it is a coherent concatenation." Phase 4 can't automate the visual check, but it CAN:
  - Print the reel path so the user can `open output/reels/justin_timberlake_highlight.mp4`.
  - Probe the reel and assert duration ≈ sum of clip durations (within 5s for keyframe drift).
  - Log first clip path so user can `open` that too.
  - Treat the visual check as a manual gate — the §0.5 print includes `[manual-check] play output/reels/justin_timberlake_highlight.mp4 to confirm coherence`.
- **Phase 4 has no CLIP, no numpy, no torch.** Pure ffmpeg subprocess orchestration. Expected `__main__` runtime on JT (4 clips, total 37.805s of output): ~10–30 seconds (4 ffmpeg invocations + 1 concat + ffprobe checks). Not the heavy phase.

</specifics>

<deferred>
## Deferred Ideas

- **`--cleanup` flag** for auto-deleting intermediate clips after concat → v2 (D-61, D-62).
- **Frame-precise cuts via `-ss` AFTER `-i` + re-encode** → v2; spec §10 says ~1s imprecision is acceptable.
- **Hardware encoder (h264_videotoolbox on Apple Silicon)** for the fallback re-encode path → v2 perf optimization. CPU `libx264 -crf 18` is fine for v1.
- **Audio loudness normalization across clips** (`loudnorm` filter) → out of scope; we're stream-copying audio when possible.
- **Subtitle/metadata preservation** → out of scope; body cam footage doesn't have these.
- **Progress-bar TUI for the per-clip extraction loop** → cosmetic; tqdm-style progress would be nice but not required.
- **Pre-validation in `extract_clip`** (probe the input video before invoking ffmpeg) → adds latency for no benefit; ffmpeg's own error handling is sufficient.

</deferred>

---

*Phase: 04-export*
*Context gathered: 2026-05-06*
*Mode: --auto (single pass; recommended option auto-selected per gray area)*
