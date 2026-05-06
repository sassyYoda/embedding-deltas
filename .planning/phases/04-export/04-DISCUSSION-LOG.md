# Phase 4: Export - Discussion Log

> **Audit trail only.** Decisions captured in `04-CONTEXT.md`.

**Date:** 2026-05-06
**Phase:** 4 — Export
**Mode:** `--auto`
**Areas discussed:** `-ss` placement, concat fallback, manifest path safety, validation pre-flight, verification harness, cleanup behavior, dependency surface

---

## Area A — `-ss` Placement (Pitfall 11)

| Option | Selected |
|--------|----------|
| `-ss` BEFORE `-i` (spec §7 verbatim) — fast seek, stream copy compatible, ~1s imprecision | ✓ |
| `-ss` AFTER `-i` — frame-precise but forces re-encode | |

**Auto-selected:** Before `-i`. Spec §10 row 12 says ~1s imprecision is acceptable. Frame precision is v2 only.

---

## Area B — Concat Fallback When Demuxer Fails

| Option | Selected |
|--------|----------|
| concat-filter re-encode (`-c:v libx264 -crf 18`) on demuxer failure | ✓ |
| Hard-fail on demuxer error | |

**Auto-selected:** Fallback path. Concat demuxer is fragile across codec/timebase variants (Pitfall 12). The fallback re-encodes — slower but guaranteed correctness. Logged via `[concat] FALLBACK: ...` so the deviation is visible.

---

## Area C — Pre-flight Validation

| Option | Selected |
|--------|----------|
| `validate_clips_for_concat()` runs `ffprobe` first; demuxer or fallback chosen accordingly | ✓ |
| Try demuxer always; fall back on subprocess error | |

**Auto-selected:** Pre-flight. Avoids the demuxer-error-after-N-seconds path (which produces partial reels then errors); upfront ffprobe is cheap.

---

## Area D — Manifest Path Safety (Pitfall 13)

| Option | Selected |
|--------|----------|
| Absolute paths via `os.path.abspath`; `-safe 0`; escape single-quotes in filenames | ✓ |
| Relative paths; rely on `cwd` consistency | |

**Auto-selected:** Defensive paths. Pitfall 13 documents specific failure modes for relative paths and unescaped quotes.

---

## Area E — Verification Harness

| Option | Selected |
|--------|----------|
| `if __name__ == "__main__":` block (matches Phases 1–3) | ✓ |
| Separate test file (pytest scaffolding) | |

**Auto-selected:** Inline `__main__` block.

---

## Area F — Cleanup Behavior

| Option | Selected |
|--------|----------|
| Keep intermediate clips on disk after concat; future `--cleanup` flag is v2 | ✓ |
| Auto-delete intermediates immediately after successful concat | |

**Auto-selected:** Keep. Spec §0.5 verifies "each intermediate clip exists and is non-zero bytes" — at any time, not just immediately. Debug value if reel concat fails. Disk cost is small.

---

## Area G — Dependency Surface for `export.py`

| Option | Selected |
|--------|----------|
| stdlib + `utils` only — NO numpy, torch, cv2, open_clip, ruptures | ✓ |
| Allow `numpy` for clip-list manipulation if convenient | |

**Auto-selected:** Minimal surface. Phase 4 is purely subprocess orchestration. Static check `'import torch/cv2/ruptures' not in src` enforces.

---

## Claude's Discretion

- Type-hint style, docstring style.
- `tuple[bool, str]` vs `dataclass` for `validate_clips_for_concat` return type → tuple for v1.
- ffmpeg `-loglevel` flag → omit; `capture_output=True` already silences.
- `pathlib.Path` support via `os.fspath(...)` wrapping.
- Reel-duration drift tolerance → `< 5.0s`.

## Deferred Ideas

- `--cleanup` flag → v2
- Frame-precise cuts (`-ss` after `-i` + re-encode) → v2
- Hardware encoder for fallback (`h264_videotoolbox`) → v2 perf
- Audio loudness normalization → out of scope
- Progress-bar TUI → cosmetic
- Pre-validation in `extract_clip` → unnecessary
