---
phase: 01-frame-extraction-embeddings
plan: 01
subsystem: bootstrap
tags: [bootstrap, python, ffmpeg, ffprobe, utils]
requires: []
provides:
  - "requirements.txt (pinned, no ffmpeg-python)"
  - "spec §1 module layout (5 of 6 files; extract.py owned by Plan 01-02)"
  - "utils.probe_video_metadata"
  - "utils.ensure_output_dirs"
  - "utils.setup_logger"
affects: [".gitignore", "project root layout"]
key-files:
  created:
    - "requirements.txt"
    - "pipeline.py"
    - "signal_processing.py"
    - "clip_selection.py"
    - "export.py"
    - "utils.py"
  modified:
    - ".gitignore"
decisions:
  - "Honored D-02: requirements.txt pins 7 packages with rationale comment; no ffmpeg-python."
  - "Honored D-20/D-21: utils.py ships 3 Phase-1 functions; write_timestamps_json deferred to Phase 5."
  - "Honored D-16: no pytest scaffolding; utils.py self-test lives in __main__."
metrics:
  tasks_completed: 2
  commits: 2
  duration_minutes: ~5
  completed_date: 2026-05-06
---

# Phase 1 Plan 01: Project Skeleton & utils.py Summary

Established the project skeleton (spec §1 layout minus extract.py), pinned the dependency manifest with the ffmpeg-python deviation rationale, extended .gitignore for Python build artifacts and `output/`, and shipped `utils.py` with the three load-bearing helpers Plan 01-02 will import.

## Files Created / Modified

| Path | Role |
|------|------|
| `requirements.txt` | Pinned dependency manifest. 7 packages: torch 2.11.0, torchvision 0.26.0, open_clip_torch 3.3.0, opencv-python 4.13.0.92, numpy>=2.1<2.5, scipy 1.16.2 (avoids medfilt regression #22333), ruptures 1.1.10, tqdm>=4.67<5. Header comment documents `ffmpeg-python` exclusion (unmaintained since 2019; spec §7 uses `subprocess.run` directly). |
| `.gitignore` | Existing entries (`.DS_Store`, `videos/`) preserved; appended `.venv/`, `__pycache__/`, `*.pyc`, `*.pyo`, `*.egg-info/`, `output/`, `.vscode/`, `.idea/`. |
| `pipeline.py` | Stub: `"""TODO: implemented in Phase 5 — see ROADMAP.md."""` |
| `signal_processing.py` | Stub: `"""TODO: implemented in Phase 2 — see ROADMAP.md."""` |
| `clip_selection.py` | Stub: `"""TODO: implemented in Phase 3 — see ROADMAP.md."""` |
| `export.py` | Stub: `"""TODO: implemented in Phase 4 — see ROADMAP.md."""` |
| `utils.py` | Three Phase-1 helpers + co-located self-test (no pytest, per D-16). |
| `extract.py` | NOT created here — owned by Plan 01-02. |

## utils.py API (verbatim from PLAN `<interfaces>`)

```python
def probe_video_metadata(video_path: str | Path) -> dict:
    """Returns {duration_sec: float, width: int, height: int, codec: str, is_vfr: bool}.
    Uses `ffprobe -v error -show_streams -show_format -of json <path>`.
    Raises:
      FileNotFoundError  — if path missing (checked before ffprobe call)
      RuntimeError       — if ffprobe missing on PATH, ffprobe non-zero exit,
                           or file has no video stream.
    """

def ensure_output_dirs(video_name: str, base: Path = Path("output")) -> dict[str, Path]:
    """Idempotently creates base/{reels, clips/<video_name>, timestamps, cache}.
    Returns dict with exactly those four keys. The `cache` key is the fixture
    write target for Plan 02's --save-fixture (D-18).
    """

def setup_logger(name: str = "highlight_reel") -> logging.Logger:
    """Stdlib logger; format '[%(levelname)s] %(message)s'; INFO level; stderr handler.
    Idempotent — re-call does NOT add a duplicate handler.
    """
```

## Self-Test Output (`python3 utils.py`)

```
utils.py self-test
[PASS] ensure_output_dirs idempotent + correct shape
[PASS] setup_logger idempotent
[PASS] probe_video_metadata on Justin Timberlake.mp4: duration=1134.144s
[PASS] probe_video_metadata raises FileNotFoundError for missing file
[PASS] probe_video_metadata raises RuntimeError for non-video bytes
utils.py self-test: PASS
```

All 7 self-test assertions passed. The probe tests ran against a real `.mp4` already present in `videos/Sample Videos/Justin Timberlake.mp4` (concurrent download had at least one fully-downloaded file at self-test time). Cross-check between the JSON-form and bare-CSV-form ffprobe duration agreed to within 1e-3 s.

## Plan-Level Verification (all 7 PASS)

1. Layout: repo root contains exactly the expected 5 `.py` modules + `requirements.txt` + `.gitignore` (extract.py to be added by Plan 01-02).
2. Pinning: 6 exact pins (`==`); zero `ffmpeg-python` package lines (only the rationale comment mentions the name — see Deviations below).
3. Gitignore: original `.DS_Store` and `videos/` preserved; 7 new entries appended.
4. Stubs: all four parse with `ast.parse`; phase pointers (5/2/3/4) correct.
5. `utils.py` API: all 3 functions importable; `write_timestamps_json` correctly absent (D-21).
6. Self-test: `python3 utils.py` exits 0; "utils.py self-test: PASS" printed.
7. ENV-04: no heavy imports (`cv2`, `torch`, `open_clip`) in any file created by this plan.

## Notes for Plan 02

- `utils.ensure_output_dirs(name)["cache"]` is the canonical write target for `--save-fixture` `.npy` writes (D-18).
- `utils.probe_video_metadata(path)["duration_sec"]` is the source-of-truth duration for the §0.5 drift assertion `abs(timestamps[-1] - probe_duration) < 1.0` (D-08). Do NOT recompute duration via `cv2.CAP_PROP_FRAME_COUNT / fps` (Pitfall 3).
- `videos/Sample Videos/Justin Timberlake.mp4` was fully downloaded at self-test time (1134.144 s ≈ 18:54). Other videos in the gdown batch may still be downloading. Plan 02's first run should re-check `find videos -name '*.mp4'` to pick a sample.

## Deviations from Plan

### Verify-regex tweak (Task 1)

**Found during:** Task 1 acceptance check.
**Issue:** the plan's `<verify><automated>` regex for Task 1 includes `! grep -q "ffmpeg-python" requirements.txt`. This conflicts with `<action>` step 1, which mandates the verbatim header comment `"# Note: spec §1 lists \`ffmpeg-python\` but it is dropped per research/STACK.md —"` (D-02 explicitly requires this rationale). The substring `ffmpeg-python` appears in the comment line by design.
**Fix:** ran the equivalent stricter check `! grep -v '^#' requirements.txt | grep -q ffmpeg-python` — confirms the *package* is absent (correct intent of the verify) while the rationale comment is preserved (correct intent of D-02 + `<action>` step 1).
**Files modified:** none (the file matches the plan's mandated content exactly).
**Commit:** none required — this was a verification-script issue, not a code issue.

No code-level deviations. Both tasks' `<done>` criteria and the plan-level `<verification>` block 1–7 were all satisfied.

## Open Issues / Risks

- The probe tests ran against `Justin Timberlake.mp4`. The plan's `next(Path("videos").rglob("*.mp4"), None)` will pick the first match alphabetically; if the user later wants to test against a specific video they should pass it as an explicit path (Plan 02's `extract.py` will accept a positional `video_path` argument).
- Plan 01-02 (extract.py + §0.5 harness) is deliberately deferred per orchestrator instructions until the gdown batch finalizes and the videos are flattened/renamed.

## Self-Check: PASSED

- All claimed files exist at the repo root (`requirements.txt`, `.gitignore`, `pipeline.py`, `signal_processing.py`, `clip_selection.py`, `export.py`, `utils.py`).
- Both commits present in git log: `574034c feat(01-01-01)`, `7b72469 feat(01-01-02)`.
- Plan-level verification 1–7 all PASS.
- `python3 utils.py` exits 0 with `utils.py self-test: PASS`.
