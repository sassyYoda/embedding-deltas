"""Utility helpers for the Body Cam Highlight Reel pipeline (Phase 1 subset, per D-20).

Implements:
  - probe_video_metadata(path) -> dict   # source of truth for video metadata via ffprobe (D-05)
  - ensure_output_dirs(video_name) -> dict[str, Path]   # idempotent mkdir for output tree
  - setup_logger(name) -> logging.Logger   # optional stdlib logger; not used by extract.py's §0.5 prints

Deferred to Phase 5 (D-21):
  - write_timestamps_json — JSON schema and 3-state PELT field belong with the orchestrator.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path


def probe_video_metadata(video_path: str | Path) -> dict:
    """Single source of truth for video metadata. Uses ffprobe (NOT cv2).

    Returns:
        {
            "duration_sec": float,   # from container's format.duration
            "width": int,
            "height": int,
            "codec": str,            # e.g. "h264"
            "is_vfr": bool,          # avg_frame_rate != r_frame_rate (diagnostic only — D-06)
        }

    Raises:
        FileNotFoundError: if video_path does not exist (checked before invoking ffprobe).
        RuntimeError: if ffprobe is not on PATH, ffprobe exits non-zero, or the file
            has no video stream.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe not found on PATH; install ffmpeg (which bundles ffprobe). "
            "On macOS: `brew install ffmpeg`."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {e.stderr.strip()}"
        ) from e

    data = json.loads(proc.stdout)
    video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError(f"no video stream found in {video_path}")

    v = video_streams[0]
    duration_sec = float(data["format"]["duration"])

    # is_vfr: avg_frame_rate differs from r_frame_rate when source is VFR.
    # Diagnostic only — sampling strategy never branches on this (D-06).
    avg_fr = v.get("avg_frame_rate", "0/0")
    r_fr = v.get("r_frame_rate", "0/0")
    is_vfr = avg_fr != r_fr

    return {
        "duration_sec": duration_sec,
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "codec": v.get("codec_name", "unknown"),
        "is_vfr": is_vfr,
    }


def ensure_output_dirs(video_name: str, base: Path = Path("output")) -> dict[str, Path]:
    """Idempotently create output/{reels,clips/<video_name>,timestamps,cache}.

    Returns a dict with keys 'reels', 'clips', 'timestamps', 'cache' mapping to
    the (now-existing) Path objects. The 'clips' Path is base/clips/<video_name>
    (per-video subdirectory); the others are flat under base/.

    The 'cache' key is the fixture write target for Plan 02's --save-fixture (D-18).
    """
    paths = {
        "reels":      base / "reels",
        "clips":      base / "clips" / video_name,
        "timestamps": base / "timestamps",
        "cache":      base / "cache",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(name: str = "highlight_reel") -> logging.Logger:
    """Return a stdlib Logger with [%(levelname)s] %(message)s format; idempotent.

    Phase 1's extract.py uses print() directly (matches spec §0.5 verbatim). This helper
    is provided so Phase 5 can opt into structured logging without re-architecting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # idempotent — only attach handler on first call
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    import tempfile

    print("utils.py self-test")

    # ── Tests 5, 6, 7 (no video file required) ─────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        tmp_base = Path(td) / "output"
        paths = ensure_output_dirs("test_video", base=tmp_base)
        assert set(paths.keys()) == {"reels", "clips", "timestamps", "cache"}, paths.keys()
        assert all(isinstance(p, Path) for p in paths.values())
        assert paths["clips"].name == "test_video", paths["clips"]
        assert all(p.is_dir() for p in paths.values())
        # idempotent re-call
        paths2 = ensure_output_dirs("test_video", base=tmp_base)
        assert paths == paths2
        print("[PASS] ensure_output_dirs idempotent + correct shape")

    log1 = setup_logger("utils_self_test")
    h_count = len(log1.handlers)
    log2 = setup_logger("utils_self_test")
    assert len(log2.handlers) == h_count, (
        f"setup_logger added duplicate handler: {h_count} -> {len(log2.handlers)}"
    )
    print("[PASS] setup_logger idempotent")

    # ── Tests 1, 2, 3, 4 (video file required) ─────────────────────────────────
    sample = next(Path("videos").rglob("*.mp4"), None)
    if sample is None:
        print("[SKIP] no .mp4 found under videos/ — skipping probe tests (re-run after download)")
    else:
        meta = probe_video_metadata(sample)
        assert set(meta.keys()) == {"duration_sec", "width", "height", "codec", "is_vfr"}, meta.keys()
        assert meta["duration_sec"] > 0, meta
        assert meta["width"] > 0 and meta["height"] > 0, meta
        assert isinstance(meta["codec"], str) and meta["codec"], meta
        assert isinstance(meta["is_vfr"], bool), meta
        # cross-check duration against the bare-CSV ffprobe form (Gap 3 invocation a)
        bare = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "csv=p=0", str(sample)],
            capture_output=True, text=True, check=True,
        )
        bare_duration = float(bare.stdout.strip())
        assert abs(meta["duration_sec"] - bare_duration) < 1e-3, (meta["duration_sec"], bare_duration)
        print(f"[PASS] probe_video_metadata on {sample.name}: duration={meta['duration_sec']:.3f}s")

        try:
            probe_video_metadata(Path("/nonexistent/__definitely_not_a_file__.mp4"))
        except FileNotFoundError as e:
            assert "video not found" in str(e), e
            print("[PASS] probe_video_metadata raises FileNotFoundError for missing file")
        else:
            raise AssertionError("expected FileNotFoundError for missing file")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"not a real video file " * 2)
            tmp_path = Path(tmp.name)
        try:
            probe_video_metadata(tmp_path)
        except RuntimeError as e:
            msg = str(e).lower()
            assert "ffprobe failed" in msg or "no video stream" in msg, e
            print("[PASS] probe_video_metadata raises RuntimeError for non-video bytes")
        else:
            raise AssertionError("expected RuntimeError for non-video file")
        finally:
            tmp_path.unlink(missing_ok=True)

    print("utils.py self-test: PASS")
