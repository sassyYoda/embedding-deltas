#!/usr/bin/env python3
"""Resumable streaming extract for marcus_jordan.mp4.

Improvements over /tmp/extract_marcus_streaming.py:
- Checkpoints each chunk's embeddings to `output/cache/marcus_jordan_partial_NNN.npy`
  immediately after embedding it. On crash/sleep/kill, resume picks up at the
  next chunk instead of restarting from frame 0.
- On successful completion, concatenates all partials into the final
  `marcus_jordan_embeddings.npy` + `_timestamps.npy`, then deletes partials.
- Log goes to `output/cache/marcus_jordan_extract.log` (NOT /tmp, which gets
  cleaned by macOS) — survives reboot.
- Designed to be invoked via `caffeinate -i .venv/bin/python3 extract_marcus_resumable.py`
  so macOS's idle-sleep doesn't kill it.

Output: same final paths as extract.py --save-fixture:
  output/cache/marcus_jordan_embeddings.npy  (N, 768) float32 L2-normalized
  output/cache/marcus_jordan_timestamps.npy  (N,)   float64
"""
from __future__ import annotations

import gc
import os
import re
import sys
import time
from pathlib import Path

REPO = Path("/Users/aryanahuja/abel_police_interview")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import cv2
import numpy as np
import torch

from extract import load_model, embed_frames
from utils import probe_video_metadata, ensure_output_dirs

VIDEO = REPO / "videos" / "marcus_jordan.mp4"
TARGET_FPS = 2.0
BATCH_SIZE = 32
CHUNK_BATCHES = 8  # 8 batches × 32 frames = 256 frames per persistent chunk (~85s of video)
CACHE_EVICT_EVERY_N_BATCHES = 8

DIRS = ensure_output_dirs("marcus_jordan")
CACHE_DIR = DIRS["cache"]
LOG_PATH = CACHE_DIR / "marcus_jordan_extract.log"
EMB_FINAL = CACHE_DIR / "marcus_jordan_embeddings.npy"
TS_FINAL = CACHE_DIR / "marcus_jordan_timestamps.npy"
PARTIAL_PATTERN = re.compile(r"marcus_jordan_partial_(\d{4})\.npy$")
TS_PARTIAL_PATTERN = re.compile(r"marcus_jordan_partial_ts_(\d{4})\.npy$")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def list_partials() -> list[tuple[int, Path, Path]]:
    """Return sorted list of (chunk_idx, embedding_path, timestamp_path) for existing partials."""
    embs = {}
    tss = {}
    for p in CACHE_DIR.iterdir():
        m = PARTIAL_PATTERN.match(p.name)
        if m:
            embs[int(m.group(1))] = p
            continue
        m = TS_PARTIAL_PATTERN.match(p.name)
        if m:
            tss[int(m.group(1))] = p
    paired = []
    for idx in sorted(embs.keys()):
        if idx in tss:
            paired.append((idx, embs[idx], tss[idx]))
        else:
            log(f"WARN partial chunk {idx:04d} has emb but no ts — discarding")
            embs[idx].unlink()
    return paired


# Truncate log on first start of a fresh run (no partials exist yet); preserve on resume.
_existing_partials = list_partials()
if not _existing_partials and LOG_PATH.exists():
    LOG_PATH.unlink()
log(f"video: {VIDEO}")

if EMB_FINAL.exists():
    log(f"final fixture already exists: {EMB_FINAL} — exiting cleanly.")
    sys.exit(0)

meta = probe_video_metadata(str(VIDEO))
duration_sec = meta["duration_sec"]
log(f"duration={duration_sec:.3f}s codec={meta['codec']} is_vfr={meta['is_vfr']}")

sample_interval = 1.0 / TARGET_FPS
target_times_sec = np.arange(0.0, duration_sec, sample_interval)
N_TARGET = len(target_times_sec)
chunk_n_frames = CHUNK_BATCHES * BATCH_SIZE  # 256
n_chunks_total = (N_TARGET + chunk_n_frames - 1) // chunk_n_frames
log(f"target {N_TARGET} samples = {n_chunks_total} chunks of {chunk_n_frames} frames each")

# ── Resume detection ──────────────────────────────────────────────────────────
existing = list_partials()
if existing:
    last_idx = existing[-1][0]
    resume_chunk = last_idx + 1
    skip_n_frames = resume_chunk * chunk_n_frames
    log(f"RESUMING from chunk {resume_chunk} ({skip_n_frames} frames already done)")
    log(f"  found {len(existing)} partial chunk(s): {[i for i, _, _ in existing]}")
else:
    resume_chunk = 0
    skip_n_frames = 0
    log(f"FRESH START — no partials")

t0 = time.monotonic()
log(f"loading model...")
model, preprocess, device = load_model()
log(f"model loaded in {time.monotonic() - t0:.1f}s on device={device}")

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"failed to open {VIDEO}")

current_chunk_frames: list[np.ndarray] = []
current_chunk_timestamps: list[float] = []
current_chunk_idx = resume_chunk

t_started = time.monotonic()
n_processed_this_run = 0

for i, target_ms in enumerate(target_times_sec * 1000.0):
    if i < skip_n_frames:
        # Skip frames already covered by partial chunks. We still need to advance the
        # video reader for accuracy, but reading + discarding is wasteful — instead
        # we just won't `read` until we hit a frame we want.
        continue

    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
    ret, frame = cap.read()
    if not ret:
        log(f"EOS at sample {i}/{N_TARGET}")
        break
    actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_chunk_frames.append(frame_rgb)
    current_chunk_timestamps.append(actual_ms / 1000.0)
    n_processed_this_run += 1

    if len(current_chunk_frames) >= chunk_n_frames:
        # Flush this chunk: embed, save partials, evict cache.
        emb = embed_frames(current_chunk_frames, model, preprocess, device,
                           batch_size=BATCH_SIZE)
        ts = np.asarray(current_chunk_timestamps, dtype=np.float64)
        emb_path = CACHE_DIR / f"marcus_jordan_partial_{current_chunk_idx:04d}.npy"
        ts_path = CACHE_DIR / f"marcus_jordan_partial_ts_{current_chunk_idx:04d}.npy"
        np.save(emb_path, emb.astype(np.float32))
        np.save(ts_path, ts)

        elapsed = time.monotonic() - t_started
        rate = n_processed_this_run / elapsed if elapsed > 0 else 0.0
        remaining = N_TARGET - i - 1
        eta = remaining / rate if rate > 0 else float("inf")
        log(f"CHUNK {current_chunk_idx + 1}/{n_chunks_total} done "
            f"(frames {(current_chunk_idx) * chunk_n_frames}..{i}); "
            f"this-run elapsed={elapsed:.0f}s rate={rate:.2f}fps eta={eta:.0f}s")

        # Free memory.
        current_chunk_frames = []
        current_chunk_timestamps = []
        current_chunk_idx += 1

        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

# Flush trailing partial chunk (if any).
if current_chunk_frames:
    emb = embed_frames(current_chunk_frames, model, preprocess, device, batch_size=BATCH_SIZE)
    ts = np.asarray(current_chunk_timestamps, dtype=np.float64)
    emb_path = CACHE_DIR / f"marcus_jordan_partial_{current_chunk_idx:04d}.npy"
    ts_path = CACHE_DIR / f"marcus_jordan_partial_ts_{current_chunk_idx:04d}.npy"
    np.save(emb_path, emb.astype(np.float32))
    np.save(ts_path, ts)
    log(f"FINAL CHUNK {current_chunk_idx + 1}/{n_chunks_total} done "
        f"(trailing partial of {len(current_chunk_frames)} frames)")
    current_chunk_frames = []
    current_chunk_timestamps = []
    current_chunk_idx += 1

cap.release()

# ── Concatenate partials → final fixture ──────────────────────────────────────
log(f"concatenating partials into final fixture...")
all_partials = list_partials()
emb_chunks = [np.load(p) for _, p, _ in all_partials]
ts_chunks = [np.load(p) for _, _, p in all_partials]
embeddings = np.concatenate(emb_chunks, axis=0)
timestamps = np.concatenate(ts_chunks, axis=0)
log(f"final embeddings shape={embeddings.shape} dtype={embeddings.dtype}")
log(f"timestamps shape={timestamps.shape} first 5={timestamps[:5].tolist()} "
    f"last={timestamps[-1]:.3f}s")

norms = np.linalg.norm(embeddings, axis=1)
assert np.allclose(norms, 1.0, atol=1e-5), \
    f"L2-norm fail post-concat: min={norms.min()} max={norms.max()}"
log(f"L2-norm check: PASS  (norms in [{norms.min():.6f}, {norms.max():.6f}])")

drift = abs(timestamps[-1] - duration_sec)
log(f"drift={drift:.3f}s (last_ts={timestamps[-1]:.3f}s vs probe={duration_sec:.3f}s)")
assert drift < 5.0, f"timestamp drift {drift}s exceeds 5s tolerance"

np.save(EMB_FINAL, embeddings.astype(np.float32))
np.save(TS_FINAL, timestamps.astype(np.float64))
log(f"wrote {EMB_FINAL}")
log(f"wrote {TS_FINAL}")

# Clean up partials only after final fixture is durable on disk.
for _, p, ts_p in all_partials:
    p.unlink()
    ts_p.unlink()
log(f"cleaned up {len(all_partials)} partial chunks")

total_elapsed = time.monotonic() - t_started
log(f"DONE. this-run elapsed={total_elapsed:.0f}s "
    f"({n_processed_this_run} new frames, total {embeddings.shape[0]} frames)")
