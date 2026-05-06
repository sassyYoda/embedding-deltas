"""Phase 2: cosine deltas, median smoothing, rolling MAD, opt-in PELT.

Module API (consumed by Phase 3's clip_selection and Phase 5's pipeline.run()):
    compute_deltas(embeddings) -> (N-1,) float64
    smooth_deltas(raw, kernel_size=5) -> (N-1,) float64       # mode='reflect' (deviation from spec §5)
    mad_normalize(smoothed, window_samples=180) -> (N-1,) float64  # MAD floor 1e-3 (deviation from spec §5)
    detect_changepoints(smoothed, penalty=3.0) -> list[int]   # LAZY-IMPORTS ruptures
    score_index_to_timestamp(i, timestamps) -> float          # the ONLY index→seconds site (D-24)

Verification harness (spec §0.5) lives in `if __name__ == "__main__":` per D-34 / Phase 1 D-16.
Run: `python signal_processing.py [<video_name>]` to verify the module against the cached
Phase 1 fixture at `output/cache/<video_name>_{embeddings,timestamps}.npy`.

Phase 2 owns 3 pitfalls (research/PITFALLS.md): 8 (off-by-one alignment — synthetic test in
__main__), 9 (medfilt edge effects — switched to ndimage.median_filter mode='reflect'),
10 (MAD-zero — floor raised to 1e-3). Each is covered by a runtime assertion in the harness.

Two locked deviations from spec §5 verbatim (documented inline at each function):
  - D-27: scipy.ndimage.median_filter(mode='reflect') instead of the spec's
          scipy.signal medfilt (zero-padding edge effects → phantom dips at
          boundaries — Pitfall 9).
  - D-29: MAD floor 1e-3 (raised from the spec's much-looser float-precision guard);
          static-footage windows produce MAD in the [1e-7, 1e-3] range which would
          round-trip to ceiling-pegged scores under the looser guard (Pitfall 10).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation


def compute_deltas(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive embeddings.

    Embeddings are already L2-normalized (Phase 1 D-13 hard-asserts this), so cosine
    similarity reduces to a dot product. Returns ``1 - clip(dot, -1, 1)``.

    Args:
        embeddings: (N, 768) float32 L2-normalized array. N must be >= 2.

    Returns:
        (N-1,) float64 array. ``deltas[i]`` is the change ARRIVING AT frame i+1
        (spec §4 closing note). This is INDEX SPACE — no timestamps involved.
        Use ``score_index_to_timestamp(i, ts)`` for the conversion (D-23, D-24).

    Pitfall: callers must NOT inline ``timestamps[i]`` for delta i — the alignment
    is ``timestamps[i+1]`` (Pitfall 8 / research/PITFALLS.md §8).
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError(f"embeddings must be (N, 768); got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        raise ValueError(
            f"need at least 2 embeddings to compute deltas; got {embeddings.shape[0]}"
        )
    # Output dtype float64 (D-31). NOTE: numpy 2.x changed np.sum semantics so that
    # axis-reductions of float32 stay float32 (unlike the float64-by-default behavior
    # of older numpy versions referenced in D-22). Cast to float64 explicitly to keep
    # the contract Phase 3 / Phase 5 expect for downstream MAD/score arithmetic.
    dots = np.sum(embeddings[:-1] * embeddings[1:], axis=1, dtype=np.float64)
    dots = np.clip(dots, -1.0, 1.0)
    return 1.0 - dots


def score_index_to_timestamp(score_idx: int, timestamps: np.ndarray) -> float:
    """The ONLY index→seconds conversion site for delta/score arrays (D-24 / Pitfall 8).

    ``score[i]`` / ``smoothed[i]`` / ``delta[i]`` all align to ``timestamps[i+1]`` —
    the delta represents change ARRIVING AT frame i+1 (spec §4 closing note).
    Phase 3 (clip_selection.py) MUST import this function rather than computing
    ``timestamps[idx + 1]`` inline.

    Returns a Python float (not numpy scalar) so JSON serialization downstream
    (Phase 5) is clean.
    """
    return float(timestamps[score_idx + 1])


def smooth_deltas(raw_deltas: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Median-filter raw deltas with edge-preserving boundary handling (D-27 / Pitfall 9).

    DEVIATION from spec §5 verbatim: spec calls the ``scipy.signal`` medfilt
    function which zero-pads edges and produces phantom dips at the first/last 2
    samples (Pitfall 9 / research/PITFALLS.md §9). We use
    ``scipy.ndimage.median_filter(mode='reflect')`` instead — same kernel-5 median
    in the interior, reflected boundaries preserve edge values.

    Args:
        raw_deltas: 1-D array of cosine deltas (typically the output of compute_deltas).
        kernel_size: odd integer; default 5 per spec §5.

    Returns:
        1-D float64 array, same shape as input.
    """
    if kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be odd; got {kernel_size}")
    return median_filter(raw_deltas, size=kernel_size, mode="reflect")


def mad_normalize(smoothed: np.ndarray, window_samples: int = 180) -> np.ndarray:
    """Rolling MAD normalization over a centered window (D-29 / Pitfall 10).

    Implements spec §5 step 2 with one DEVIATION: MAD floor is 1e-3 (raised from
    the spec's much-looser float-precision guard — Pitfall 10 / research/PITFALLS.md §10).
    Static-footage windows produce MAD values in the [1e-7, 1e-3] range that round-trip
    through float math and produce ceiling-pegged scores when used as a divisor; 1e-3
    is well below typical body-cam baseline noise (~0.01) but well above
    float-precision artifacts.

    Output is clipped to [0.0, 10.0] per spec §5 and dtype is float64 (D-31).

    Args:
        smoothed: 1-D array (typically the output of smooth_deltas).
        window_samples: centered window full-width in samples; default 180 (= 90s at 2 fps).

    Returns:
        1-D float64 array, same shape as input, all values in [0.0, 10.0].
    """
    n = len(smoothed)
    half = window_samples // 2  # 90 for default 180
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        local = smoothed[lo:hi]
        local_median = float(np.median(local))
        # scale=1.0 → raw MAD (no consistency factor 1.4826); matches spec §5 intent
        # of "number of MADs above local median". scipy 1.16.2 default is scale=1.0
        # but we pin it explicitly so future scipy version bumps don't silently change.
        local_mad = float(median_abs_deviation(local, scale=1.0))
        # Pitfall 10: 1e-3 floor (raised from spec's much-looser float-precision guard) —
        # static-footage MAD ∈ [1e-7, 1e-3] poisons downstream scoring under the looser guard.
        if local_mad > 1e-3:
            out[i] = (smoothed[i] - local_median) / local_mad
        else:
            out[i] = 0.0  # static-footage window: emit zero
    return np.clip(out, 0.0, 10.0)  # spec §5


def detect_changepoints(smoothed: np.ndarray, penalty: float = 3.0) -> list[int]:
    """PELT changepoint detection (opt-in supplementary signal — D-32 / SIGP-02).

    LAZY IMPORT: ``import ruptures`` happens inside this function body. When
    ``--pelt`` is off (the default critical path), ``ruptures`` is NEVER loaded
    into ``sys.modules``. Verified by ``assert 'ruptures' not in sys.modules``
    after a non-PELT run (D-34 step 6).

    Args:
        smoothed: (M,) array (typically the output of smooth_deltas).
        penalty: PELT penalty parameter (default 3.0 per spec §5).

    Returns:
        list[int] of changepoint indices into ``smoothed``. The trailing
        ``len(signal)`` index that ``ruptures.Pelt.predict`` always emits is dropped.
    """
    import ruptures as rpt  # LAZY IMPORT — see docstring; do NOT move to module top.
    model = rpt.Pelt(model="rbf").fit(smoothed.reshape(-1, 1))
    changepoints = model.predict(pen=penalty)[:-1]  # drop trailing len(signal)
    return [int(cp) for cp in changepoints]
