#!/usr/bin/env python3

"""Selectable 2D CFAR detectors for range-azimuth heatmaps.

All of these operate on natural-log magnitude input, matching what
dsp.compute_range_azimuth_capon() produces (`np.log(np.abs(...))`, not dB or
10*log10). The threshold is therefore additive in nats, not a linear power
multiplier.

Log-domain CFAR
---------------
Textbook CA-CFAR averages linear power and multiplies by alpha. Averaging
logs instead gives the geometric rather than the arithmetic mean, which for
exponentially distributed noise power sits a constant ~0.577 nats (the
Euler-Mascheroni constant) below it. A log turns scaling into an offset, so
that bias is constant and scale-invariant and is absorbed into the
threshold. Log-domain CA-CFAR is a valid detector whose threshold reads
differently from the textbook alpha.

What log-domain averaging does not fix is heterogeneous clutter, which is
why this module offers several detectors instead of more options on
ca_cfar_2d.

Choosing a method for the tabletop case
---------------------------------------
  ca  Cell-averaging, the pipeline default. Optimal for a single target in
      homogeneous noise. Its failure mode is target masking: a second
      object inside the training window of the cell under test (CUT) raises
      the estimated noise floor and suppresses the detection. On a tabletop
      with several objects inside a ~25 deg azimuth beam, that is the normal
      case.

  os  Ordered-statistic. Takes the k-th smallest training cell instead of
      the mean, so a minority of contaminated cells (an interfering target)
      cannot drag the estimate up. Recommended for multi-object tabletop
      scenes. Costs a sort per cell.

      OS-CFAR is exactly equivalent in the log and linear domains: the log
      is monotonic, so the k-th order statistic of the logs is the log of
      the k-th order statistic.

      Cost, measured on a Jetson over an 80x141 heatmap with the default
      window (guard 2, noise 4): ca 3.5 ms, go/so 1.7 ms, os 74 ms. OS is
      about 20x CA but still allows about 13 Hz on these maps, above the
      5 Hz of near.cfg and near_mimo.cfg. test.cfg (20 Hz, 224 range bins)
      was not timed. If OS becomes the limit, reduce noise_len.

      It stays robust only while at most (1 - rank_frac) of the training
      set is contaminated. Past that its estimate is dragged up too.

  go  Greatest-of. Splits the training window into leading and lagging
      halves along range and takes the larger mean. Suppresses false alarms
      at a clutter edge, such as the range bin where the table surface's
      return begins, at the cost of sensitivity.

  pct Global percentile. Not a CFAR: it keeps the top (100 - threshold_pct)%
      of cells in the whole frame, with no local noise estimate.

      It exists because a local-neighbourhood detector did not work on this
      radar's near-field heatmaps. On the single-bottle recording (Experiment 1) the
      usable dynamic range was about 17 dB (median cell about 88 dB, 99th
      percentile about 105 dB) and structured by sidelobe energy, so CA-CFAR
      flagged 7-16% of all cells at every threshold tried from 3 to 30 dB,
      while a global percentile isolated the near-field object, clustering
      at 0.28-0.60 m. In the elevation work, a p99 power gate was also the
      only measure that tracked phase stability.

      Trade-off: a global threshold has no constant-false-alarm property. It
      does not adapt to a changing noise floor, and a single dominant
      reflector (a flat backdrop facing the radar) takes the whole budget.
      It works for a static scene with fixed geometry. Prefer 'os' when a
      local estimate works, and 'pct' when it does not.

  so  Smallest-of. Takes the smaller half-mean. Better than CA at resolving
      two targets closely spaced in range (each protects the other's
      estimate), but prone to false alarms at clutter edges.

The window geometry is the same for ca, os, go and so: a
(2*(guard_len+noise_len)+1)^2 square window, minus a (2*guard_len+1)^2 guard
region around the cell under test. Cells closer to the border than
guard_len+noise_len are never tested (the mask stays False there).
"""

import numpy as np
from numba import njit


CFAR_METHODS = ('ca', 'os', 'go', 'so', 'pct')


@njit(cache=True)
def ca_cfar_2d(heatmap, guard_len=2, noise_len=4, threshold_abs=2.5):
    """
    Applies Cell-Averaging CFAR (CA-CFAR) on a 2D heatmap.
    Designed for Log-scale input (heatmap values are log magnitudes).

    Detection logic: Cell > Local_Mean + Threshold

    Args:
        heatmap (np.ndarray): 2D array of log-magnitude values.
        guard_len (int): Number of guard cells around CUT (Cell Under Test).
        noise_len (int): Number of noise training cells around guard cells.
        threshold_abs (float): Generalized threshold value added to local noise level.

    Returns:
        mask (np.ndarray): Boolean mask of detections.
    """
    rows, cols = heatmap.shape
    mask = np.zeros((rows, cols), dtype=np.bool_)

    # Pre-calculate area counts
    side_full = 2 * (noise_len + guard_len) + 1
    side_guard = 2 * guard_len + 1
    num_train = (side_full * side_full) - (side_guard * side_guard)

    r_start = noise_len + guard_len
    r_end = rows - (noise_len + guard_len)
    c_start = noise_len + guard_len
    c_end = cols - (noise_len + guard_len)

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):

            outer_sum = 0.0
            inner_sum = 0.0

            # Simple sliding window (can be optimized with Integral Image for speed)
            # For typical radar heatmap sizes (64x180), naive is fast enough with Numba

            # Iterate over the training + guard window
            for i in range(-noise_len - guard_len, noise_len + guard_len + 1):
                for j in range(-noise_len - guard_len, noise_len + guard_len + 1):
                    val = heatmap[r + i, c + j]

                    # Add to outer sum (includes guard and CUT for now)
                    outer_sum += val

                    # If inside guard region, add to inner sum
                    if abs(i) <= guard_len and abs(j) <= guard_len:
                        inner_sum += val

            # Subtract inner (guard + CUT) from outer to get only training cells
            noise_sum = outer_sum - inner_sum
            noise_level = noise_sum / num_train

            # Check detection
            # For log data: CUT > Mean + Threshold
            if heatmap[r, c] > (noise_level + threshold_abs):
                mask[r, c] = True

    return mask


@njit(cache=True)
def os_cfar_2d(heatmap, guard_len=2, noise_len=4, threshold_abs=2.5, rank_frac=0.75):
    """Ordered-statistic CFAR. Noise estimate = k-th smallest training cell.

    rank_frac selects k as rank_frac * num_train (clamped to a valid index).
    0.75 is the usual starting point: high enough to sit above the noise
    bulk, and low enough that the top ~25% of training cells, where an
    interfering target's energy lands, cannot influence the estimate. Lower
    it if targets are still masked. Raise it toward 1.0 to approach
    CA-CFAR's sensitivity (and its masking behaviour).
    """
    rows, cols = heatmap.shape
    mask = np.zeros((rows, cols), dtype=np.bool_)

    side_full = 2 * (noise_len + guard_len) + 1
    side_guard = 2 * guard_len + 1
    num_train = (side_full * side_full) - (side_guard * side_guard)
    if num_train < 1:
        return mask

    k = int(rank_frac * num_train)
    if k < 0:
        k = 0
    if k > num_train - 1:
        k = num_train - 1

    r_start = noise_len + guard_len
    r_end = rows - (noise_len + guard_len)
    c_start = noise_len + guard_len
    c_end = cols - (noise_len + guard_len)

    train = np.empty(num_train, dtype=np.float64)

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            n = 0
            for i in range(-noise_len - guard_len, noise_len + guard_len + 1):
                for j in range(-noise_len - guard_len, noise_len + guard_len + 1):
                    if abs(i) <= guard_len and abs(j) <= guard_len:
                        continue  # guard region + CUT, never training
                    train[n] = heatmap[r + i, c + j]
                    n += 1

            # Sort the reused buffer in place: no per-cell allocation (about
            # 11k per 80x141 frame). A partial selection up to k is
            # O(k*num_train), with k=105 of 140, and measured 8x slower here
            # than one O(n log n) sort.
            train[:n].sort()

            if heatmap[r, c] > (train[k] + threshold_abs):
                mask[r, c] = True

    return mask


@njit(cache=True)
def _go_so_cfar_2d(heatmap, guard_len, noise_len, threshold_abs, take_greatest):
    """Shared kernel for GO/SO-CFAR. The halves are split along the range axis.

    Range is the split axis rather than azimuth because the clutter edges
    on a tabletop (the table surface's leading edge, the near-field leakage
    boundary) are range-aligned. They separate near rows from far rows, so
    each half-window stays homogeneous.
    """
    rows, cols = heatmap.shape
    mask = np.zeros((rows, cols), dtype=np.bool_)

    span = noise_len + guard_len
    side_full = 2 * span + 1
    # Each half = the rows strictly beyond the guard region on one side.
    num_half = noise_len * side_full
    if num_half < 1:
        return mask

    r_start = span
    r_end = rows - span
    c_start = span
    c_end = cols - span

    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            lead_sum = 0.0
            lag_sum = 0.0
            for i in range(-span, span + 1):
                if abs(i) <= guard_len:
                    continue  # guard rows (and the CUT row) contribute to neither half
                for j in range(-span, span + 1):
                    val = heatmap[r + i, c + j]
                    if i < 0:
                        lag_sum += val
                    else:
                        lead_sum += val

            lag_level = lag_sum / num_half
            lead_level = lead_sum / num_half
            if take_greatest:
                noise_level = lag_level if lag_level > lead_level else lead_level
            else:
                noise_level = lag_level if lag_level < lead_level else lead_level

            if heatmap[r, c] > (noise_level + threshold_abs):
                mask[r, c] = True

    return mask


def go_cfar_2d(heatmap, guard_len=2, noise_len=4, threshold_abs=2.5):
    """Greatest-of CFAR: fewer false alarms at range-aligned clutter edges."""
    return _go_so_cfar_2d(heatmap, guard_len, noise_len, threshold_abs, True)


def so_cfar_2d(heatmap, guard_len=2, noise_len=4, threshold_abs=2.5):
    """Smallest-of CFAR: better at resolving targets closely spaced in range."""
    return _go_so_cfar_2d(heatmap, guard_len, noise_len, threshold_abs, False)


def pct_detector_2d(heatmap, threshold_pct=99.5):
    """Global percentile threshold. See the module docstring on 'pct'."""
    hm = np.asarray(heatmap, dtype=np.float64)
    return hm >= np.percentile(hm, float(threshold_pct))


def cfar_2d(heatmap, method='ca', guard_len=2, noise_len=4, threshold=2.5,
            os_rank_frac=0.75, threshold_pct=99.5):
    """Dispatch to one of CFAR_METHODS. See the module docstring for choice guidance.

    heatmap is converted to a contiguous float64 array. The numba kernels are
    compiled per input dtype, and letting both float32 and float64 through
    would double compile time and cache size.
    """
    method = str(method).lower()
    hm = np.ascontiguousarray(heatmap, dtype=np.float64)
    guard_len = int(guard_len)
    noise_len = int(noise_len)
    threshold = float(threshold)

    if method == 'pct':
        # No sliding window, so the window-fit check below does not apply.
        if hm.ndim != 2:
            raise ValueError(f'cfar_2d expects a 2D heatmap, got shape {hm.shape}')
        return pct_detector_2d(hm, threshold_pct)

    if hm.ndim != 2:
        raise ValueError(f'cfar_2d expects a 2D heatmap, got shape {hm.shape}')

    span = guard_len + noise_len
    if hm.shape[0] <= 2 * span or hm.shape[1] <= 2 * span:
        # The window does not fit, so every cell would be a border cell.
        # Raise instead of returning an all-False mask from the kernels'
        # empty loops, which would look like a valid frame with no
        # detections.
        raise ValueError(
            f'CFAR window (guard_len={guard_len} + noise_len={noise_len}) does not fit '
            f'in a {hm.shape[0]}x{hm.shape[1]} heatmap')

    if method == 'ca':
        return ca_cfar_2d(hm, guard_len, noise_len, threshold)
    if method == 'os':
        return os_cfar_2d(hm, guard_len, noise_len, threshold, float(os_rank_frac))
    if method == 'go':
        return go_cfar_2d(hm, guard_len, noise_len, threshold)
    if method == 'so':
        return so_cfar_2d(hm, guard_len, noise_len, threshold)
    raise ValueError(f"Unknown cfar_method '{method}'; expected one of {CFAR_METHODS}")
