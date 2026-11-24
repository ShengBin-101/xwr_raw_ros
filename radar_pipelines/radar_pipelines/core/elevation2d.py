#!/usr/bin/env python3

"""Per-cell elevation from the AOP's two virtual-array rows.

Exploratory. This interferometric elevation has never been checked against
ground truth (a target at a known height), and it is off by default: the
point-cloud pipeline's elevation_mode defaults to 'none'. Treat any absolute
elevation it produces as unverified. nodes/elevation_calibration.py is the
tool for that check.

The problem
-----------
xwr_raw/elevation.py's estimators return one elevation angle per range bin,
a 1D array indexed by range. Every azimuth at a given range shares it. In a
scene with two objects at the same range but different bearings (the normal
tabletop case), at most one of them can be right, and the point cloud gets
a single elevation across the whole row.

They are also a peak search (Capon or Bartlett over a scan grid) across a
2-element aperture. A 2-element array's beam pattern is nearly flat, so the
search has no confident interior peak and saturates at the scan boundary.
In one live frame, about half of 63 points reported an elevation within a
few tenths of a degree of the scan limit (-45.0 to -45.3 deg at
angle_range=45), including points from a real target at 0.44-0.55 m.
Switching from Capon to Bartlett did not change this, because the problem is
the search, not the beamformer.

What this does instead
----------------------
Two elements cannot resolve two targets in elevation. That is an aperture
limit and nothing here changes it. Two elements can estimate the angle of a
single dominant scatterer by interferometry: the phase difference between
them is a direct, closed-form measurement, with no search to saturate.

The AOP's 12-channel TDM-MIMO cube contains two full rows of the same
4-element azimuth ULA (2x2 RX grid at lambda/2, RX0/RX1 sharing a column, see
xwr_raw/elevation.py's geometry notes and the antenna diagram):

    row B (the validated azimuth ULA)  [11, 9, 7, 5]   RX in {1, 3}
    row A (same X positions, other Z)  [10, 8, 6, 4]   RX in {0, 2}

Element j of row B pairs vertically with element j of row A. So:

  1. Range-FFT and azimuth-beamform each row separately, on the same
     steering grid the detection heatmap uses.
  2. Per (range, azimuth) cell, the row-to-row phase difference is
     dphi = 2*pi*(d/lambda)*sin(elevation). With d = lambda/2 that is
     dphi = pi*sin(elevation), unambiguous over the full +-90 deg (no
     grating lobes at lambda/2).
  3. elevation = arcsin(dphi / pi).

The result is elevation per cell, not per range bin, and it cannot saturate
at a boundary because there is no boundary: it is the arcsin of a measured
ratio.

Limits
------
  * Still a 2-element aperture. One dominant scatterer per cell is assumed.
    Two objects at the same range and bearing but different heights produce
    one blended phase, and the reported angle lies somewhere between them
    without any error. Coherence (below) is the only warning.
  * Accuracy is bound by SNR, not by the aperture: sigma_phase ~
    1/sqrt(2*SNR), so sigma_elevation ~ sigma_phase / (pi*cos(el)). Low-SNR
    cells give near-random angles. `coherence` measures how consistently the
    chirps agree on the phase difference, and gating on it discards cells
    whose elevation is noise. CFAR already removes most of these, but not
    all.
  * The sign depends on the board revision. xwr_raw/elevation.py documents
    two AOP hardware revisions with RX1/RX2 swapped top-for-bottom, and its
    AOP_ELEVATION_SIGN_ES2 is reused here. It defaults to ES2.0. If a target
    at a known height comes out mirrored, the board is ES1.0: flip that
    constant, not this code.
  * lambda/2 vertical spacing is assumed (get_antenna_spacing() reports it
    for this platform). It is exposed as vertical_spacing_lambda.

The flip_aop_phase offset
-------------------------
dsp.reshape_frame applies flip_aop_phase for this platform: a 180 deg phase
inversion on RX0 and RX2, which is exactly row A. That leaves a constant pi
offset between the rows, which must be removed before converting phase to an
angle. row_phase_offset_deg defaults to 180.0 for this reason.

Measured on live hardware with near_mimo.cfg and the radar facing down at a
table, so the dominant return (the surface directly below) should sit near
boresight, at an elevation near 0. Top 0.1% of heatmap cells:

    row_phase_offset_deg =   0 : median +66.7 deg, p10..p90 = +29.1 .. +80.5
    row_phase_offset_deg = 180 : median  -3.9 deg, p10..p90 =  -7.2 ..  -0.9

180 puts the dominant return within 4 deg of boresight with a 6 deg spread.
0 puts it at +67 deg with a 51 deg spread. The wide spread at 0 pushes dphi
near +-pi, where arcsin(dphi/pi) has an unbounded derivative, so small phase
noise produces large angle swings (an offset of 0 gave readings near +-88
deg on live data). At 180 the estimate sits near dphi = 0, where arcsin is
well conditioned.

Not checked against ground truth. The above is a physical prior (a radar
facing down sees the surface below at boresight), not a measurement against
a target at a known height. It settles the 0-vs-pi question and fixes the
offset to within a few degrees, but it does not confirm the sign convention
(AOP_ELEVATION_SIGN_ES2, ES1.0 vs ES2.0 boards) or rule out a small residual
bias. Run nodes/elevation_calibration.py with a corner reflector at a known
height before trusting absolute elevations.

The coherence measure returned alongside does not discriminate on this
hardware. Measured live, per-chirp coherence is ~1.00 at every power level
(median 0.996 even in the bottom 50% of cells), because a static scene is
stable from chirp to chirp, so clutter phase repeats as faithfully as a
target's. Cell power is what correlates with phase stability here (row-to-row
concentration 0.988 in the top 0.1% of cells, 0.56-0.79 below), which is why
the pipeline gates on elevation_min_power_pct and leaves the coherence gate
off by default.
"""

import numpy as np

from xwr_raw.dsp import gen_steering_vec
from xwr_raw.elevation import AOP_ELEVATION_SIGN_ES2

# Virtual-channel indices for the two rows, in matching X order so that
# column j of one row is the vertical partner of column j of the other.
# row_b is exactly the azimuth ULA used everywhere else in this package
# (core/frame.py's aop_azimuth_indices default), which is what keeps the
# elevation grid aligned with the detection heatmap's azimuth axis.
AOP_ROW_B = (11, 9, 7, 5)   # TX2/TX1 x RX3/RX1
AOP_ROW_A = (10, 8, 6, 4)   # TX2/TX1 x RX2/RX0, same columns, other Z


def estimate_elevation_interferometric(radar_cube, angle_range=70, angle_res=1,
                                       vertical_spacing_lambda=0.5,
                                       row_a=AOP_ROW_A, row_b=AOP_ROW_B,
                                       sign=AOP_ELEVATION_SIGN_ES2,
                                       row_phase_offset_deg=180.0):
    """Elevation angle per (range, azimuth) cell, plus a per-cell confidence.

    Args:
        radar_cube: full virtual-channel cube (n_chirps, n_virtual, n_samples),
            i.e. process_frame_to_range_azimuth's return_full_cube output,
            not the 4-channel azimuth selection.
        angle_range, angle_res: must match the detection heatmap's azimuth
            grid, or the returned map will not line up with it cell for cell.
        vertical_spacing_lambda: row separation in wavelengths (lambda/2 on
            this platform).
        row_phase_offset_deg: constant phase to subtract from the row-to-row
            difference before converting to an angle. Defaults to 180, which
            removes the pi that flip_aop_phase leaves between the rows (see
            the module docstring for the live measurement behind that).

    Returns:
        elevation_deg (n_range, n_angle) float32
        coherence     (n_range, n_angle) float32 in [0, 1]
    """
    n_virtual = radar_cube.shape[1]
    needed = max(max(row_a), max(row_b)) + 1
    if n_virtual < needed:
        raise ValueError(
            f'Interferometric elevation needs {needed} virtual channels (both rows of '
            f'the AOP MIMO array) but this frame has {n_virtual}. That means a SINGLE-TX '
            f'chirp profile is running -- a single TX gives only one row, and one row '
            f'carries no elevation information at all. Use chirp_profile:=near_mimo, or '
            f"set elevation_mode to 'none'.")

    sub_a = radar_cube[:, list(row_a), :]
    sub_b = radar_cube[:, list(row_b), :]

    # Range FFT. Cast to complex128 explicitly: numpy 1.x upcasts complex64
    # on fft while numpy 2.x preserves it, and either may be installed.
    ra = np.fft.fft(sub_a.astype(np.complex128), axis=2)
    rb = np.fft.fft(sub_b.astype(np.complex128), axis=2)

    _, steering = gen_steering_vec(angle_range, angle_res, len(row_a))
    w = np.conjugate(steering).astype(np.complex128)     # (n_theta, n_ant)

    # Beamform both rows on the same grid: (n_chirps, n_theta, n_range).
    ya = np.einsum('ta,car->ctr', w, ra)
    yb = np.einsum('ta,car->ctr', w, rb)

    # Cross-correlate the rows and average over chirps. Averaging the complex
    # product (not the per-chirp angles) makes this an estimator rather than
    # a single noisy sample: chirps that agree on the phase add coherently
    # and chirps that disagree cancel, which is what makes |cross| a usable
    # confidence measure below.
    cross = np.mean(yb * np.conjugate(ya), axis=0)       # (n_theta, n_range)
    power = np.mean(np.abs(yb) * np.abs(ya), axis=0)

    dphi = np.angle(cross)
    if row_phase_offset_deg:
        # Re-wrap to (-pi, pi] after removing the offset, or angles near the
        # wrap point would fold to the wrong side.
        dphi = np.angle(np.exp(1j * (dphi - np.deg2rad(row_phase_offset_deg))))
    # dphi = 2*pi*(d/lambda)*sin(el)
    sin_el = dphi / (2.0 * np.pi * float(vertical_spacing_lambda))
    elevation_deg = sign * np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    with np.errstate(divide='ignore', invalid='ignore'):
        coherence = np.where(power > 0, np.abs(cross) / power, 0.0)

    # Transpose to (n_range, n_angle) to match the detection heatmap.
    return (np.ascontiguousarray(elevation_deg.T, dtype=np.float32),
            np.ascontiguousarray(np.clip(coherence.T, 0.0, 1.0), dtype=np.float32))
