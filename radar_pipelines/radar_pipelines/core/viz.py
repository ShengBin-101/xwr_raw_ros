#!/usr/bin/env python3

"""Display-contrast helpers. Rendering only: never touches published data.

Plain min/max normalization is dominated by one or two extreme bins: the
near-field TX/RX coupling null, or a single very strong broadband clutter
return. Everything else then compresses into one end of the colour scale.
Clipping to percentiles first ignores those extremes, so the remaining cells
spread over the full colour scale. The same narrow usable dynamic range is
what drove `pct` in core/cfar.py, there as a detection threshold rather than
a display stretch.
"""

import numpy as np

from xwr_raw.rexo_processing import normalize_for_rexo


def clip_and_normalize(vals, plow=2.0, phigh=98.0, stat_mask=None, target_range=(0, 255)):
    """Percentile-clipped contrast stretch to uint8.

    stat_mask restricts which cells the percentile thresholds are computed
    over. The clip still applies everywhere. The Cartesian views need it:
    their zero-padded out-of-FoV corners would otherwise fill the
    percentiles with exact zeros and flatten the stretch.

    Falls back to plain min/max when the clip window collapses (a uniform or
    empty frame), instead of dividing by ~0.
    """
    src = vals[stat_mask] if stat_mask is not None and np.any(stat_mask) else vals
    finite = src[np.isfinite(src)]
    if finite.size == 0:
        return normalize_for_rexo(vals, method='minmax', target_range=target_range)
    lo, hi = np.percentile(finite, [plow, phigh])
    if hi - lo < 1e-10:
        return normalize_for_rexo(vals, method='minmax', target_range=target_range)
    return normalize_for_rexo(np.clip(vals, lo, hi), method='minmax',
                              target_range=target_range)
