#!/usr/bin/env python3

"""Static-clutter suppression by explicit background capture.

Both pipelines use this one implementation. Each node captures its
background in the node, triggered by a ROS parameter, by averaging N frames
at the same processing stage where the subtraction is later applied, so the
baseline and the live heatmap line up bin for bin. A baseline recorded from
a published heatmap topic would not: those topics are published after
condition_heatmap(), while the subtraction happens before it.

Why explicit capture and not an adaptive filter
-----------------------------------------------
An adaptive or running background has to guess what counts as static
clutter and what counts as a slow-moving real target. Here the targets are
static: a bottle on a table has the same zero-Doppler signature as the table
itself, so a Doppler-based or adaptive-decay filter cannot tell them apart.
The operator's remove-then-place step supplies the information the signal
does not contain.

Linear vs log domain
--------------------
The two pipelines pass different units:

  'subtract_clip'  REXO heatmaps are linear magnitude, where 0.0 means "no
                   signal", so a negative result after subtraction has no
                   meaning and is clipped to 0.
  'subtract'       The Capon range-azimuth heatmap is natural-log magnitude,
                   where 0.0 is log(1), a value unrelated to the noise
                   floor. Clipping there would carve an artificial floor
                   that CFAR's guard and training window could read as a
                   clutter edge, so the difference is left signed.
"""

import os

import numpy as np


SUBTRACT = 'subtract'
SUBTRACT_CLIP = 'subtract_clip'


def _warn(logger, msg, throttle=None):
    if logger is None:
        return
    try:
        if throttle is not None:
            logger.warn(msg, throttle_duration_sec=throttle)
        else:
            logger.warn(msg)
    except TypeError:  # a plain (non-rclpy) logger without throttle support
        logger.warn(msg)


class BackgroundBuffer:
    """Captures and applies one static-background baseline for one heatmap.

    A node holds one of these per view. The REXO pipeline holds two, one for
    hm_hori and one for hm_vert, and `name` keeps their log lines and .npy
    filenames apart.
    """

    def __init__(self, name, mode=SUBTRACT):
        if mode not in (SUBTRACT, SUBTRACT_CLIP):
            raise ValueError(f"mode must be '{SUBTRACT}' or '{SUBTRACT_CLIP}', got '{mode}'")
        self.name = name
        self.mode = mode
        self.background = None
        self._buffer = []

    @property
    def has_background(self):
        return self.background is not None

    def handle_capture(self, heatmap, capture_active, n_frames, logger=None):
        """Accumulate frames while capture_active, and average once n_frames are collected.

        Returns one of 'idle' | 'started' | 'collecting' | 'done' | 'aborted'.
        The caller resets its own trigger parameter to false on 'done' (and
        may treat 'aborted' as already reset). This class holds no reference
        to a ROS node.
        """
        if not capture_active:
            if self._buffer:
                # The trigger was reset before the target count was
                # reached. Discard the partial capture instead of averaging
                # fewer frames than the operator asked for.
                _warn(logger,
                      f'[{self.name}] Background capture aborted with only '
                      f'{len(self._buffer)} frames collected.')
                self._buffer = []
                return 'aborted'
            return 'idle'

        self._buffer.append(np.array(heatmap, copy=True))
        n_target = max(1, int(n_frames))
        n_have = len(self._buffer)

        if n_have >= n_target:
            self.background = np.mean(np.stack(self._buffer[:n_target]), axis=0)
            self._buffer = []
            if logger is not None:
                logger.info(
                    f'[{self.name}] Background captured from {n_target} frames '
                    f'(mean={self.background.mean():.3e}, max={self.background.max():.3e}). '
                    f'Suppression now active if enabled.')
            return 'done'

        if n_have == 1 and logger is not None:
            logger.info(
                f'[{self.name}] Background capture started, collecting {n_target} frames...')
        return 'collecting' if n_have > 1 else 'started'

    def apply(self, heatmap, enabled=True, logger=None):
        """Subtract the captured baseline. No-op if disabled or never captured."""
        if not enabled or self.background is None:
            return heatmap
        if self.background.shape != heatmap.shape:
            _warn(logger,
                  f'[{self.name}] background shape {self.background.shape} != heatmap shape '
                  f'{heatmap.shape} (bin counts changed since capture?) -- skipping '
                  f'subtraction this frame.', throttle=5.0)
            return heatmap
        out = heatmap - self.background
        if self.mode == SUBTRACT_CLIP:
            out = np.clip(out, 0.0, None)
        return out

    # -- persistence ------------------------------------------------------
    # A captured baseline costs the operator several minutes (clear the
    # scene, wait, put everything back), so it is saved to disk and survives
    # a node restart.

    def path(self, directory):
        return os.path.join(directory, f'background_{self.name}.npy')

    def save(self, directory, logger=None):
        if self.background is None:
            _warn(logger, f'[{self.name}] No background captured yet -- nothing to save.')
            return None
        os.makedirs(directory, exist_ok=True)
        path = self.path(directory)
        np.save(path, self.background.astype(np.float32))
        if logger is not None:
            logger.info(f'[{self.name}] Saved background baseline to {path}')
        return path

    def load(self, directory, logger=None):
        path = self.path(directory)
        if not os.path.exists(path):
            _warn(logger,
                  f'[{self.name}] No saved background at {path} -- capture one by setting '
                  f'background_capture:=true with the scene physically cleared. '
                  f'Running without suppression until then.')
            return False
        self.background = np.load(path).astype(np.float32)
        if logger is not None:
            logger.info(
                f'[{self.name}] Loaded background baseline from {path} '
                f'(max={self.background.max():.3e})')
        return True
