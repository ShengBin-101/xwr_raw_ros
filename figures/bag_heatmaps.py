#!/usr/bin/env python3
"""Range-azimuth heatmaps from a rosbag2 recording of xwr_raw_ros/RadarFrameFull messages (topic /radar1/radar_data).

For the IWR6843AOP: reshape each frame (flip_aop_phase), stack three frames, keep the four azimuth channels
(virtual 11, 9, 7, 5), Capon at 1 degree over +-90 degrees. dsp.compute_range_azimuth_capon returns ln(power),
so dB = 10/ln(10) * value. make_person_figure.py uses this file's parse_frame() to read raw frames.

    python3 figures/bag_heatmaps.py BAG.db3 OUT.npz
"""
import sqlite3
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xwr_raw_ros" / "src"))
from xwr_raw import dsp  # noqa: E402


def parse_frame(blob):
    """CDR-decode one xwr_raw_ros/RadarFrameFull message."""
    o = 4

    def align(n):
        nonlocal o
        o = 4 + ((o - 4 + n - 1) // n) * n

    def u32():
        nonlocal o
        align(4); v = struct.unpack_from("<I", blob, o)[0]; o += 4; return v

    def f32():
        nonlocal o
        align(4); v = struct.unpack_from("<f", blob, o)[0]; o += 4; return v

    def string():
        nonlocal o
        n = u32(); s = blob[o:o + n - 1].decode(); o += n; return s

    def seq(fmt, size):
        nonlocal o
        n = u32(); align(size); v = np.frombuffer(blob, dtype=fmt, count=n, offset=o).copy(); o += n * size; return v

    sec = u32(); nsec = u32(); string()                       # header
    platform = string()
    adc = blob[o]; o += 1
    range_bias = f32()
    rx_phase_bias = seq("<f4", 4)
    chirp_time, chirp_slope, frame_time, vmax, vres, fs, range_max, range_res = (f32() for _ in range(8))
    rx = seq("u1", 1); tx = seq("u1", 1); shape = seq("<u4", 4); data = seq("<i2", 2)
    return SimpleNamespace(stamp=sec + nsec * 1e-9, platform=platform, adc_output_fmt=adc, range_bias=range_bias,
                           rx_phase_bias=list(rx_phase_bias), range_max=range_max, range_res=range_res,
                           rx=list(rx), tx=list(tx), shape=list(shape), data=data)


def heatmaps(bag):
    con = sqlite3.connect(f"file:{bag}?mode=ro", uri=True)
    c = con.cursor()
    tid = c.execute("select id from topics where name='/radar1/radar_data'").fetchone()[0]
    frames = [parse_frame(bytes(r[0])) for r in c.execute(
        "select data from messages where topic_id=? order by timestamp", (tid,))]
    cubes = [dsp.reshape_frame(f, flip_aop_phase=True) for f in frames]
    out = []
    for k in range(2, len(cubes)):
        cube = np.concatenate(cubes[k - 2:k + 1], axis=0)
        cube = np.stack([cube[:, 11, :], cube[:, 9, :], cube[:, 7, :], cube[:, 5, :]], axis=1)
        out.append(dsp.compute_range_azimuth_capon(cube, 1, 90))
    f0 = frames[0]
    return np.array(out), f0, len(frames)


if __name__ == "__main__":
    hm, f0, n = heatmaps(sys.argv[1])
    np.savez_compressed(sys.argv[2], ln_power=hm, range_max=f0.range_max, range_res=f0.range_res,
                        range_bias=f0.range_bias, platform=f0.platform, shape=np.array(f0.shape), tx=np.array(f0.tx))
    print(f"{n} frames -> {hm.shape} heatmaps; platform {f0.platform}, shape {f0.shape}, tx {f0.tx}, "
          f"range_max {f0.range_max:.3f} m, range_res {f0.range_res:.4f} m")
