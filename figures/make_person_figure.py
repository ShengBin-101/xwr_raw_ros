#!/usr/bin/env python3
"""The dissertation's Figure 2.2: a person about 1 m away, seen by the radar from above and from the side.

    python3 figures/make_person_figure.py                  # writes figures/out/interim_person_views.pdf
    python3 figures/make_person_figure.py --photo P.png    # with panel (a), the camera photo, if you have it

Radar frames: data/person_radar_frames.db3, the three frames of a 30 Mar 2026 recording (test.cfg: three
transmitters, 224 samples) nearest the camera frame of the photo in panel (a). The photo shows a person and is not
published, so without --photo panel (a) is left blank. Views from the REXO-style
preprocessing (rexo_processing.compute_range_azimuth_rexo / compute_range_elevation_rexo, polar output: FFT
beamforming over the 4 azimuth / 2 elevation elements, still echoes only), range scaled by 0.79402, test.cfg's
header ratio (see make_radar_figures.py).

Angle bins: the processing fft-shifts the angle axis and then reverses it, so bin j sits at
sin(theta) = 2 (j + 1 - N/2) / N (rexo_processing.fft_angle_bins). Only the antenna's +-60 deg field of view is drawn (REXO's angle_range): beyond it the bins
approach end-fire, where the four-element array has no useful resolution and the sine mapping stretches.
"""
import argparse
import importlib.util
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
BAG = HERE / "data" / "person_radar_frames.db3"
OUT = HERE / "out"
T_MATCH = 1774854606639526963                    # the camera frame of the photo in panel (a)
PHOTO_SHAPE = (534, 947)                         # the photo's size after the black bars are cropped
spec = importlib.util.spec_from_file_location("bh", HERE / "bag_heatmaps.py")
bh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bh)
import xwr_raw.rexo_processing as rp  # noqa: E402  (path set by bag_heatmaps)

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8.5})


def views():
    con = sqlite3.connect(f"file:{BAG}?mode=ro", uri=True)
    c = con.cursor()
    tid = c.execute("select id from topics where name='/radar1/radar_data'").fetchone()[0]
    rows = c.execute("select timestamp, data from messages where topic_id=? order by abs(timestamp - ?) limit 3",
                     (tid, T_MATCH)).fetchall()
    frames = [bh.parse_frame(bytes(d)) for _, d in sorted(rows)]
    cube = np.concatenate([bh.dsp.reshape_frame(f, flip_aop_phase=True) for f in frames], axis=0)
    f0 = frames[0]
    kw = dict(output_cartesian=False, n_range_bins=int(f0.shape[2]), range_max=f0.range_max,
              range_resolution_m=f0.range_res)
    az = rp.compute_range_azimuth_rexo(cube, platform="xWR68xx_AOP", **kw)
    el = rp.compute_range_elevation_rexo(cube, platform="xWR68xx_AOP", **kw)
    r = np.arange(az.shape[0]) * f0.range_res * 0.79402
    return az, el, r


FOV = np.radians(60.0)                           # the antenna's field of view, +-60 deg


def to_angles(n):
    s = 2 * (np.arange(n) + 1 - n / 2) / n     # fftshift, then the reversal in rexo_processing
    return np.arcsin(np.clip(s, -1, 1))


def draw(photo=None):
    az, el, r = views()
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    if photo:
        im = np.asarray(Image.open(photo).convert("RGB"))
        rows = np.where(im.mean(axis=(1, 2)) > 25)[0]
        ax.imshow(im[rows.min():rows.max() + 1])
    else:                                       # a blank panel the size of the cropped photo, so the layout holds
        ax.imshow(np.full(PHOTO_SHAPE + (3,), 236, dtype=np.uint8))
        ax.text(0.5, 0.5, "camera photo\nnot published", transform=ax.transAxes, ha="center", va="center",
                color="#8a8a8a")
    ax.set_title("(a) radar detections on the person", loc="left")
    ax.axis("off")
    keep = (r > 0.2) & (r < 3.0)
    for k, (hm, title, xl, yl) in enumerate(((az, "(b) from above", "sideways (m)", "forward (m)"),
                                             (el, "(c) from the side", "forward (m)", "up (m)"))):
        ax = fig.add_subplot(gs[k + 1])
        th_all = to_angles(hm.shape[1])
        inside = np.abs(th_all) <= FOV
        th = th_all[inside]
        db = 20 * np.log10(np.maximum(hm[keep][:, inside], 1e-9))
        db -= db.max()
        rk = r[keep]                                # cell edges in polar space, so the fan has straight sides
        r_e = np.concatenate([[rk[0] - (rk[1] - rk[0]) / 2], (rk[:-1] + rk[1:]) / 2, [rk[-1] + (rk[-1] - rk[-2]) / 2]])
        t_e = np.concatenate([[th[0] - (th[1] - th[0]) / 2], (th[:-1] + th[1:]) / 2, [th[-1] + (th[-1] - th[-2]) / 2]])
        R, T = np.meshgrid(r_e, t_e)
        if k == 0:
            X, Y = R * np.sin(T), R * np.cos(T)
        else:
            X, Y = R * np.cos(T), R * np.sin(T)
        ax.pcolormesh(X, Y, db.T, cmap="magma", vmin=-25, vmax=0, shading="flat")
        ax.set_aspect("equal")
        ax.set_facecolor("#ececec")                 # outside the field of view or the 0.2-3 m range: no data
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
        i = np.unravel_index(np.argmax(db), db.shape)
        rr, tt = r[keep][i[0]], th[i[1]]
        if k == 0:
            px, py = rr * np.sin(tt), rr * np.cos(tt)
            label, off = f"person, {rr:.1f} m", (0.30, 0.66)
        else:                                   # two elements: the arc spans every elevation, so no height
            px, py = rr, 0.0
            label, off = f"person, {rr:.1f} m:\nno height detail", (0.30, 0.05)
        ax.annotate(label, (px, py), xytext=off, textcoords="axes fraction", color="white", fontsize=7, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
        if k == 0:
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(0, 3.05)
        else:
            ax.set_xlim(0, 3.05)
            ax.set_ylim(-1.6, 1.6)
        print(f"view {k}: strongest echo {rr:.2f} m at {np.degrees(tt):+.0f} deg")
    fig.savefig(OUT / "interim_person_views.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw the person figure (Figure 2.2).")
    ap.add_argument("--photo", help="the camera photo with the radar detections, for panel (a)")
    OUT.mkdir(exist_ok=True)
    draw(ap.parse_args().photo)
