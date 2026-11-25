#!/usr/bin/env python3
"""The dissertation's Chapter 2 radar figures for Experiments 1 and 2, from their recorded results.

    python3 figures/make_radar_figures.py        # writes figures/out/exp1_distances.pdf and exp2_merge.pdf

Experiment 1 (data/exp1_results.json, recorded 28 Nov 2025 with configs/6843aop/test.cfg): the radar's
    distance inside the bottle's outline, per frame. Distances are scaled by 0.79402, the ratio of test.cfg's
    header maximum range (7.17 m) to the range the driver computes from its chirp parameters (9.03 m). The
    program that wrote results.json no longer exists, so this experiment cannot be rerun from the radar data.
Experiment 2 (data/exp2_heatmap.npy, 17 Aug 2026): range x azimuth, 80 x 181 bins at 1 degree, holding ln(power)
    from a dsp.compute_range_azimuth_* beamformer, so dB = 10/ln(10) * value. The program that wrote it was not
    kept: it was the Capon beamformer. Range is scaled by near_mimo's 0.79688
    (2.88 / 3.6141 m, the same kind of header ratio). The camera-predicted bearings of the two objects
    (-10 and +15 degrees, 25 cm apart) were measured at the time.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out"
EXP1 = DATA / "exp1_results.json"
INK, GREY, BLUE, RED, ORANGE = "#222222", "#8a8a8a", "#1f5fb4", "#b2182b", "#e07b28"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8.5})


def exp1_distances():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    rows = [r for r in json.load(open(EXP1)) if r.get("status") == "ok" and r.get("n_points_in_mask", 0) > 0]
    d = np.array([r["depth_m"] for r in rows]) * 0.79402
    cloth = 0.389
    ax.axvspan(cloth - 0.02, cloth + 0.02, color="#f2c9a6", alpha=0.6, lw=0)
    ax.hist(d, bins=np.arange(0.30, 0.48, 0.005), color=BLUE, alpha=0.9)
    ax.axvline(cloth, color=ORANGE, lw=1.6)
    ax.text(cloth + 0.023, ax.get_ylim()[1] * 0.88, "the cloth behind\nthe bottle (camera)\n0.389 m", color=ORANGE,
            fontsize=7.5, va="top")
    ax.text(0.302, ax.get_ylim()[1] * 0.88, f"radar, {len(d)} frames\nmedian {np.median(d):.3f} m", color=BLUE,
            fontsize=7.5, va="top")
    ax.text(cloth + 0.023, ax.get_ylim()[1] * 0.42, "shaded: one 4 cm\ndistance step", color=GREY, fontsize=7,
            va="top")
    ax.set_xlim(0.30, 0.48)
    ax.set_xlabel("distance from the sensor (m)")
    ax.set_ylabel("frames")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "exp1_distances.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return len(d), float(np.median(d)), float(np.std(d))


def exp2():
    hm = np.load(DATA / "exp2_heatmap.npy").astype(float)             # 80 range x 181 azimuth
    r = np.arange(hm.shape[0]) * 0.045176 * 0.79688               # calibrated metres
    th = np.radians(np.linspace(-90, 90, hm.shape[1]))
    db = hm * 10 / np.log(10)                                    # ln(power) -> dB
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.3], wspace=0.45)
    # (a) the scene
    ax = fig.add_subplot(gs[0])
    im = np.asarray(Image.open(DATA / "exp2_rgb.png").convert("RGB"))
    h, w = im.shape[:2]
    ax.imshow(im[int(0.30 * h):int(0.95 * h), int(0.28 * w):int(0.82 * w)])
    for x, lab in ((0.20, "flask"), (0.62, "bottle")):
        ax.text(x, 0.93, lab, transform=ax.transAxes, color="white", fontsize=7, ha="center", va="top")
    ax.set_title("(a) the scene", loc="left")
    ax.axis("off")
    # (b) top-down radar map near the objects
    ax = fig.add_subplot(gs[1])
    keep = (r > 0.25) & (r < 0.95)
    R, T = np.meshgrid(r[keep], th)
    X, Y = R * np.sin(T), R * np.cos(T)
    sub = db[keep].T
    sub = sub - sub.max()
    ax.pcolormesh(X, Y, sub, cmap="magma", vmin=-20, vmax=0, shading="nearest")
    for b, lab in ((-10, "flask"), (15, "bottle")):
        x, y = 0.576 * np.sin(np.radians(b)), 0.576 * np.cos(np.radians(b))
        ax.add_patch(Circle((x, y), 0.035, fill=False, ec="white", lw=1.2))
        ax.text(x, y + 0.06, lab, color="white", ha="center", fontsize=7)
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(0.25, 0.95)
    ax.set_aspect("equal")
    ax.set_xlabel("sideways (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title("(b) the radar's map", loc="left")
    ax.set_xticks([-0.4, 0.0, 0.4])
    # (c) echo strength against direction at the objects' distance
    ax = fig.add_subplot(gs[2])
    deg = np.linspace(-90, 90, hm.shape[1])
    cut = hm[15:18].mean(0) * 10 / np.log(10)
    cut -= cut.max()
    ax.plot(deg, cut, color=BLUE, lw=1.5)
    for b, lab, ha, dx in ((-10, "flask", "right", -2), (15, "bottle", "left", 2)):
        ax.axvline(b, color=RED, ls="--", lw=1)
        ax.text(b + dx, 1.0, lab, color=RED, ha=ha, fontsize=7)
    ax.annotate("", xy=(-10, -15.0), xytext=(15, -15.0), arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.8))
    ax.text(19, -15.0, "25°: about\none beam width", color=GREY, fontsize=7, va="center")
    ax.set_xlim(-90, 90)
    ax.set_ylim(-16.5, 2.5)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("direction (degrees)")
    ax.set_ylabel("echo (dB)")
    ax.set_title("(c) echo at 0.58 m, by direction", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "exp2_merge.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return float(cut.min())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print("exp1:", exp1_distances())
    print("exp2 cut min:", exp2())
