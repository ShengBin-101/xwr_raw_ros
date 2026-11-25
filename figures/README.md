# Chapter 2 figures

The scripts and data behind three radar figures in Chapter 2, "Radar-Camera Fusion: A Feasibility
Study", of Chan Sheng Bin's dissertation "Robotic Grasping of Transparent Objects: From Radar-Camera
Fusion to Learned Depth Completion" (National University of Singapore, 2026). They need the repository's `xwr_raw` package (found from here, no build needed), NumPy,
SciPy, numba, Matplotlib and Pillow.

```bash
python3 figures/make_radar_figures.py     # out/exp1_distances.pdf, out/exp2_merge.pdf
python3 figures/make_person_figure.py     # out/interim_person_views.pdf
```

| Figure | Script | Data | Reproduces |
|---|---|---|---|
| Experiment 1: the radar's distance to a bottle | `make_radar_figures.py` | `data/exp1_results.json`: the per-frame distances (97 frames used) | from the stored distances: median 0.380 m |
| Experiment 2: two objects 25 cm apart | `make_radar_figures.py` | `data/exp2_heatmap.npy` (80 x 181, ln power), `data/exp2_rgb.png` | from the stored heatmap |
| A person about 1 m away, from above and from the side | `make_person_figure.py` | `data/person_radar_frames.db3`: three raw frames | from the raw radar frames, through `xwr_raw` |

Rendered at 200 dpi, all three match the dissertation's figures pixel for pixel. The person figure's
panel (a) is a camera photo of the person and is not published: without `--photo` that panel is
left blank, and panels (b) and (c) still match pixel for pixel.

What cannot be rerun:

- **Experiment 1.** The program that turned the recording into `exp1_results.json` was deleted
  during the project. The figure is drawn from its stored output. The profile was
  `xwr_raw_ros/configs/6843aop/test.cfg` (4.0 cm range bins).
- **Experiment 2.** The program that wrote `exp2_heatmap.npy` was not kept. Its values are
  ln(power) from `xwr_raw.dsp`'s Capon beamformer.

Distances are scaled by the profile's header ratio (0.79402 for test.cfg, 0.79688 for
near_mimo.cfg), as explained in the repository README.

`bag_heatmaps.py` reads `xwr_raw_ros/RadarFrameFull` messages from a rosbag2 file (`.db3`) and
writes range-azimuth heatmaps: `python3 figures/bag_heatmaps.py BAG.db3 OUT.npz`.
