# radar_pipelines

ROS 2 processing nodes for the raw ADC frames that the `xwr_raw_ros` driver in
this repository publishes from a TI IWR6843AOPEVM with a DCA1000EVM. Hardware
and driver setup are described in [`../xwr_raw_ros/README.md`](../xwr_raw_ros/README.md).

The package has two pipelines over one shared DSP layer:

- **heatmap**: range-azimuth (`hm_hori`) and range-elevation (`hm_vert`)
  heatmaps in the dual-view layout of the MMVR dataset, the input format of
  the REXO model.
- **pointcloud**: a 3D point cloud from CFAR detection on a Capon
  range-azimuth heatmap.

The processing code in `radar_pipelines/core` and `radar_pipelines/pipelines`
does not import rclpy, so it can be run from a bag replay, a notebook or the
tests. The nodes in `radar_pipelines/nodes` read their parameters in every
callback, so `ros2 param set` on a running node takes effect on the next frame.

## Nodes

The three processing nodes (`heatmap_node.py`, `pointcloud_node.py` and
`heatmap_viz.py`) subscribe to `radar_topic` (default `radar1/radar_data`,
type `xwr_raw_ros/RadarFrameFull`) and take a `chirp_profile` parameter that
selects the calibration sidecar (see [Calibration](#calibration)). An empty
`chirp_profile` runs uncalibrated with a warning, and an unknown name is an
error. On the first frame each node checks the frame's `range_max` against
the sidecar and logs an error if they disagree.

### `heatmap_node.py` (node `radar_heatmap_rexo`)

| Topic | Type | Content |
|---|---|---|
| `radar_rexo/hm_hori` | `sensor_msgs/Image` 32FC1 | range-azimuth heatmap, linear magnitude |
| `radar_rexo/hm_vert` | `sensor_msgs/Image` 32FC1 | range-elevation heatmap, linear magnitude |
| `radar_rexo/dual_view` | `std_msgs/Float32MultiArray` | both views stacked, layout `[view, range, angle]` |
| `radar_rexo/hm_hori_viz`, `radar_rexo/hm_vert_viz` | `sensor_msgs/Image` bgr8 | colourized views, if `enable_visualization` |
| `radar_rexo/debug_virtual_channel_power` | `std_msgs/Float32MultiArray` | mean power per virtual channel, every `channel_power_log_period` frames, if `publish_channel_power_debug` |

| Parameter | Default | Meaning |
|---|---|---|
| `azimuth_fov`, `elevation_fov` | 60.0, 45.0 | half field of view in degrees |
| `n_range_bins`, `n_azimuth_bins`, `n_elevation_bins` | 256, 128, 128 | output size |
| `doppler_window_bins` | 3 | Doppler bins summed on each side of zero Doppler |
| `window_sidelobe_db` | 65.0 | Chebyshev taper sidelobe level (at least 45) |
| `output_cartesian` | true | Cartesian grid (row 0 farthest) or polar (row 0 nearest) |
| `conditioning_enabled` | true | frame-to-frame arc removal and smoothing |
| `min_range_m` | 0.0 | zero rows nearer than this, in calibrated metres (0 = off) |
| `background_enabled` | false | subtract a captured static background |
| `background_capture` | false | set true to capture a background (see below) |
| `background_n_frames` | 30 | frames averaged per capture |
| `background_dir` | `~/.ros/radar_pipelines/background/rexo` | where baselines are saved and loaded |
| `viz_clip_low_pct`, `viz_clip_high_pct` | 2.0, 98.0 | percentile clip for the `_viz` topics only |

### `pointcloud_node.py` (node `radar_pointcloud`)

| Topic | Type | Content |
|---|---|---|
| `radar_pointcloud` | `sensor_msgs/PointCloud2` | fields x, y, z, intensity (float32) |
| `radar_pointcloud/markers` | `visualization_msgs/MarkerArray` | a sphere and a label per cluster, if `publish_cluster_markers` |
| `radar_pointcloud/debug_heatmap_pre_cfar` | `sensor_msgs/Image` 32FC1 | the range-azimuth array CFAR sees (natural-log magnitude), if `publish_debug_heatmap` |

Points are published in a camera optical convention (x right, y down,
z forward) in `frame_id` (default `radar_horizontal_link`). Nothing in this
package publishes a transform for that frame.

| Parameter | Default | Meaning |
|---|---|---|
| `beamformer` | `capon` | `capon` or `bartlett` |
| `capon_angle_range`, `capon_angle_resolution` | 70, 1 | azimuth scan of plus or minus 70 deg in 1 deg steps |
| `cfar_method` | `ca` | `ca`, `os`, `go`, `so` or `pct` (see `core/cfar.py`) |
| `cfar_threshold` | 2.5 | additive threshold in nats (the heatmap is natural-log magnitude) |
| `cfar_guard_len`, `cfar_noise_len` | 2, 4 | CFAR window |
| `cfar_os_rank_frac` | 0.75 | rank for `os` |
| `cfar_threshold_pct` | 99.5 | percentile for `pct` |
| `min_range_idx` | 3 | drop detections in the first range bins |
| `near_field_exclusion_m` | 0.0 | replace rows nearer than this before CFAR (0 = off) |
| `background_enabled` | true | subtract a captured static background |
| `background_capture`, `background_n_frames`, `background_dir` | false, 30, `~/.ros/radar_pipelines/background/pointcloud` | as for the heatmap node |
| `elevation_mode` | `none` | `none` (z = 0), `interferometric` or `per_range` (see [Elevation](#elevation-exploratory)) |
| `cluster_tolerance_m`, `cluster_min_points`, `cluster_max_points` | 0.15, 2, 500 | marker clustering |

The point-cloud pipeline needs a 3-TX TDM-MIMO profile (`near_mimo` or
`test`). With the single-TX `near` profile the node logs an error on every
frame and publishes nothing.

### `heatmap_viz.py` (node `radar_heatmap_viz`)

A debugging view. It runs the heatmap pipeline in polar mode and publishes
four bgr8 images with metre and degree ticks: `radar_viz/range_azimuth`,
`radar_viz/range_elevation`, `radar_viz/range_azimuth_cartesian` and
`radar_viz/range_elevation_cartesian`. The polar angle axis covers all FFT
bins, which are uniform in sin(theta) over plus or minus 90 deg, so its ticks
are not evenly spaced in degrees. Parameters: the field-of-view and bin-count
parameters of the heatmap node, `display_range_max` (0 = full range),
`viz_clip_low_pct`, `viz_clip_high_pct`, and `snapshot_dir` with
`snapshot_period_s` to write PNG snapshots. It applies the range calibration
but no background subtraction or near-field exclusion.

### `view_heatmaps.py`

Shows `radar_rexo/hm_hori_viz` and `radar_rexo/hm_vert_viz` side by side in
an OpenCV window. Needs a display. Press `q` to quit.

### `capture_background.py`

Triggers a background capture on a running node and waits for it:

```bash
ros2 run radar_pipelines capture_background.py --node /radar_pointcloud
ros2 run radar_pipelines capture_background.py --node /radar_heatmap_rexo --timeout 60
```

Clear the scene first (mount, table and backdrop only). The node averages
`background_n_frames` frames at the stage where it later subtracts them,
saves the baseline to `background_dir` and resets `background_capture`.
Put the objects back and set `background_enabled:=true` if it is not already.
A saved baseline is loaded at start-up when `background_enabled` is true.

### `elevation_calibration.py`

A command-line tool for checking interferometric elevation against a target
at a known height (see [Elevation](#elevation-exploratory)):

```bash
ros2 run radar_pipelines elevation_calibration.py --true-elevation-deg -12.5
```

## Launch

`launch/radar_pipelines.launch.py` starts the driver (`xwr_raw_ros`
`radar.launch.py`) with `configs/6843aop/<chirp_profile>.cfg` and the chosen
pipelines with the matching calibration sidecar:

```bash
ros2 launch radar_pipelines radar_pipelines.launch.py
ros2 launch radar_pipelines radar_pipelines.launch.py pipeline:=both cfar_method:=os
ros2 launch radar_pipelines radar_pipelines.launch.py start_driver:=false   # bag replay or separate driver
```

| Argument | Default | Meaning |
|---|---|---|
| `chirp_profile` | `near_mimo` | selects both the driver's .cfg and the calibration sidecar |
| `pipeline` | `heatmap` | `heatmap`, `pointcloud` or `both` |
| `start_driver` | `true` | set false to attach to a running driver or a bag replay |
| `radar_topic` | `radar1/radar_data` | input topic |
| `azimuth_fov`, `elevation_fov`, `doppler_window_bins` | 60.0, 45.0, 3 | heatmap pipeline |
| `cfar_method`, `cfar_threshold`, `capon_angle_range` | `ca`, 2.5, 70 | point-cloud pipeline |

If you start the driver another way, set each node's `chirp_profile` to the
profile the driver runs.

## Calibration

Each chirp profile shipped with the driver has a sidecar in `config/calib/`:
`near.yaml`, `near_mimo.yaml` and `test.yaml`. Nodes load it by name.
`RADAR_PIPELINES_CALIB_DIR` adds a directory that is searched first.

- `range_scale = true_range_max_m / raw_range_max_m`. `raw_range_max_m` is
  the `range_max` the driver computes from the profileCfg line
  (`sample_rate * c / (2 * slope)`). `true_range_max_m` is the "Maximum
  unambiguous Range" that TI's mmWave Visualizer writes in the .cfg header,
  which TI derates below that value. The ratio (0.797 for `near` and
  `near_mimo`, 0.794 for `test`) is TI's derating, not a fit to measured
  distances. `range_offset` is 0.
- The scale is applied to the whole range axis. The driver spaces range bins
  at `range_max / n_samples`, which equals c / (2B), the spacing the chirp
  parameters give. After scaling, each bin covers about 0.8 of that.
- The azimuth calibration (`azimuth_scale` 1, `azimuth_offset_deg` 0) is the
  identity, so azimuth is not calibrated.
- The values are kept because they produced the recorded results.

`near` and `near_mimo` share a profileCfg line, so their sidecars hold the
same values, and the first-frame `range_max` check cannot tell them apart.

## REXO-format heatmaps

The heatmap pipeline follows the MMVR dataset's dual-view layout (256 x 128
range-azimuth and range-elevation maps, linear magnitude, Cartesian grid) but
is not an exact copy of MMVR's preprocessing:

- Doppler integration sums a window of plus or minus 3 bins around zero
  Doppler (`doppler_window_bins`), not the full Doppler axis.
- A Dolph-Chebyshev taper is applied across the antenna channels before the
  angle FFT.
- Azimuth uses 4 virtual channels (11, 9, 7, 5) and elevation uses 2, each
  the average of two virtual channels at the same height (4 with 8, and 5
  with 9).
- Range is interpolated to 256 bins. `near_mimo` has 80 range bins.

## Elevation (exploratory)

`elevation_mode:=interferometric` (`core/elevation2d.py`) estimates one
elevation per heatmap cell from the phase difference between the AOP's two
virtual-array rows. It is exploratory: it has never been checked against
ground truth, and it is off by default (`elevation_mode` is `none`, so every
point has z = 0). Its 180 deg row phase offset comes from a physical prior
(a radar facing down sees the surface below at boresight), not from a target
at a known height. `elevation_calibration.py` is the tool for that check.
`per_range` uses `xwr_raw/elevation.py`'s per-range-bin search and is kept for
comparison.

## Tests

```bash
cd radar_pipelines
PYTHONPATH=../xwr_raw_ros/src python3 -m pytest -q test
```

The test builds synthetic frames and runs both pipelines, every CFAR method,
background capture and the interferometric elevation on them. No radar is
needed.

## Licence

Apache-2.0, as for the rest of this repository.
