# xwr_raw_ros for ROS 2 and the IWR6843AOP

Raw ADC capture from a TI IWR6843AOPEVM through a DCA1000EVM, in ROS 2 Humble, with the
processing and figures of a small radar study. This is the `ros2` branch of a fork of
[ConnectedSystemsLab/xwr_raw_ros](https://github.com/ConnectedSystemsLab/xwr_raw_ros): upstream's
ROS 1 driver, ported to ROS 2. Upstream's code is on the branch `main`, unchanged.

The radar study was the first part of a final-year project at the National University of Singapore
(2026). It found that the IWR6843AOP's small antenna array cannot separate neighbouring objects on
a table, and the project moved to learned depth completion with an RGB-D camera:
[transparent-grasp-stack](https://github.com/ShengBin-101/transparent-grasp-stack).

| Folder | What it is |
|---|---|
| [`xwr_raw_ros/`](xwr_raw_ros/README.md) | the driver: ROS 2 node, the `xwr_raw` Python package (capture and signal processing), messages, chirp profiles, udev rule |
| [`radar_pipelines/`](radar_pipelines/README.md) | processing nodes: point clouds (CFAR) and range-azimuth / range-elevation heatmaps, with viewers |
| [`figures/`](figures/README.md) | the scripts and data for three radar figures of the dissertation |

## Hardware, firmware and network

- **Boards.** TI IWR6843AOPEVM (the ES2.0 revision is assumed for the elevation sign) and DCA1000EVM,
  connected by the LVDS ribbon, as in [the driver README](xwr_raw_ros/README.md#hardware-setup-iwr6843aopevm).
- **Firmware.** TI's mmWave SDK out-of-box demo on the radar, flashed with UniFlash. The DCA1000 keeps
  its factory FPGA image. The driver sends it a fixed FPGA configuration.
- **Network.** The host port to the DCA1000 needs the static address `192.168.33.30/24`. The DCA1000's
  factory address is `192.168.33.180`. UDP ports 4096 (commands) and 4098 (data) must be free. These
  are the launch file's defaults (`xwr_raw.radar_pub.RadarPub` has upstream's other defaults, `.181`
  and 5096/5098, which the node overrides).
- **Receive buffer.** The driver asks for an 8 MB UDP receive buffer, which Linux caps at
  `net.core.rmem_max`: `sudo sysctl -w net.core.rmem_max=8388608`.
- **Serial port.** The radar's configuration port, `/dev/ttyUSB0` by default, `/dev/ttyISK_00` with
  the udev rule. Your user must be in the `dialout` group.

## Build

```bash
mkdir -p ~/radar_ws/src && cd ~/radar_ws/src
git clone -b ros2 https://github.com/ShengBin-101/xwr_raw_ros.git
cd ~/radar_ws
rosdep install --from-paths src --ignore-src -y
colcon build
source install/setup.bash
```

## Run

```bash
# the driver alone, publishing /radar1/radar_data
ros2 launch xwr_raw_ros radar.launch.py radar1_cfg:=$(ros2 pkg prefix xwr_raw_ros)/share/xwr_raw_ros/configs/6843aop/near_mimo.cfg

# the driver and the processing, with one profile for both
ros2 launch radar_pipelines radar_pipelines.launch.py chirp_profile:=near_mimo
```

## Chirp profiles

In `xwr_raw_ros/configs/6843aop/`. "Range bin" and "maximum range" are what the driver computes from
the chirp parameters.

| Profile | Transmitters | Chirps per frame | Samples | Range bin | Maximum range | Frame rate | Used for |
|---|---|---|---|---|---|---|---|
| `near.cfg` | TX1 only | 16 | 80 | 4.5 cm | 3.61 m | 5 Hz | short range, no azimuth MIMO |
| `near_mimo.cfg` | TX1, TX2, TX3 | 48 | 80 | 4.5 cm | 3.61 m | 5 Hz | the processing's default |
| `test.cfg` | TX1, TX2, TX3 | 60 | 224 | 4.0 cm | 9.03 m | 20 Hz | the dissertation's recordings |

**Range scale.** The processing scales range by the ratio of the profile header's "maximum
unambiguous range" to the range the driver computes: 2.88 / 3.6141 = 0.79688 for `near.cfg` and
`near_mimo.cfg`, 7.17 / 9.03 = 0.79402 for `test.cfg`. Without it, distances read about 26 % too far,
judged by eye against the camera. With it, in the dissertation's first experiment the radar's median
distance inside a bottle's outline was 0.380 m, and the camera put the cloth behind the bottle at
0.389 m. It is a ratio of two numbers TI's tool prints, not a measured calibration, so treat it as
approximate. Each profile's value is in
`radar_pipelines/config/calib/`.

## Things to know

- **Angle axis of the Cartesian heatmaps.** The angle FFT's bins are uniform in sin(theta) over
  ±90°. Until this release, `xwr_raw.rexo_processing.polar_to_cartesian_grid` spread them linearly
  over ±60°, which draws a target at 55° at about 49°. Cartesian heatmaps made with that version,
  including the project's REXO trials, carry the error. The polar heatmaps were not affected.
- **Elevation is exploratory.** The AOP's vertical array has two elements. The elevation code has
  never been checked against ground truth and is off by default.
- **REXO is not included.** The heatmaps follow the layout of the MMVR dataset, which MERL's REXO
  detector reads ([merlresearch/radar-bbox-diffusion](https://github.com/merlresearch/radar-bbox-diffusion),
  AGPL-3.0). In this project REXO's detections were not meaningful on this radar: REXO was trained on
  another radar's heatmaps, which the IWR6843AOP cannot produce.
- **No recordings are published**, apart from the three raw frames in `figures/data/`.

## Licence and credit

Apache License 2.0 ([`LICENSE`](LICENSE)). [`NOTICE`](NOTICE) lists what this branch changed in
upstream's files and what it added. If you use the driver, please cite upstream's papers (see
[the driver README](xwr_raw_ros/README.md#citation)).
