"""End-to-end test for the radar_pipelines package.

Synthesises a RadarFrameFull-shaped object with a point target at a known
range/azimuth and runs both pipelines plus every CFAR method over it.
"""
import numpy as np

FAILS = []


def check(name, cond, detail=''):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ''))
    if not cond:
        FAILS.append(name)


def run_all():
    FAILS.clear()

    # ---------------------------------------------------------------- imports
    from radar_pipelines.core import (
        load_calibration, available_profiles, cfar_2d, CFAR_METHODS,
        BackgroundBuffer, cluster_points_kdtree,
    )
    from radar_pipelines.pipelines import (
        RexoHeatmapConfig, RexoHeatmapPipeline, PointCloudConfig, PointCloudPipeline,
    )
    check('imports', True)

    # ------------------------------------------------------------ calibration
    profiles = available_profiles()
    check('calib: the three shipped profiles are discoverable',
          sorted(profiles) == ['near', 'near_mimo', 'test'], str(profiles))

    cal = load_calibration('near_mimo')
    check('calib: near_mimo scale', abs(cal.range_scale - 0.79688) < 1e-6, f'{cal.range_scale}')
    check('calib: apply_range', abs(cal.apply_range(3.6141) - 2.88) < 0.01,
          f'{cal.apply_range(3.6141):.4f}')

    cal_t = load_calibration('test')
    check('calib: test scale', abs(cal_t.range_scale - 0.79402) < 1e-6, f'{cal_t.range_scale}')

    check('calib: identity when unset', load_calibration('').is_identity)

    try:
        load_calibration('does_not_exist')
        check('calib: unknown profile raises', False, 'no exception')
    except FileNotFoundError:
        check('calib: unknown profile raises (not silent identity)', True)


    class _L:
        def __init__(self): self.errs = []
        def error(self, m, **k): self.errs.append(m)
        def warn(self, m, **k): pass
        def info(self, m, **k): pass


    class _F:
        pass


    f = _F(); f.range_max = 9.03  # test.cfg's raw value, wrong for near_mimo
    log = _L()
    check('calib: mismatch guard fires', cal.check_against_frame(f, log) is False
          and 'MISMATCH' in log.errs[0])
    f.range_max = 3.6141
    check('calib: guard silent when consistent', cal.check_against_frame(f, _L()) is True)

    # ------------------------------------------------------------------- CFAR
    rng = np.random.default_rng(0)
    hm = rng.normal(5.0, 0.2, size=(64, 48))
    hm[30, 24] += 6.0            # isolated strong target
    hm[30, 30] += 6.0            # second target, 6 cells away in azimuth
    for m in CFAR_METHODS:
        mask = cfar_2d(hm, method=m, guard_len=2, noise_len=4, threshold=2.5)
        hits = set(zip(*np.where(mask)))
        check(f'cfar[{m}]: finds both targets',
              (30, 24) in hits and (30, 30) in hits,
              f'{len(hits)} detections')

    # CA vs OS under masking: put an interferer inside the training window.
    hm2 = rng.normal(5.0, 0.2, size=(64, 48))
    hm2[30, 20] += 5.0
    # A strong neighbouring object smeared over an azimuth beamwidth. It puts
    # ~20 of the 140 training cells for (30,20) far above the noise floor, which
    # drags CA-CFAR's mean above the target and masks it. That is 14% of the
    # training set, inside OS-CFAR's tolerance at rank_frac=0.75 (which
    # discards the top 25%), so OS still sees a clean floor. Push contamination
    # past (1 - rank_frac) and OS fails too: it buys robustness, not immunity.
    hm2[26:36, 24:26] += 20.0
    ca = cfar_2d(hm2, method='ca', guard_len=2, noise_len=4, threshold=2.5)
    os_ = cfar_2d(hm2, method='os', guard_len=2, noise_len=4, threshold=2.5)
    check('cfar: OS beats CA on the masking case',
          bool(os_[30, 20]) and not bool(ca[30, 20]),
          f'ca={bool(ca[30,20])} os={bool(os_[30,20])}')

    try:
        cfar_2d(np.zeros((5, 5)), method='ca', guard_len=2, noise_len=4)
        check('cfar: too-small heatmap raises', False)
    except ValueError:
        check('cfar: too-small heatmap raises', True)
    try:
        cfar_2d(hm, method='nope')
        check('cfar: bad method raises', False)
    except ValueError:
        check('cfar: bad method raises', True)

    # --------------------------------------------------------------- clutter
    bg = BackgroundBuffer('t', mode='subtract')
    statuses = [bg.handle_capture(np.full((4, 4), 3.0), True, 3) for _ in range(3)]
    check('clutter: capture completes on Nth frame',
          statuses == ['started', 'collecting', 'done'], str(statuses))
    check('clutter: subtraction applies',
          np.allclose(bg.apply(np.full((4, 4), 5.0)), 2.0))
    bgc = BackgroundBuffer('t2', mode='subtract_clip')
    for _ in range(2):
        bgc.handle_capture(np.full((4, 4), 9.0), True, 2)
    check('clutter: subtract_clip floors at 0',
          np.allclose(bgc.apply(np.full((4, 4), 5.0)), 0.0))
    check('clutter: shape mismatch is a no-op, not a crash',
          bgc.apply(np.full((8, 8), 5.0)).shape == (8, 8))

    # ------------------------------------------------- synthetic radar frame
    N_CHIRPS, N_RX, N_SAMPLES = 48, 4, 80


    def make_frame(target_bin=20, snr=40.0):
        fr = _F()
        fr.platform = 'xWR68xx_AOP'
        fr.adc_output_fmt = 1
        fr.range_bias = 0.0
        fr.range_max = 3.6141
        fr.range_res = 0.0452
        fr.rx_phase_bias = [1.0, 0.0] * 12  # one complex bias per virtual channel
        fr.rx = [1] * N_RX
        fr.tx = [1, 1, 1]
        fr.shape = [N_CHIRPS, N_RX, N_SAMPLES]
        n = np.arange(N_SAMPLES)
        beat = np.exp(2j * np.pi * target_bin * n / N_SAMPLES)
        cube = np.zeros((N_CHIRPS, N_RX, N_SAMPLES), dtype=complex)
        for c in range(N_CHIRPS):
            for r in range(N_RX):
                # small per-RX phase ramp -> an off-boresight azimuth
                cube[c, r] = snr * beat * np.exp(1j * 0.4 * r)
        cube += (np.random.default_rng(1).normal(0, 1, cube.shape)
                 + 1j * np.random.default_rng(2).normal(0, 1, cube.shape))
        inter = np.empty(cube.size * 2)
        inter[0::2] = cube.real.ravel()
        inter[1::2] = cube.imag.ravel()
        fr.data = np.clip(inter, -32768, 32767).astype(np.int16)

        class _H:
            stamp = None
        fr.header = _H()
        return fr


    # ------------------------------------------------------- REXO pipeline
    pipe = RexoHeatmapPipeline(calibration=cal)
    cfg = RexoHeatmapConfig(n_range_bins=64, n_azimuth_bins=32, n_elevation_bins=32)
    res = pipe.process(make_frame(), cfg)
    check('rexo: hm_hori shape', res.hm_hori.shape == (64, 32), str(res.hm_hori.shape))
    check('rexo: hm_vert shape', res.hm_vert.shape == (64, 32), str(res.hm_vert.shape))
    check('rexo: finite output', np.all(np.isfinite(res.hm_hori)))
    check('rexo: nonzero output', float(np.max(res.hm_hori)) > 0)
    check('rexo: range_max calibrated', abs(res.actual_range_max - 2.88) < 0.01,
          f'{res.actual_range_max:.4f}')

    # near-field exclusion + background capture through the pipeline
    cfg_nf = RexoHeatmapConfig(n_range_bins=64, n_azimuth_bins=32, n_elevation_bins=32,
                               min_range_m=1.0)
    res_nf = pipe.process(make_frame(), cfg_nf)
    check('rexo: near-field exclusion zeroes near rows',
          float(np.max(res_nf.hm_hori_raw[-5:, :])) == 0.0)

    # Polar output has the near rows first (row 0 nearest), the reverse of
    # the Cartesian layout, so the exclusion must zero the first rows only.
    cfg_nf_polar = RexoHeatmapConfig(n_range_bins=64, n_azimuth_bins=32, n_elevation_bins=32,
                                     min_range_m=1.0, output_cartesian=False,
                                     conditioning_enabled=False)
    res_nfp = RexoHeatmapPipeline(calibration=cal).process(make_frame(), cfg_nf_polar)
    n_rows = res_nfp.hm_hori_raw.shape[0]
    n_near = int(np.sum(np.linspace(0.0, res_nfp.actual_range_max, n_rows) < 1.0))
    check('rexo: polar near-field exclusion zeroes the first rows only',
          0 < n_near < n_rows
          and float(np.max(res_nfp.hm_hori_raw[:n_near, :])) == 0.0
          and float(np.max(res_nfp.hm_hori_raw[n_near:, :])) > 0.0,
          f'{n_near} of {n_rows} rows excluded')

    cfg_bg = RexoHeatmapConfig(n_range_bins=64, n_azimuth_bins=32, n_elevation_bins=32,
                               background_capture=True, background_n_frames=2)
    s = [pipe.process(make_frame(), cfg_bg).capture_status for _ in range(2)]
    check('rexo: background capture reaches done', s[-1] == 'done', str(s))

    # ------------------------------------------------- pointcloud pipeline
    pc = PointCloudPipeline(calibration=cal)
    pcfg = PointCloudConfig(capon_angle_resolution=4, capon_angle_range=60,
                            cfar_threshold=1.0, background_enabled=False)
    r1 = pc.process(make_frame(), pcfg)
    check('pointcloud: first frame is warmup', r1.warmup is True)
    r2 = pc.process(make_frame(), pcfg)
    check('pointcloud: second frame produces a heatmap',
          r2.debug_heatmap is not None and not r2.warmup)
    check('pointcloud: detections found', len(r2.points) > 0, f'{len(r2.points)} points')
    if r2.points:
        p = r2.points[0]
        check('pointcloud: point schema',
              set(p) == {'x', 'y', 'z', 'intensity', 'range', 'azimuth', 'elevation',
                         'elevation_coherence', 'elevation_valid'})
        rmax = max(q['range'] for q in r2.points)
        check('pointcloud: ranges within calibrated max', rmax <= 2.89, f'max range {rmax:.3f}m')
        check('pointcloud: z==0 with elevation_mode=none',
              all(q['z'] == 0.0 for q in r2.points))

    # ------------------------------------------- interferometric elevation
    from radar_pipelines.core.elevation2d import (
        estimate_elevation_interferometric, AOP_ROW_A, AOP_ROW_B)

    def synth_cube(az_deg, el_deg, n_chirps=32, n_samples=80, rbin=20):
        n = np.arange(n_samples)
        beat = np.exp(2j * np.pi * rbin * n / n_samples)
        cube = np.zeros((n_chirps, 12, n_samples), dtype=complex)
        el_phase = np.exp(1j * np.pi * np.sin(np.deg2rad(el_deg)))
        for j in range(4):
            azp = np.exp(-1j * np.pi * j * np.sin(np.deg2rad(az_deg)))
            cube[:, AOP_ROW_A[j], :] += 100 * beat * azp
            cube[:, AOP_ROW_B[j], :] += 100 * beat * azp * el_phase
        return cube

    errs = []
    for el_true in (0.0, -20.0, 10.0, 35.0, -40.0):
        # row_phase_offset_deg=0: synth_cube models the ideal array directly,
        # with no flip_aop_phase inversion to undo. The live path uses the
        # 180 default (see core/elevation2d.py).
        emap, coh = estimate_elevation_interferometric(
            synth_cube(0.0, el_true), angle_range=70, angle_res=1,
            row_phase_offset_deg=0.0)
        errs.append(abs(float(emap[20, 70]) - el_true))
    check('elevation: recovers known angles', max(errs) < 0.5,
          f'max error {max(errs):.4f} deg')
    check('elevation: map is per-CELL (2D)', emap.ndim == 2 and emap.shape == (80, 141),
          str(emap.shape))

    # coherence must separate a real return from noise
    rng2 = np.random.default_rng(7)
    noisy = synth_cube(0.0, 20.0) + (rng2.normal(0, 200, (32, 12, 80))
                                      + 1j * rng2.normal(0, 200, (32, 12, 80)))
    emap_n, coh_n = estimate_elevation_interferometric(noisy, angle_range=70, angle_res=1,
                                                       row_phase_offset_deg=0.0)
    check('elevation: coherence high on target, low on empty sky',
          coh_n[20, 70] > 0.7 and coh_n[5, 5] < 0.5,
          f'target {coh_n[20,70]:.2f} vs empty {coh_n[5,5]:.2f}')

    # single-TX cube must fail loudly, not silently
    try:
        estimate_elevation_interferometric(np.zeros((16, 8, 80), dtype=complex))
        check('elevation: single-TX raises', False)
    except ValueError as e:
        check('elevation: single-TX raises clearly', 'single-TX' in str(e).lower()
              or 'SINGLE-TX' in str(e))

    # end to end through the pipeline
    pc_e = PointCloudPipeline(calibration=cal)
    ecfg = PointCloudConfig(capon_angle_resolution=4, capon_angle_range=60,
                            cfar_threshold=1.0, background_enabled=False,
                            elevation_mode='interferometric', elevation_min_coherence=0.0,
                            elevation_min_power_pct=0.0)
    pc_e.process(make_frame(), ecfg)
    re_ = pc_e.process(make_frame(), ecfg)
    check('elevation: pipeline produces nonzero z',
          len(re_.points) > 0 and any(q['z'] != 0.0 for q in re_.points),
          f"{len(re_.points)} points, z range "
          f"[{min(q['z'] for q in re_.points):.3f}, {max(q['z'] for q in re_.points):.3f}]")

    try:
        PointCloudPipeline(calibration=cal).process(
            make_frame(), PointCloudConfig(elevation_mode='bogus'))
        check('elevation: bad mode raises', False)
    except ValueError:
        check('elevation: bad mode raises', True)

    for m in CFAR_METHODS:
        pc_m = PointCloudPipeline(calibration=cal)
        c = PointCloudConfig(capon_angle_resolution=4, capon_angle_range=60,
                             cfar_method=m, cfar_threshold=1.0, background_enabled=False)
        pc_m.process(make_frame(), c)
        rm = pc_m.process(make_frame(), c)
        check(f'pointcloud[{m}]: runs end to end', rm.debug_heatmap is not None,
              f'{len(rm.points)} points')

    # clustering
    pos = np.array([[0, 0, 1.0], [0.01, 0, 1.0], [0.02, 0, 1.0], [5, 5, 5.0]])
    cl = cluster_points_kdtree(pos, 0.15, 2, 500)
    check('cluster: one 3-point cluster, singleton dropped',
          len(cl) == 1 and len(cl[0]) == 3, f'{[len(c) for c in cl]}')

    return list(FAILS)


def test_radar_pipelines():
    failures = run_all()
    assert not failures, f'{len(failures)} check(s) failed: {failures}'


if __name__ == '__main__':
    import sys as _s
    _f = run_all()
    print()
    print(f'{len(_f)} FAILURE(S): {_f}' if _f else 'all checks passed')
    _s.exit(1 if _f else 0)
