# checks for HydroHETERO. No network needed. Run with: pytest tests/

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrohetero import model  # noqa: E402


def test_field_statistics():
    rng = np.random.default_rng(1)
    for structure in model.STRUCTURES:
        K = model.k_field(200, 100, 1.0, 10.0, 1.5, 6.0, structure, rng)
        y = np.log(K)
        assert y.mean() == pytest.approx(np.log(10.0), abs=1e-9)
        assert y.std() == pytest.approx(1.5, rel=1e-9)
        scales = [model.integral_scale(np.log(model.k_field(200, 100, 1.0, 10.0, 1.5, 6.0, structure, rng)), 1.0)
                  for _ in range(10)]
        assert np.mean(scales) == pytest.approx(6.0, rel=0.1)


def test_connected_transform_keeps_the_distribution():
    rng = np.random.default_rng(2)
    y = model.connect(rng.standard_normal(200000))
    assert y.mean() == pytest.approx(0, abs=0.01) and y.std() == pytest.approx(1, abs=0.01)
    assert np.percentile(y, [5, 50, 95]) == pytest.approx([-1.645, 0, 1.645], abs=0.02)


def test_uniform_flow_is_exact():
    h, qx, qy = model.flow(np.full((30, 60), 4.0), 2.0, 0.01)
    assert qx == pytest.approx(0.04, rel=1e-10) and np.abs(qy).max() < 1e-12
    assert model.effective_k(qx, 2.0, 0.01) == pytest.approx(4.0, rel=1e-10)


def test_flow_conserves_water():
    rng = np.random.default_rng(3)
    K = model.k_field(120, 60, 1.0, 10.0, 2.0, 5.0, 'Connected high K', rng)
    h, qx, qy = model.flow(K, 1.0, 0.01)
    net = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    assert np.abs(net).max() < 1e-9 * qx.max()
    assert qx[:, 0].sum() == pytest.approx(qx[:, -1].sum(), rel=1e-9)


def test_layers_give_the_arithmetic_mean():
    # flow along layers: the effective K is the arithmetic mean
    K = np.repeat(np.array([1.0, 10.0, 100.0, 5.0])[:, None], 50, axis=1).repeat(5, axis=0)
    h, qx, qy = model.flow(K, 1.0, 0.01)
    assert model.effective_k(qx, 1.0, 0.01) == pytest.approx(K[:, 0].mean(), rel=1e-9)


def test_connectivity_changes_effective_k():
    # Zinn and Harvey (2003): connected high K conducts more than the geometric mean,
    # connected low K less, and a multi-Gaussian field in 2D sits close to it
    rng = np.random.default_rng(4)
    ratio = {}
    for structure in model.STRUCTURES:
        values = []
        for _ in range(12):
            K = model.k_field(200, 100, 1.0, 10.0, 2.0, 5.0, structure, rng)
            values.append(model.effective_k(model.flow(K, 1.0, 0.01)[1], 1.0, 0.01) / 10.0)
        ratio[structure] = np.mean(values)
    assert ratio['Connected high K'] > 1.3 and ratio['Connected low K'] < 0.7
    assert ratio['Multi-Gaussian'] == pytest.approx(1.0, abs=0.12)


def test_uniform_field_matches_ogata_banks():
    rng = np.random.default_rng(5)
    h, qx, qy = model.flow(np.full((40, 200), 10.0), 1.0, 0.01)
    arrival = model.track(qx, qy, 1.0, 0.3, [50.0, 150.0], 20000, rng, alpha_l=0.5, alpha_t=0.05)['arrival']
    v = 10.0 * 0.01 / 0.3
    history = ([0.0, 200.0], [100.0, 20.0])
    for k, x in enumerate((50.0, 150.0)):
        times = np.linspace(0.2, 3.0, 40) * x / v
        simulated = model.breakthrough(arrival[:, k], times, *history)
        expected = model.uniform_breakthrough(x, times, *history, v, 0.5 * v)
        assert np.abs(simulated - expected).max() < 1.5      # out of 100, with 20,000 particles


def test_macrodispersion_at_small_variance():
    # arrival-time variance in a multi-Gaussian field of small variance, against first-order theory
    rng = np.random.default_rng(6)
    g = dict(nx=400, ny=120, d=1.0, k_geometric=10.0, sigma=0.5, length=4.0, structure='Multi-Gaussian',
             gradient=0.01, porosity=0.3, alpha_l=0.05, alpha_t=0.005, diffusion=0.0, planes=[350.0],
             n_particles=3000, source=None, snapshots=())
    fields = [model.k_field(400, 120, 1.0, 10.0, 0.5, 4.0, 'Multi-Gaussian', rng) for _ in range(10)]
    arrival = np.concatenate([model.transport(K, g, rng)['arrival'][:, 0] for K in fields])
    v = 10.0 * 0.01 / 0.3
    assert arrival.mean() == pytest.approx(350.0 / v, rel=0.03)
    apparent = arrival.var() * v ** 2 / (2 * 350.0)
    assert apparent == pytest.approx(model.macrodispersivity(0.5, 4.0) + 0.05, rel=0.25)


def test_colour_maps():
    import matplotlib
    from hydrohetero import plots
    assert set(plots.COLOUR_MAPS) >= {'viridis', 'cubehelix', 'gist_earth', 'Purples', 'Greys', 'YlOrBr', 'BuGn', 'BuPu', 'YlGnBu'}
    assert all(name in matplotlib.colormaps for name in plots.COLOUR_MAPS)
    # paths are white on the dark maps and near black on the light ones
    assert plots.path_style('viridis')['color'] == 'white' and plots.path_style('cividis')['color'] == 'white'
    assert plots.path_style('Greys')['color'] == '0.1' and plots.path_style('YlOrBr')['color'] == '0.1'


def test_detail_run_tracks_to_the_edge():
    rng = np.random.default_rng(9)
    K = model.k_field(120, 60, 1.0, 10.0, 1.0, 5.0, 'Multi-Gaussian', rng)
    g = dict(d=1.0, gradient=0.01, porosity=0.3, alpha_l=0.1, alpha_t=0.01, diffusion=0.0, planes=[30.0, 60.0],
             n_particles=300, source=None, snapshots=[50.0])
    plain, detail = model.transport(K, g, rng), model.transport(K, g, rng, detail=True)
    assert plain['arrival'].shape == detail['arrival'].shape == (300, 2)
    assert detail['paths'][-1, :, 0].min() > 119.0          # every plotted path reaches the right-hand edge
    assert np.median(detail['arrival'][:, 1]) == pytest.approx(np.median(plain['arrival'][:, 1]), rel=0.1)


def test_strip_source_and_snapshots():
    rng = np.random.default_rng(7)
    h, qx, qy = model.flow(np.full((40, 100), 10.0), 1.0, 0.01)
    out = model.track(qx, qy, 1.0, 0.3, [80.0], 500, rng, source=(10.0, 20.0), snapshots=[60.0], paths=5)
    assert out['y0'].min() >= 10.0 and out['y0'].max() <= 20.0
    assert np.nanmean(out['snapshots'][0][:, 0]) == pytest.approx(60.0 * 10.0 * 0.01 / 0.3, rel=0.02)
    assert out['paths'].shape[1:] == (5, 3)


def test_release_points_and_tracking_from_a_point():
    rng = np.random.default_rng(8)
    x, y = model.release_points('Line', 5, rng, 10.0, 0.0, 10.0, 40.0)
    assert list(y) == [0.0, 10.0, 20.0, 30.0, 40.0] and set(x) == {10.0}
    x, y = model.release_points('Region', 200, rng, 5.0, 10.0, 15.0, 30.0)
    assert 5 <= x.min() and x.max() <= 15 and 10 <= y.min() and y.max() <= 30
    # in a uniform field with no dispersion, travel time is distance over velocity
    h, qx, qy = model.flow(np.full((40, 100), 10.0), 1.0, 0.01)
    start = model.release_points('Point', 3, rng, 20.0, 12.0)
    out = model.track(qx, qy, 1.0, 0.3, [80.0], 3, rng, start=start, paths=3)
    assert out['arrival'][:, 0] == pytest.approx(60.0 / (10.0 * 0.01 / 0.3), rel=1e-9)
    assert out['paths'][:, :, 1] == pytest.approx(12.0) and out['paths'][0, :, 2] == pytest.approx(0.0)


def test_export_formats():
    from hydrohetero import export
    K = np.arange(1.0, 7.0).reshape(2, 3)          # ny = 2, nx = 3. K[j, i] with j along y
    by_x = export.node_table([K], 2.0, export.NODE_ORDERS[0])
    assert by_x['K field 1 (m/d)'].tolist() == [1, 2, 3, 4, 5, 6]
    assert by_x['x (m)'].tolist() == [1, 3, 5, 1, 3, 5] and by_x['y (m)'].tolist() == [1, 1, 1, 3, 3, 3]
    by_y = export.node_table([K], 2.0, export.NODE_ORDERS[1])
    assert by_y['K field 1 (m/d)'].tolist() == [1, 4, 2, 5, 3, 6] and by_y['y (m)'].tolist() == [1, 3, 1, 3, 1, 3]
    # grids and MODFLOW arrays are written from the top row (highest y) down
    grid = np.loadtxt(export.grid_text(K, ',').splitlines(), delimiter=',')
    assert grid.tolist() == [[4, 5, 6], [1, 2, 3]]
    assert np.loadtxt(export.grid_text(K, ' ').splitlines()).tolist() == [[4, 5, 6], [1, 2, 3]]
    import io, zipfile
    names = zipfile.ZipFile(io.BytesIO(export.archive([K, K], 2.0, {'nodes', 'grid', 'modflow'},
                                                     export.NODE_ORDERS[0], 'x'))).namelist()
    assert {'settings.txt', 'K_by_node.csv', 'grid/K_002.csv', 'modflow/k_001.dat', 'modflow/README.txt'} <= set(names)


def test_app_runs():
    testing = pytest.importorskip('streamlit.testing.v1')
    app = testing.AppTest.from_file(str(ROOT / 'app.py'), default_timeout=300)
    app.run()
    assert not app.exception and not app.info

    def press(label):
        [button for button in app.button if button.label == label][0].click().run()
        assert not app.exception

    def downloads():
        return [button.label for button in app.get('download_button')]

    [slider for slider in app.slider if slider.label.startswith('Number of K fields')][0].set_value(4).run()
    assert not downloads()                         # nothing until the fields are generated
    press('Generate fields')
    assert not app.info and 'Download the fields (.zip)' not in downloads()
    press('Prepare the download')
    assert 'Download the fields (.zip)' in downloads()
    press('Run transport')
    assert 'Download the transport workbook (.xlsx)' in downloads()
    press('Track particles')
    assert 'Download paths (.csv)' in downloads()
    [box for box in app.selectbox if box.label.startswith('Colour map')][0].set_value('YlGnBu').run()
    assert not app.exception
    # changing an input leaves every button usable and says that the results are out of date
    assert not app.get('warning')
    [box for box in app.number_input if box.label.startswith('Standard deviation')][0].set_value(2.0).run()
    [box for box in app.number_input if box.label.startswith('Mean hydraulic gradient')][0].set_value(0.02).run()
    assert len(app.get('warning')) == 2 and not any(button.disabled for button in app.button)
    press('Generate fields')
    press('Run transport')
    assert not app.get('warning')
    [radio for radio in app.radio if radio.label == 'Release from'][0].set_value('Region').run()
    press('Track particles')
    assert not app.exception
