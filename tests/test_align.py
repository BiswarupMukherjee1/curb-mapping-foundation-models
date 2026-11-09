import sys
from pathlib import Path

import numpy as np
from affine import Affine
from scipy import ndimage
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrs_ai.align import densify, edge_energy_grid, gradient_magnitude, peak_subpixel, phase_correlation  # noqa: E402
from pcrs_ai.io import read_world_file  # noqa: E402


def test_world_file_centre_to_corner(tmp_path):
    p = tmp_path / "t.wld"
    p.write_text("0.05\n0\n0\n-0.05\n1000.025\n2000.975\n")
    tr = read_world_file(p)
    assert np.isclose(tr.c, 1000.0) and np.isclose(tr.f, 2001.0)
    assert tr == Affine(0.05, 0, 1000.0, 0, -0.05, 2001.0)


def test_phase_correlation_subpixel():
    rng = np.random.default_rng(0)
    a = ndimage.gaussian_filter(rng.random((256, 256)), 2)
    b = ndimage.shift(a, (-1.5, 2.25), order=3, mode="wrap")      # (rows, cols)
    dx, dy = phase_correlation(a, b)
    assert abs(dx - 2.25) < 0.15 and abs(dy + 1.5) < 0.15


def test_edge_energy_recovers_line_offset():
    img = np.zeros((200, 200), np.float32)
    img[:, 103:] = 1.0                                 # vertical edge between col 102 and 103 (at x = 102.5)
    img = ndimage.gaussian_filter(img, 0.7)
    grad = gradient_magnitude(img, 1.0)
    rows = np.arange(20, 180, 1.0)
    cols = np.full_like(rows, 100.0)                  # line drawn 2.5 px left of the edge
    shifts = np.arange(-5, 5.01, 0.5)
    E = edge_energy_grid(grad, cols, rows, shifts)
    dx, dy, sharp = peak_subpixel(E, shifts)
    assert abs(dx - 2.5) < 0.3 and sharp > 2


def test_densify_spacing():
    xy, t = densify([LineString([(0, 0), (1, 0)])], 0.1)
    assert len(xy) == 10 and np.allclose(t, [[1, 0]] * 10)


def test_densify_returns_source_ids():
    from shapely.geometry import MultiLineString
    g = [LineString([(0, 0), (1, 0)]), None, MultiLineString([[(0, 1), (0.5, 1)], [(2, 2), (2, 2.3)]])]
    xy, t, ids = densify(g, 0.1, return_ids=True)
    assert set(ids.tolist()) == {0, 2} and (ids == 0).sum() == 10 and (ids == 2).sum() == 5 + 3
