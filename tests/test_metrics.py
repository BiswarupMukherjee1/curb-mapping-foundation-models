import sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrs_ai.metrics import line_metrics  # noqa: E402

GT = [LineString([(0, 0), (10, 0)])]


def test_perfect_prediction():
    m = line_metrics(GT, GT)
    assert m["completeness_5cm"] > 0.99 and m["correctness_5cm"] > 0.99 and m["rmse_matched_5cm_cm"] < 0.5


def test_offset_prediction_respects_tolerance():
    pred = [LineString([(0, 0.08), (10, 0.08)])]          # 8 cm off
    m = line_metrics(GT, pred)
    assert m["completeness_5cm"] < 0.01 and m["completeness_10cm"] > 0.99 and m["correctness_10cm"] > 0.99


def test_half_length_and_false_positive():
    pred = [LineString([(0, 0), (5, 0)]), LineString([(0, 3), (5, 3)])]   # half the GT + 5 m spurious
    m = line_metrics(GT, pred)
    assert abs(m["completeness_10cm"] - 0.5) < 0.02 and abs(m["correctness_10cm"] - 0.5) < 0.02


def test_direction_check_rejects_crossing_line():
    pred = [LineString([(5, -1), (5, 1)])]                # perpendicular, touches GT at one point
    m = line_metrics(GT, pred, max_angle_deg=30)
    assert m["correctness_10cm"] < 0.15


def test_visibility_weight():
    m = line_metrics(GT, [LineString([(0, 0), (5, 0)])], gt_weight=lambda xy: (xy[:, 0] < 5).astype(float))
    assert m["completeness_10cm"] > 0.98                   # only the visible half counts
