import sys
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrs_ai.metrics import line_metrics  # noqa: E402
from pcrs_ai.vectorize import prob_to_lines, trace_skeleton  # noqa: E402


def test_trace_simple_chain_and_cycle():
    sk = np.zeros((20, 20), bool)
    sk[5, 2:15] = True                       # open chain
    sk[12, 12:16] = sk[15, 12:16] = True     # square cycle
    sk[12:16, 12] = sk[12:16, 15] = True
    chains = trace_skeleton(sk)
    n = sum(len(c) for c in chains)
    assert n >= 13 + 12 and any(len(c) == 13 for c in chains)


def test_prob_band_recovers_line_within_tolerance():
    gsd = 0.05
    tr = Affine(gsd, 0, 1000.0, 0, -gsd, 2000.0)
    prob = np.zeros((200, 200), np.float32)
    prob[99:102, 20:180] = 0.9               # 3 px band centred on row 100 -> y = 2000 - 100.5 * 0.05
    lines, conf = prob_to_lines(prob, tr, threshold=0.5)
    y = 2000.0 - 100.5 * gsd
    gt = [LineString([(1000 + 20 * gsd, y), (1000 + 180 * gsd, y)])]
    m = line_metrics(gt, lines)
    assert m["completeness_5cm"] > 0.9 and m["correctness_5cm"] > 0.9 and abs(conf.mean() - 0.9) < 1e-3
