"""Cross-validate the Python ripser pipeline against Wolfram symbolic ground truth.

Reads ``validation/symbolic_diagrams.json`` produced by
``validation/symbolic_persistence.wls`` and compares each H1 birth/death
pair against ripser's output on the same point clouds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from void.topology.persistence import compute_persistence

BASELINE = Path(__file__).parent.parent / "validation" / "symbolic_diagrams.json"
# ripser computes VR persistence in float32; agreement with arbitrary-precision
# Wolfram is limited to ~1e-5. We compare the dominant (longest-persistence) H1
# feature, which is what carries scientific signal.
ABS_TOL = 1e-5


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.skip(
            "Wolfram symbolic baseline not generated; "
            "run `wolframscript -file validation/symbolic_persistence.wls`"
        )
    return json.loads(BASELINE.read_text())


@pytest.mark.parametrize("name", ["circle_8", "gaussian_blob"])
def test_ripser_dominant_h1_matches_wolfram(baseline, name):
    cloud_data = baseline["clouds"][name]
    points = np.asarray(cloud_data["points"], dtype=float)
    expected_pairs = [
        (float(b), float(d))
        for b, d in cloud_data["h1_pairs"]
        if d != "Infinity"
    ]
    assert expected_pairs, f"{name}: Wolfram baseline has no finite H1 pairs"
    b_exp, d_exp = max(expected_pairs, key=lambda bd: bd[1] - bd[0])

    pd = compute_persistence(points, maxdim=1)
    h1 = pd.diagrams[1]
    finite = h1[np.isfinite(h1[:, 1])] if len(h1) else h1
    assert len(finite) > 0, f"{name}: ripser produced no finite H1 pairs"
    b_act, d_act = max(
        ((float(b), float(d)) for b, d in finite),
        key=lambda bd: bd[1] - bd[0],
    )

    assert b_act == pytest.approx(b_exp, abs=ABS_TOL), (
        f"{name}: dominant-H1 birth mismatch ripser={b_act} wolfram={b_exp}"
    )
    assert d_act == pytest.approx(d_exp, abs=ABS_TOL), (
        f"{name}: dominant-H1 death mismatch ripser={d_act} wolfram={d_exp}"
    )
