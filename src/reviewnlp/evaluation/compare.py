"""Structural comparison with a float tolerance, for verifying archived runs.

Verification scripts recompute metrics from saved logits and check them
against numbers recorded when the run happened. Exact ``==`` makes that check
depend on the SciPy/NumPy build doing the recomputation rather than on the
science: identical inputs through an identical expression can land a few ULP
apart across versions. ``matches`` compares structure exactly and floats with
a relative tolerance, so drift in the last digits passes while a real
regression - which moves these values by orders of magnitude - still fails.
"""

from __future__ import annotations

import math

REL_TOL = 1e-9


def matches(actual, expected, rel_tol: float = REL_TOL) -> bool:
    """Same structure, with a relative tolerance on float leaves.

    Dicts must have identical key sets, sequences identical lengths; ints,
    strings and bools compare exactly. Only floats are given slack.

    >>> matches({"p": 5.02808218497859e-06}, {"p": 5.028082184978545e-06})
    True
    >>> matches({"macro_f1": 0.9634}, {"macro_f1": 0.9573})
    False
    """
    # bool is a subclass of int, so it has to be settled before the numeric
    # branch - otherwise True == 1 would compare equal.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, float) or isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False
        # abs_tol stays 0: this is a verification tool, and a value that
        # drifted away from an exact zero is a finding, not noise.
        return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=0.0)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            matches(actual[key], expected[key], rel_tol) for key in actual
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            matches(a, e, rel_tol) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected
