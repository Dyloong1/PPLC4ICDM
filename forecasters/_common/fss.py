"""Forecast Skill Score against persistence.

``FSS = 1 - rel_L2(forecaster) / rel_L2(persistence)``.

``FSS > 0`` means the forecaster beats naive persistence (assume the
field at the prediction horizon equals the field at ``t``); ``FSS < 0``
means persistence wins.
"""

from __future__ import annotations


def forecast_skill_score(rel_l2_forecaster: float,
                          rel_l2_persistence: float) -> float:
    if rel_l2_persistence <= 0:
        return float("nan")
    return float(1.0 - rel_l2_forecaster / rel_l2_persistence)
