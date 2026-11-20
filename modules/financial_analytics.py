"""Illustrative inspection-cost analytics. All inputs are user assumptions."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FinancialResult:
    savings_per_inspection: float
    annual_gross_savings: float
    annual_platform_cost: float
    annual_net_benefit: float
    roi_percent: float | None        # None when platform cost is zero
    break_even_inspections: int | None  # None when savings per inspection <= 0
    conventional_cost: float
    targeted_cost: float
    annual_inspections: int
    monthly_subscription: float


def calculate(conventional_cost: float, targeted_cost: float,
              annual_inspections: int, monthly_subscription: float) -> FinancialResult:
    """Apply the TerraSignal illustrative model.

    savings/inspection = conventional - targeted (may be negative)
    gross = savings/inspection x annual inspections
    platform = monthly x 12
    net = gross - platform
    ROI % = net / platform x 100 (undefined if platform cost is 0)
    break-even = platform / savings per inspection, rounded up
    """
    if min(conventional_cost, targeted_cost, monthly_subscription) < 0:
        raise ValueError("Costs cannot be negative.")
    if annual_inspections < 0 or int(annual_inspections) != annual_inspections:
        raise ValueError("Annual inspections must be a non-negative whole number.")
    n = int(annual_inspections)
    per = conventional_cost - targeted_cost
    gross = per * n
    platform = monthly_subscription * 12
    net = gross - platform
    roi = (net / platform * 100) if platform > 0 else None
    if platform == 0:
        be: int | None = 0
    elif per <= 0:
        be = None
    else:
        be = math.ceil(round(platform / per, 9))
    return FinancialResult(per, gross, platform, net, roi, be, conventional_cost,
                           targeted_cost, n, monthly_subscription)


def cumulative_curve(res: FinancialResult, max_n: int | None = None) -> list[tuple[int, float]]:
    """(inspections, net benefit) points for charting."""
    top = max_n if max_n is not None else max(res.annual_inspections * 2, 10)
    return [(k, res.savings_per_inspection * k - res.annual_platform_cost)
            for k in range(0, top + 1)]


ILLUSTRATIVE_NOTE = ("All financial figures are illustrative and based only on the "
                     "assumptions entered. They are not quotations, market prices or "
                     "guaranteed savings. Potential avoided environmental damage is "
                     "not counted as a proven financial benefit.")
