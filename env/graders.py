# env/graders.py
# ═══════════════════════════════════════════════
# IMPROVED TASK GRADERS
#
# Fairer scoring with proper partial credit.
# Deterministic and reproducible.
# Difficulty properly calibrated.
# ═══════════════════════════════════════════════

from typing import Dict, List
from env.models import EnvironmentState, DailyMetrics


def _calculate_service_level(metrics: List[DailyMetrics]) -> float:
    """
    Helper: calculates overall service level
    across all days and all SKUs.
    Service level = % of demand fulfilled.
    """
    total_demanded = sum(
        sum(m.units_demanded.values()) for m in metrics
    )
    total_sold = sum(
        sum(m.units_sold.values()) for m in metrics
    )
    if total_demanded == 0:
        return 1.0
    return min(1.0, total_sold / total_demanded)


def _calculate_stockout_rate(
    metrics: List[DailyMetrics],
    total_days: int
) -> float:
    """
    Helper: what fraction of days had NO stockouts.
    1.0 = never stocked out
    0.0 = stocked out every day
    """
    stockout_days = sum(
        1 for m in metrics if len(m.stockouts) > 0
    )
    return 1.0 - (stockout_days / max(1, total_days))


def _calculate_budget_score(state: EnvironmentState) -> float:
    """
    Helper: scores budget usage.
    Rewards using budget wisely.
    Penalizes going over budget.
    """
    if state.budget_remaining < 0:
        # Over budget - penalize
        overage = abs(state.budget_remaining)
        return max(0.0, 1.0 - (overage / state.total_budget))

    used_ratio = (
        (state.total_budget - state.budget_remaining)
        / state.total_budget
    )

    # Reward using 40-90% of budget
    if 0.40 <= used_ratio <= 0.90:
        return 1.0
    elif used_ratio < 0.40:
        # Under-used - might have caused stockouts
        return 0.6 + (used_ratio / 0.40) * 0.4
    else:
        # Slightly over ideal - small penalty
        return max(0.5, 1.0 - (used_ratio - 0.90) * 2)


def grade_task_easy(state: EnvironmentState) -> float:
    """
    TASK 1 GRADER — Single SKU (Easy)

    Scoring:
    Service level  → 50% weight
    Stockout rate  → 30% weight
    Budget score   → 20% weight

    Easy task: agent should score 0.65-0.85
    Random agent:  ~0.30
    Perfect agent: ~0.95
    """
    if not state.daily_metrics:
        return 0.0

    metrics = state.daily_metrics

    # Calculate components
    service  = _calculate_service_level(metrics)
    no_stock = _calculate_stockout_rate(metrics, len(metrics))
    budget   = _calculate_budget_score(state)

    # Weighted final score
    final = (
        service  * 0.50 +
        no_stock * 0.30 +
        budget   * 0.20
    )

    # Bonus for excellent performance
    if service >= 0.98 and no_stock >= 0.95:
        final = min(1.0, final + 0.05)

    return round(final, 4)


def grade_task_medium(state: EnvironmentState) -> float:
    """
    TASK 2 GRADER — Multi SKU (Medium)

    Scoring:
    Service level     → 40% weight
    Stockout rate     → 25% weight
    Budget efficiency → 20% weight
    SKU balance       → 15% weight

    Medium task: agent should score 0.50-0.70
    Random agent:  ~0.20
    Perfect agent: ~0.90
    """
    if not state.daily_metrics:
        return 0.0

    metrics = state.daily_metrics

    # Main components
    service  = _calculate_service_level(metrics)
    no_stock = _calculate_stockout_rate(metrics, len(metrics))
    budget   = _calculate_budget_score(state)

    # SKU balance score
    # Checks if agent managed ALL SKUs well
    # not just the easy ones
    sku_scores = {}
    for m in metrics:
        for sku_id, demanded in m.units_demanded.items():
            if sku_id not in sku_scores:
                sku_scores[sku_id] = {"demanded": 0, "sold": 0}
            sku_scores[sku_id]["demanded"] += demanded
            sku_scores[sku_id]["sold"] += m.units_sold.get(sku_id, 0)

    sku_service_levels = []
    for sku_id, totals in sku_scores.items():
        if totals["demanded"] > 0:
            sl = totals["sold"] / totals["demanded"]
            sku_service_levels.append(sl)

    # Balance = worst SKU service level
    # Penalizes ignoring any single SKU
    if sku_service_levels:
        worst_sku = min(sku_service_levels)
        balance   = (sum(sku_service_levels) / len(sku_service_levels)
                     * 0.7 + worst_sku * 0.3)
    else:
        balance = 1.0

    final = (
        service  * 0.40 +
        no_stock * 0.25 +
        budget   * 0.20 +
        balance  * 0.15
    )

    return round(final, 4)


def grade_task_hard(state: EnvironmentState) -> float:
    """
    TASK 3 GRADER — Disruption Crisis (Hard)

    Scoring:
    Overall service level    → 25% weight
    Post bankruptcy service  → 25% weight
    Holiday surge handling   → 25% weight
    Cost efficiency          → 25% weight

    Hard task: agent should score 0.35-0.55
    Random agent:  ~0.15
    Perfect agent: ~0.85
    """
    if not state.daily_metrics:
        return 0.0

    metrics    = state.daily_metrics
    total_days = len(metrics)

    # Overall service level
    overall_service = _calculate_service_level(metrics)

    # Post bankruptcy performance (days 45+)
    # Tests if agent recovered from supplier loss
    if total_days > 44:
        post_metrics    = metrics[44:]
        post_service    = _calculate_service_level(post_metrics)
    else:
        post_service    = overall_service

    # Holiday surge performance (days 75+)
    # Tests if agent prepared for demand spike
    if total_days > 74:
        holiday_metrics = metrics[74:]
        holiday_service = _calculate_service_level(holiday_metrics)
    else:
        holiday_service = overall_service * 0.8

    # Cost efficiency
    # Did agent make money despite disruptions?
    if state.total_revenue > 0:
        profit_margin = (
            (state.total_revenue - state.total_costs)
            / state.total_revenue
        )
        cost_score = min(1.0, max(0.0, profit_margin + 0.3))
    else:
        cost_score = 0.1

    final = (
        overall_service  * 0.25 +
        post_service     * 0.25 +
        holiday_service  * 0.25 +
        cost_score       * 0.25
    )

    # Hard task bonus for exceptional performance
    if overall_service >= 0.90:
        final = min(1.0, final + 0.05)

    return round(final, 4)


def get_grader(task_id: str):
    """Returns correct grader for task"""
    graders = {
        "task_easy":   grade_task_easy,
        "task_medium": grade_task_medium,
        "task_hard":   grade_task_hard,
    }
    return graders.get(task_id, grade_task_easy)