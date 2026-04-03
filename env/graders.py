# ## FILE 4 — `env/graders.py`

# ### What Is It?
# ```
# The JUDGE of our environment.
# After agent completes a task → grader scores it.

# Like a teacher marking your exam:
# Task 1 completed → grader gives score 0.0 to 1.0
# Task 2 completed → grader gives score 0.0 to 1.0
# Task 3 completed → grader gives score 0.0 to 1.0
# ```

# ### Why Is It Important?
# ```
# Judges literally run our graders to evaluate us!
# This is 25% of total hackathon score.

# Must be:
# ✅ Deterministic (same input = same score always)
# ✅ Fair (partial credit for partial success)
# ✅ Clear (obvious what succeeded or failed)
# Open env/graders.py and paste:
# python# env/graders.py
# # ═══════════════════════════════════════════════
# # TASK GRADERS — Scores agent performance
# #
# # Each task has its own grader function.
# # All graders return scores between 0.0 and 1.0
# # ═══════════════════════════════════════════════

from typing import Dict, List
from env.models import EnvironmentState, DailyMetrics


def grade_task_easy(state: EnvironmentState) -> float:
    """
    TASK 1 GRADER — Single SKU Management (Easy)

    Scores based on:
    - Service level: did we fulfill customer demand? (40%)
    - No stockouts:  did inventory ever hit zero?   (30%)
    - Budget:        did we stay within budget?     (30%)

    Returns float 0.0 to 1.0
    """
    if not state.daily_metrics:
        return 0.0

    metrics = state.daily_metrics

    # ── SERVICE LEVEL (40%) ────────────────────
    # What % of customer demand did we fulfill?
    total_demanded = sum(
        sum(m.units_demanded.values())
        for m in metrics
    )
    total_sold = sum(
        sum(m.units_sold.values())
        for m in metrics
    )

    if total_demanded == 0:
        service_level = 1.0
    else:
        service_level = min(1.0, total_sold / total_demanded)

    # ── STOCKOUT SCORE (30%) ───────────────────
    # Penalize days where stockouts occurred
    stockout_days = sum(
        1 for m in metrics if len(m.stockouts) > 0
    )
    total_days = len(metrics)
    stockout_score = 1.0 - (stockout_days / max(1, total_days))

    # ── BUDGET SCORE (30%) ─────────────────────
    # Did agent stay within budget?
    budget_used = state.total_budget - state.budget_remaining
    if budget_used <= state.total_budget:
        budget_score = 1.0
    else:
        # Over budget — penalize proportionally
        overage = budget_used - state.total_budget
        budget_score = max(0.0, 1.0 - (overage / state.total_budget))

    # ── FINAL SCORE ────────────────────────────
    final_score = (
        service_level  * 0.40 +
        stockout_score * 0.30 +
        budget_score   * 0.30
    )

    return round(final_score, 4)


def grade_task_medium(state: EnvironmentState) -> float:
    """
    TASK 2 GRADER — Multi-SKU Budget Management (Medium)

    Scores based on:
    - Service level across all SKUs  (35%)
    - Budget efficiency              (25%)
    - Inventory balance              (25%)
    - No critical stockouts          (15%)

    Returns float 0.0 to 1.0
    """
    if not state.daily_metrics:
        return 0.0

    metrics = state.daily_metrics

    # ── SERVICE LEVEL (35%) ────────────────────
    total_demanded = sum(
        sum(m.units_demanded.values()) for m in metrics
    )
    total_sold = sum(
        sum(m.units_sold.values()) for m in metrics
    )
    service_level = (
        min(1.0, total_sold / total_demanded)
        if total_demanded > 0 else 1.0
    )

    # ── BUDGET EFFICIENCY (25%) ────────────────
    # Reward using budget wisely — not too much, not too little
    budget_used_ratio = (
        (state.total_budget - state.budget_remaining)
        / state.total_budget
    )
    # Ideal: use 60-90% of budget
    if 0.60 <= budget_used_ratio <= 0.90:
        budget_score = 1.0
    elif budget_used_ratio < 0.60:
        # Under-used budget — maybe stocked out unnecessarily
        budget_score = budget_used_ratio / 0.60
    else:
        # Over budget
        budget_score = max(0.0, 1.0 - (budget_used_ratio - 0.90) * 5)

    # ── INVENTORY BALANCE (25%) ────────────────
    # Check if inventory stayed in healthy range
    healthy_days = 0
    total_sku_days = 0

    for m in metrics:
        for sku_id, demanded in m.units_demanded.items():
            total_sku_days += 1
            sold = m.units_sold.get(sku_id, 0)
            if demanded > 0 and sold >= demanded * 0.9:
                healthy_days += 1
            elif demanded == 0:
                healthy_days += 1

    balance_score = (
        healthy_days / total_sku_days
        if total_sku_days > 0 else 1.0
    )

    # ── CRITICAL STOCKOUT PENALTY (15%) ────────
    total_stockout_events = sum(
        len(m.stockouts) for m in metrics
    )
    # Allow up to 5 stockout events before heavy penalty
    critical_score = max(
        0.0, 1.0 - (total_stockout_events / 20)
    )

    # ── FINAL SCORE ────────────────────────────
    final_score = (
        service_level  * 0.35 +
        budget_score   * 0.25 +
        balance_score  * 0.25 +
        critical_score * 0.15
    )

    return round(final_score, 4)


def grade_task_hard(state: EnvironmentState) -> float:
    """
    TASK 3 GRADER — Supplier Disruption Crisis (Hard)

    Scores based on:
    - Survived supplier bankruptcy    (20%)
    - Overall service level           (30%)
    - Holiday surge handled           (25%)
    - Cost efficiency                 (25%)

    Returns float 0.0 to 1.0
    """
    if not state.daily_metrics:
        return 0.0

    metrics = state.daily_metrics
    total_days = len(metrics)

    # ── BANKRUPTCY SURVIVAL (20%) ──────────────
    # Days 45+ should still have decent service
    # despite supplier going bankrupt on day 45
    post_bankruptcy_metrics = (
        metrics[44:] if len(metrics) > 44 else metrics
    )
    if post_bankruptcy_metrics:
        post_demanded = sum(
            sum(m.units_demanded.values())
            for m in post_bankruptcy_metrics
        )
        post_sold = sum(
            sum(m.units_sold.values())
            for m in post_bankruptcy_metrics
        )
        bankruptcy_score = (
            min(1.0, post_sold / post_demanded)
            if post_demanded > 0 else 1.0
        )
    else:
        bankruptcy_score = 0.5

    # ── OVERALL SERVICE LEVEL (30%) ────────────
    total_demanded = sum(
        sum(m.units_demanded.values()) for m in metrics
    )
    total_sold = sum(
        sum(m.units_sold.values()) for m in metrics
    )
    service_level = (
        min(1.0, total_sold / total_demanded)
        if total_demanded > 0 else 1.0
    )

    # ── HOLIDAY SURGE (25%) ────────────────────
    # Days 75-90 are holiday — check performance
    holiday_metrics = (
        metrics[74:] if len(metrics) > 74 else []
    )
    if holiday_metrics:
        holiday_demanded = sum(
            sum(m.units_demanded.values())
            for m in holiday_metrics
        )
        holiday_sold = sum(
            sum(m.units_sold.values())
            for m in holiday_metrics
        )
        holiday_score = (
            min(1.0, holiday_sold / holiday_demanded)
            if holiday_demanded > 0 else 1.0
        )
    else:
        holiday_score = 0.5

    # ── COST EFFICIENCY (25%) ──────────────────
    total_revenue = state.total_revenue
    total_costs   = state.total_costs

    if total_revenue > 0:
        profit_margin = (
            (total_revenue - total_costs) / total_revenue
        )
        cost_score = min(1.0, max(0.0, profit_margin))
    else:
        cost_score = 0.0

    # ── FINAL SCORE ────────────────────────────
    final_score = (
        bankruptcy_score * 0.20 +
        service_level    * 0.30 +
        holiday_score    * 0.25 +
        cost_score       * 0.25
    )

    return round(final_score, 4)


def get_grader(task_id: str):
    """
    Returns the correct grader function for a task.
    Used by environment to grade automatically.
    """
    graders = {
        "task_easy":   grade_task_easy,
        "task_medium": grade_task_medium,
        "task_hard":   grade_task_hard,
    }
    return graders.get(task_id, grade_task_easy)