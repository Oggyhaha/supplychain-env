# env/graders.py

from typing import Dict, List
from env.models import EnvironmentState, DailyMetrics


def _service_level(metrics: List[DailyMetrics]) -> float:
    total_d = sum(sum(m.units_demanded.values()) for m in metrics)
    total_s = sum(sum(m.units_sold.values()) for m in metrics)
    if total_d == 0:
        return 1.0
    return min(1.0, total_s / total_d)


def _stockout_free_days(metrics: List[DailyMetrics]) -> float:
    bad_days = sum(1 for m in metrics if m.stockouts)
    return 1.0 - (bad_days / max(1, len(metrics)))


def _budget_score(state: EnvironmentState) -> float:
    if state.budget_remaining < 0:
        return 0.3
    used = (state.total_budget - state.budget_remaining) / state.total_budget
    if 0.3 <= used <= 0.95:
        return 1.0
    elif used < 0.3:
        return 0.7
    else:
        return 0.5


def grade_task_easy(state: EnvironmentState) -> float:
    if not state.daily_metrics:
        return 0.0
    m = state.daily_metrics
    score = (
        _service_level(m)      * 0.50 +
        _stockout_free_days(m) * 0.30 +
        _budget_score(state)   * 0.20
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_task_medium(state: EnvironmentState) -> float:
    if not state.daily_metrics:
        return 0.0
    m = state.daily_metrics
    score = (
        _service_level(m)      * 0.50 +
        _stockout_free_days(m) * 0.30 +
        _budget_score(state)   * 0.20
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_task_hard(state: EnvironmentState) -> float:
    if not state.daily_metrics:
        return 0.0
    m = state.daily_metrics
    score = (
        _service_level(m)      * 0.50 +
        _stockout_free_days(m) * 0.30 +
        _budget_score(state)   * 0.20
    )
    return round(min(1.0, max(0.0, score)), 4)


def get_grader(task_id: str):
    return {
        "task_easy":   grade_task_easy,
        "task_medium": grade_task_medium,
        "task_hard":   grade_task_hard,
    }.get(task_id, grade_task_easy)