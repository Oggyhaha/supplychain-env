# env/environment.py
# ═══════════════════════════════════════════════
# CORE ENVIRONMENT
# ═══════════════════════════════════════════════

import yaml
from typing import Optional, Dict, List
from env.models import (
    SKU, Supplier, PurchaseOrder, DemandPattern,
    WarehouseObservation, RestockAction, SupplyChainReward,
    StepResult, EnvironmentState, DailyMetrics,
    TaskDifficulty, OrderStatus, SupplierStatus
)
from env.demand_generator import DemandGenerator
from env.suppliers import SupplierManager, get_default_suppliers
from env.graders import get_grader


# ── TASK CONFIGURATIONS ────────────────────────
TASK_CONFIGS = {

    "task_easy": {
        "difficulty": TaskDifficulty.EASY,
        "total_days": 30,
        "budget": 50000.0,
        "seed": 42,
        "skus": [
            SKU(
                sku_id="LAPTOP-001",
                name="Business Laptop 15 inch",
                unit_cost=800.0,
                selling_price=1200.0,
                holding_cost=2.0,
                stockout_penalty=50.0,
                reorder_point=20,
                max_capacity=300,
                min_order_qty=5,
                demand_pattern=DemandPattern.STABLE,
                category="electronics",
            )
        ],
        "initial_inventory": {"LAPTOP-001": 50},
        "goal": (
            "Manage inventory for 1 SKU over 30 days. "
            "Keep service level above 95% without exceeding budget."
        ),
        "events": [],
    },

    "task_medium": {
        "difficulty": TaskDifficulty.MEDIUM,
        "total_days": 60,
        "budget": 150000.0,
        "seed": 123,
        "skus": [
            SKU(
                sku_id="LAPTOP-001",
                name="Business Laptop",
                unit_cost=800.0,
                selling_price=1200.0,
                holding_cost=2.0,
                stockout_penalty=50.0,
                reorder_point=20,
                max_capacity=300,
                min_order_qty=5,
                demand_pattern=DemandPattern.STABLE,
                category="electronics",
            ),
            SKU(
                sku_id="TABLET-002",
                name="Tablet Pro",
                unit_cost=400.0,
                selling_price=650.0,
                holding_cost=1.5,
                stockout_penalty=30.0,
                reorder_point=15,
                max_capacity=400,
                min_order_qty=10,
                demand_pattern=DemandPattern.TRENDING,
                category="electronics",
            ),
            SKU(
                sku_id="CHAIR-005",
                name="Ergonomic Chair",
                unit_cost=200.0,
                selling_price=350.0,
                holding_cost=3.0,
                stockout_penalty=20.0,
                reorder_point=10,
                max_capacity=100,
                min_order_qty=5,
                demand_pattern=DemandPattern.SEASONAL,
                category="furniture",
            ),
            SKU(
                sku_id="KEYBOARD-009",
                name="Mechanical Keyboard",
                unit_cost=80.0,
                selling_price=140.0,
                holding_cost=0.5,
                stockout_penalty=10.0,
                reorder_point=25,
                max_capacity=500,
                min_order_qty=20,
                demand_pattern=DemandPattern.RANDOM,
                category="electronics",
            ),
        ],
        "initial_inventory": {
            "LAPTOP-001": 40,
            "TABLET-002": 30,
            "CHAIR-005": 15,
            "KEYBOARD-009": 60,
        },
        "goal": (
            "Manage 4 SKUs over 60 days with a $150,000 budget. "
            "Balance inventory across different demand patterns."
        ),
        "events": [],
    },

    "task_hard": {
        "difficulty": TaskDifficulty.HARD,
        "total_days": 90,
        "budget": 300000.0,
        "seed": 999,
        "skus": [
            SKU(
                sku_id="LAPTOP-001",
                name="Business Laptop",
                unit_cost=800.0,
                selling_price=1200.0,
                holding_cost=2.0,
                stockout_penalty=50.0,
                reorder_point=20,
                max_capacity=300,
                min_order_qty=5,
                demand_pattern=DemandPattern.SEASONAL,
                category="electronics",
            ),
            SKU(
                sku_id="TABLET-002",
                name="Tablet Pro",
                unit_cost=400.0,
                selling_price=650.0,
                holding_cost=1.5,
                stockout_penalty=30.0,
                reorder_point=15,
                max_capacity=400,
                min_order_qty=10,
                demand_pattern=DemandPattern.SHOCK,
                category="electronics",
            ),
            SKU(
                sku_id="CHAIR-005",
                name="Ergonomic Chair",
                unit_cost=200.0,
                selling_price=350.0,
                holding_cost=3.0,
                stockout_penalty=20.0,
                reorder_point=10,
                max_capacity=100,
                min_order_qty=5,
                demand_pattern=DemandPattern.TRENDING,
                category="furniture",
            ),
            SKU(
                sku_id="KEYBOARD-009",
                name="Mechanical Keyboard",
                unit_cost=80.0,
                selling_price=140.0,
                holding_cost=0.5,
                stockout_penalty=10.0,
                reorder_point=25,
                max_capacity=500,
                min_order_qty=20,
                demand_pattern=DemandPattern.RANDOM,
                category="electronics",
            ),
            SKU(
                sku_id="MONITOR-004",
                name="4K Monitor",
                unit_cost=350.0,
                selling_price=550.0,
                holding_cost=2.5,
                stockout_penalty=40.0,
                reorder_point=10,
                max_capacity=200,
                min_order_qty=5,
                demand_pattern=DemandPattern.TRENDING,
                category="electronics",
            ),
        ],
        "initial_inventory": {
            "LAPTOP-001": 40,
            "TABLET-002": 30,
            "CHAIR-005": 15,
            "KEYBOARD-009": 60,
            "MONITOR-004": 20,
        },
        "goal": (
            "Manage 5 SKUs over 90 days. Supplier SUP-001 goes "
            "bankrupt on day 45. Holiday demand surge on days 75-90. "
            "Maintain service level above 90%."
        ),
        "events": [
            {
                "day": 45,
                "type": "bankruptcy",
                "supplier_id": "SUP-001"
            },
            {
                "day": 75,
                "type": "demand_surge",
                "multiplier": 2.5
            },
        ],
    },
}


class SupplyChainEnvironment:
    """
    MAIN ENVIRONMENT CLASS
    Follows OpenEnv standard interface.
    """

    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {task_id}. "
                f"Choose from: {list(TASK_CONFIGS.keys())}"
            )
        self.task_id = task_id
        self.config  = TASK_CONFIGS[task_id]
        self._state: Optional[EnvironmentState] = None

    def reset(self) -> StepResult:
        """Start fresh episode"""
        config = self.config
        seed   = config["seed"]

        self._demand_gen = DemandGenerator(
            skus=config["skus"],
            seed=seed
        )

        self._supplier_mgr = SupplierManager(
            suppliers=get_default_suppliers(),
            skus=config["skus"],
            seed=seed
        )

        self._state = EnvironmentState(
            task_id=self.task_id,
            difficulty=config["difficulty"],
            current_day=1,
            total_days=config["total_days"],
            inventory=dict(config["initial_inventory"]),
            pending_orders=[],
            budget_remaining=config["budget"],
            total_budget=config["budget"],
            skus=config["skus"],
            suppliers=get_default_suppliers(),
            order_history=[],
            daily_metrics=[],
            cumulative_score=0.0,
            total_stockout_days=0,
            total_revenue=0.0,
            total_costs=0.0,
            episode_done=False,
            seed=seed,
        )

        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=SupplyChainReward(
                total_score=0.0,
                service_level=0.0,
                inventory_health=0.0,
                budget_efficiency=0.0,
                cost_efficiency=0.0,
                stockout_penalty=0.0,
                overstock_penalty=0.0,
                budget_penalty=0.0,
                stockouts_today=[],
                feedback="Episode started. Make your first decision.",
                is_critical_failure=False,
            ),
            done=False,
            info={"message": "Environment reset successfully"},
        )

    def step(self, action: RestockAction) -> StepResult:
        """Take one action — advances simulation by one day"""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.episode_done:
            raise RuntimeError("Episode done. Call reset().")

        state  = self._state
        day    = state.current_day
        errors = []

        # ── STEP 1: PROCESS ORDERS ─────────────
        new_orders = []
        order_cost = 0.0

        for item in action.orders:
            order, msg = self._supplier_mgr.place_order(
                item=item,
                current_day=day,
                budget_remaining=state.budget_remaining,
            )
            if order:
                new_orders.append(order)
                state.budget_remaining -= order.total_cost
                order_cost += order.total_cost
            else:
                errors.append(f"Order rejected: {msg}")

        state.pending_orders.extend(new_orders)
        state.order_history.extend(new_orders)

        # ── STEP 2: RECEIVE DELIVERIES ─────────
        updated_orders, deliveries = (
            self._supplier_mgr.process_daily_deliveries(
                pending_orders=state.pending_orders,
                current_day=day,
            )
        )
        state.pending_orders = [
            o for o in updated_orders
            if o.status == OrderStatus.PENDING
        ]

        for sku_id, qty in deliveries.items():
            if sku_id in state.inventory:
                sku = next(
                    s for s in state.skus
                    if s.sku_id == sku_id
                )
                state.inventory[sku_id] = min(
                    state.inventory[sku_id] + qty,
                    sku.max_capacity
                )
            else:
                state.inventory[sku_id] = qty

        # ── STEP 3 & 4: DEMAND + SALES ─────────
        demand = self._demand_gen.generate_demand(
            day=day,
            total_days=state.total_days
        )

        units_sold      = {}
        units_demanded  = {}
        stockouts       = []
        daily_revenue   = 0.0
        holding_costs   = 0.0
        stockout_losses = 0.0

        for sku in state.skus:
            sku_id    = sku.sku_id
            demanded  = demand.get(sku_id, 0)
            available = state.inventory.get(sku_id, 0)
            sold      = min(demanded, available)
            not_sold  = demanded - sold

            units_sold[sku_id]    = sold
            units_demanded[sku_id] = demanded

            state.inventory[sku_id] = available - sold

            daily_revenue   += sold * sku.selling_price
            holding_costs   += state.inventory[sku_id] * sku.holding_cost

            if not_sold > 0:
                stockouts.append(sku_id)
                stockout_losses += not_sold * sku.stockout_penalty

        state.total_revenue += daily_revenue
        state.total_costs   += holding_costs + order_cost

        daily_metric = DailyMetrics(
            units_sold=units_sold,
            units_demanded=units_demanded,
            stockouts=stockouts,
            orders_delivered=list(deliveries.keys()),
            revenue=daily_revenue,
            holding_costs=holding_costs,
            stockout_losses=stockout_losses,
        )
        state.daily_metrics.append(daily_metric)

        if stockouts:
            state.total_stockout_days += 1

        # ── STEP 5: CALCULATE REWARD ───────────
        reward = self._calculate_reward(
            daily_metric=daily_metric,
            order_cost=order_cost,
            errors=errors,
        )
        state.cumulative_score += reward.total_score

        # ── STEP 6: TRIGGER EVENTS ────────────
        self._process_events(day)

        # ── STEP 7: CHECK IF DONE ─────────────
        state.current_day += 1
        done = state.current_day > state.total_days

        if done:
            state.episode_done = True
            grader       = get_grader(self.task_id)
            final_score  = grader(state)
            reward.total_score = final_score
            reward.feedback    = (
                f"Episode complete! Final score: {final_score:.4f}"
            )

        observation = self._build_observation()
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info={
                "day":        day,
                "errors":     errors,
                "deliveries": deliveries,
                "stockouts":  stockouts,
                "order_cost": order_cost,
            },
        )

    def state(self) -> EnvironmentState:
        """Returns current internal state"""
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def _build_observation(self) -> WarehouseObservation:
        """Builds observation from current state"""
        state    = self._state
        forecast = self._demand_gen.generate_forecast(
            day=state.current_day,
            total_days=state.total_days,
        )
        return WarehouseObservation(
            current_day=state.current_day,
            total_days=state.total_days,
            days_remaining=state.total_days - state.current_day + 1,
            inventory=dict(state.inventory),
            pending_orders=list(state.pending_orders),
            budget_remaining=state.budget_remaining,
            total_budget=state.total_budget,
            demand_forecast=forecast,
            suppliers=list(state.suppliers),
            skus=list(state.skus),
            yesterday_metrics=(
                state.daily_metrics[-1]
                if state.daily_metrics else None
            ),
            task_id=self.task_id,
            goal=self.config["goal"],
            step_number=state.current_day,
            cumulative_score=state.cumulative_score,
        )

    def _calculate_reward(
        self,
        daily_metric: DailyMetrics,
        order_cost: float,
        errors: List[str],
    ) -> SupplyChainReward:
        """
        IMPROVED REWARD FUNCTION
        Meaningful signal every day.
        Partial credit for partial success.
        """
        state = self._state

        # ── SERVICE LEVEL ──────────────────────
        total_demanded = sum(daily_metric.units_demanded.values())
        total_sold     = sum(daily_metric.units_sold.values())

        if total_demanded == 0:
            service_level = 1.0
        else:
            service_level = min(1.0, total_sold / total_demanded)

        # ── INVENTORY HEALTH ───────────────────
        health_scores = []
        overstock_pen = 0.0

        for sku in state.skus:
            sku_id  = sku.sku_id
            inv     = state.inventory.get(sku_id, 0)
            max_cap = sku.max_capacity
            reorder = sku.reorder_point

            if inv == 0:
                health_scores.append(0.0)
            elif inv < reorder:
                health_scores.append(0.4)
            elif inv < reorder * 2:
                health_scores.append(0.9)
            elif inv < max_cap * 0.8:
                health_scores.append(1.0)
            else:
                health_scores.append(0.7)
                overstock_pen += 0.03

        inv_health = (
            sum(health_scores) / len(health_scores)
            if health_scores else 1.0
        )

        # ── STOCKOUT PENALTY ───────────────────
        stockout_count = len(daily_metric.stockouts)
        total_skus     = max(1, len(state.skus))
        stockout_pen   = (stockout_count / total_skus) * 0.3

        # ── BUDGET EFFICIENCY ──────────────────
        budget_ratio = (
            state.budget_remaining / state.total_budget
            if state.total_budget > 0 else 1.0
        )
        budget_pen        = 0.2 if state.budget_remaining < 0 else 0.0
        budget_efficiency = min(1.0, budget_ratio * 1.2)

        # ── ORDER REWARD ───────────────────────
        order_reward = 0.05 if order_cost > 0 else 0.0

        # ── COMBINE ────────────────────────────
        total = (
            service_level     * 0.50 +
            inv_health        * 0.25 +
            budget_efficiency * 0.10 +
            order_reward           -
            stockout_pen           -
            overstock_pen          -
            budget_pen
        )
        total = max(0.0, min(1.0, total))

        # ── FEEDBACK ───────────────────────────
        parts = []
        if service_level == 1.0:
            parts.append("Perfect service level")
        elif service_level >= 0.8:
            parts.append(f"Good service: {service_level:.0%}")
        else:
            parts.append(f"Low service: {service_level:.0%}")

        if daily_metric.stockouts:
            parts.append(f"Stockouts: {daily_metric.stockouts}")
        if order_cost > 0:
            parts.append(f"Spent: ${order_cost:,.0f}")
        if errors:
            parts.append(f"Errors: {len(errors)}")

        feedback = " | ".join(parts) if parts else "Normal day"

        return SupplyChainReward(
            total_score=round(total, 4),
            service_level=round(service_level, 4),
            inventory_health=round(inv_health, 4),
            budget_efficiency=round(budget_efficiency, 4),
            cost_efficiency=round(
                1.0 - overstock_pen - stockout_pen, 4
            ),
            stockout_penalty=round(stockout_pen, 4),
            overstock_penalty=round(overstock_pen, 4),
            budget_penalty=round(budget_pen, 4),
            stockouts_today=daily_metric.stockouts,
            feedback=feedback,
            is_critical_failure=stockout_count >= len(state.skus),
        )

    def _process_events(self, day: int):
        """Triggers scheduled events"""
        for event in self.config.get("events", []):
            if event["day"] == day:
                if event["type"] == "bankruptcy":
                    self._supplier_mgr.trigger_bankruptcy(
                        event["supplier_id"]
                    )
                elif event["type"] == "demand_surge":
                    for sku in self._state.skus:
                        base = self._demand_gen.base_demands.get(
                            sku.sku_id, 10
                        )
                        self._demand_gen.base_demands[sku.sku_id] = (
                            base * event["multiplier"]
                        )