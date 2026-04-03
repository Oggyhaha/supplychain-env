# env/demand_generator.py
# ═══════════════════════════════════════════════
# CUSTOMER DEMAND SIMULATOR
#
# Generates realistic daily demand for each SKU.
# Different patterns simulate real market behavior.
# ═══════════════════════════════════════════════

import random
import math
from typing import Dict, List, Optional
from env.models import SKU, DemandPattern


class DemandGenerator:
    """
    Generates daily customer demand for each SKU.

    Think of this as simulating customers walking
    into a store and buying products every day.

    Each SKU can have different demand patterns:
    - STABLE:   same amount every day
    - SEASONAL: more in some months, less in others
    - TRENDING: gradually increasing over time
    - RANDOM:   unpredictable each day
    - SHOCK:    sudden unexpected spike
    """

    def __init__(self, skus: List[SKU], seed: Optional[int] = None):
        """
        skus: list of products we're tracking demand for
        seed: random seed for reproducibility
              same seed = same demand every run
              important for hackathon graders!
        """
        self.skus = {sku.sku_id: sku for sku in skus}
        self.seed = seed
        self.random = random.Random(seed)

        # Base demand per SKU (units per day on average)
        self.base_demands = self._initialize_base_demands()

        # Track shock events so they don't repeat too often
        self.shock_days: Dict[str, List[int]] = {
            sku.sku_id: [] for sku in skus
        }

    def _initialize_base_demands(self) -> Dict[str, float]:
        """
        Set base daily demand for each SKU.
        This is the "normal" amount sold on a regular day.
        """
        base = {}
        for sku_id, sku in self.skus.items():
            # Base demand derived from SKU category
            if sku.category == "electronics":
                base[sku_id] = self.random.uniform(8, 15)
            elif sku.category == "consumables":
                base[sku_id] = self.random.uniform(20, 50)
            elif sku.category == "furniture":
                base[sku_id] = self.random.uniform(2, 8)
            elif sku.category == "clothing":
                base[sku_id] = self.random.uniform(15, 30)
            else:
                base[sku_id] = self.random.uniform(10, 20)
        return base

    def generate_demand(
        self,
        day: int,
        total_days: int
    ) -> Dict[str, int]:
        """
        MAIN FUNCTION — generates demand for ALL SKUs for one day.

        day:        current simulation day (1, 2, 3...)
        total_days: total days in episode (30, 60, or 90)

        Returns dict like:
        {"LAPTOP-001": 12, "TABLET-002": 8, "CHAIR-003": 3}
        """
        demands = {}

        for sku_id, sku in self.skus.items():
            base = self.base_demands[sku_id]

            # Apply pattern to get today's demand
            if sku.demand_pattern == DemandPattern.STABLE:
                demand = self._stable_demand(base, day)

            elif sku.demand_pattern == DemandPattern.SEASONAL:
                demand = self._seasonal_demand(base, day, total_days)

            elif sku.demand_pattern == DemandPattern.TRENDING:
                demand = self._trending_demand(base, day, total_days)

            elif sku.demand_pattern == DemandPattern.RANDOM:
                demand = self._random_demand(base, day)

            elif sku.demand_pattern == DemandPattern.SHOCK:
                demand = self._shock_demand(base, day, sku_id)

            else:
                demand = self._stable_demand(base, day)

            # Always round to whole number (can't sell half a laptop)
            # Always at least 0 (can't have negative demand)
            demands[sku_id] = max(0, round(demand))

        return demands

    def generate_forecast(
        self,
        day: int,
        total_days: int,
        forecast_days: int = 7
    ) -> Dict[str, List[float]]:
        """
        Generates demand FORECAST for next N days.
        This is what the agent sees to plan ahead.

        Like a weather forecast but for sales.
        Not 100% accurate — has some uncertainty.

        Returns:
        {
          "LAPTOP-001": [10.2, 11.5, 9.8, 12.1, 10.0, 10.5, 11.2],
          "TABLET-002": [8.1,  7.9,  8.5, 8.0,  7.8,  8.2,  8.0]
        }
        """
        forecast = {}

        for sku_id, sku in self.skus.items():
            sku_forecast = []
            base = self.base_demands[sku_id]

            for future_day in range(day + 1, day + forecast_days + 1):
                # Get base forecast for that day
                if sku.demand_pattern == DemandPattern.STABLE:
                    pred = self._stable_demand(base, future_day)
                elif sku.demand_pattern == DemandPattern.SEASONAL:
                    pred = self._seasonal_demand(
                        base, future_day, total_days
                    )
                elif sku.demand_pattern == DemandPattern.TRENDING:
                    pred = self._trending_demand(
                        base, future_day, total_days
                    )
                else:
                    pred = base

                # Add forecast uncertainty (±15%)
                # Real forecasts are never perfect!
                noise = self.random.uniform(0.85, 1.15)
                sku_forecast.append(round(pred * noise, 1))

            forecast[sku_id] = sku_forecast

        return forecast

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEMAND PATTERN FUNCTIONS
    # Each function returns demand for one day
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _stable_demand(self, base: float, day: int) -> float:
        """
        STABLE: roughly same amount every day.
        Small random noise to make it realistic.

        Example: Office supplies, basic food items
        Graph:   ─────────────────────── (flat line)
        """
        # Small noise: ±10% of base
        noise = self.random.gauss(0, base * 0.1)
        return max(0, base + noise)

    def _seasonal_demand(
        self,
        base: float,
        day: int,
        total_days: int
    ) -> float:
        """
        SEASONAL: follows a wave pattern.
        High demand in some periods, low in others.

        Example: Holiday decorations, winter clothing
        Graph:   ╭───╮           ╭───╮
                ╯    ╰───────────╯    ╰──
        """
        # Create a sine wave over the episode
        # sin() goes from -1 to +1 smoothly
        progress = (day / total_days) * 2 * math.pi
        seasonal_factor = 1 + 0.6 * math.sin(progress)

        # Add small noise
        noise = self.random.gauss(0, base * 0.08)
        return max(0, base * seasonal_factor + noise)

    def _trending_demand(
        self,
        base: float,
        day: int,
        total_days: int
    ) -> float:
        """
        TRENDING: slowly grows over time.
        Starts low, ends high.

        Example: New product gaining popularity
        Graph:        ╱
                    ╱
                  ╱
                ╱──────────────────
        """
        # Linear growth: starts at 50% of base, ends at 150%
        growth_factor = 0.5 + (day / total_days)
        noise = self.random.gauss(0, base * 0.1)
        return max(0, base * growth_factor + noise)

    def _random_demand(self, base: float, day: int) -> float:
        """
        RANDOM: completely unpredictable each day.
        High variance — hard for agent to forecast.

        Example: Viral products, trending items
        Graph:  │ │   │  │   │ │  │    │
                ─────────────────────────
        """
        # Random between 20% and 200% of base
        multiplier = self.random.uniform(0.2, 2.0)
        return max(0, base * multiplier)

    def _shock_demand(
        self,
        base: float,
        day: int,
        sku_id: str
    ) -> float:
        """
        SHOCK: normal most days, sudden huge spike occasionally.
        The spike is unexpected — tests agent's resilience.

        Example: Product goes viral, emergency supply need
        Graph:               │
                             │
                ─────────────┘└──────────
        """
        # 5% chance of shock event each day
        if self.random.random() < 0.05:
            # Shock! Demand is 5x to 10x normal
            shock_multiplier = self.random.uniform(5, 10)
            self.shock_days[sku_id].append(day)
            return base * shock_multiplier

        # Normal day — small noise around base
        noise = self.random.gauss(0, base * 0.1)
        return max(0, base + noise)

    def get_shock_days(self, sku_id: str) -> List[int]:
        """Returns list of days when shocks occurred for a SKU"""
        return self.shock_days.get(sku_id, [])