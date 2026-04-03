# ## FILE 3 — `env/suppliers.py`

# ### What Is It?
# ```
# Simulates companies we buy products from.
# Every real warehouse has multiple suppliers.

# Example:
# TechCorp    → sells laptops, tablets
#              lead time: 3 days, reliable
# QuickShip   → sells everything
#              lead time: 1 day, expensive
# BudgetParts → sells laptops cheap
#              lead time: 7 days, unreliable
# ```

# ### Why Is It Important?
# ```
# Agent must learn:
# ✅ Which supplier is cheapest?
# ✅ Which supplier is fastest?
# ✅ What if supplier goes bankrupt? (Task 3)
# ✅ When to pay more for faster delivery?
# ```

# ### How It Works
# ```
# Agent orders 100 laptops from TechCorp
# TechCorp says "ok, arriving in 3 days"
# Day 3 comes → 95 laptops arrive (5% delay loss)
# Agent updates inventory
# Open env/suppliers.py and paste:
# python# env/suppliers.py
# # ═══════════════════════════════════════════════
# # SUPPLIER SYSTEM
# #
# # Simulates real supplier behavior:
# # - Processing orders
# # - Random delays
# # - Price changes
# # - Bankruptcy events (Task 3)
# # ═══════════════════════════════════════════════

import random
from typing import Dict, List, Optional, Tuple
from env.models import (
    Supplier, PurchaseOrder, OrderItem,
    OrderStatus, SupplierStatus, SKU
)


# ── DEFAULT SUPPLIERS ──────────────────────────
# Pre-built suppliers for our 3 tasks

def get_default_suppliers() -> List[Supplier]:
    """
    Returns the standard set of suppliers
    used across all tasks.
    """
    return [
        Supplier(
            supplier_id="SUP-001",
            name="TechCorp Industries",
            lead_time_days=3,
            reliability=0.95,
            price_factor=1.0,
            skus_supplied=[
                "LAPTOP-001", "TABLET-002",
                "PHONE-003", "MONITOR-004"
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=500.0,
        ),
        Supplier(
            supplier_id="SUP-002",
            name="QuickShip Express",
            lead_time_days=1,
            reliability=0.90,
            price_factor=1.25,   # 25% more expensive
            skus_supplied=[
                "LAPTOP-001", "CHAIR-005",
                "DESK-006", "LAMP-007"
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=200.0,
        ),
        Supplier(
            supplier_id="SUP-003",
            name="BudgetParts Co",
            lead_time_days=7,
            reliability=0.80,    # Less reliable
            price_factor=0.85,   # 15% cheaper
            skus_supplied=[
                "LAPTOP-001", "TABLET-002",
                "HEADPHONE-008", "KEYBOARD-009"
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=1000.0,
        ),
        Supplier(
            supplier_id="SUP-004",
            name="GlobalStock Ltd",
            lead_time_days=5,
            reliability=0.88,
            price_factor=0.95,
            skus_supplied=[
                "CHAIR-005", "DESK-006",
                "MONITOR-004", "KEYBOARD-009",
                "HEADPHONE-008", "LAMP-007"
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=750.0,
        ),
    ]


class SupplierManager:
    """
    Manages all supplier interactions.

    Handles:
    - Validating orders (can supplier fulfill this?)
    - Processing orders (calculating delivery day)
    - Simulating delays (reliability factor)
    - Updating order status daily
    - Supplier events (bankruptcy, price changes)
    """

    def __init__(
        self,
        suppliers: List[Supplier],
        skus: List[SKU],
        seed: Optional[int] = None
    ):
        # Store suppliers by ID for quick lookup
        self.suppliers: Dict[str, Supplier] = {
            s.supplier_id: s for s in suppliers
        }
        # Store SKUs by ID for quick lookup
        self.skus: Dict[str, SKU] = {
            s.sku_id: s for s in skus
        }
        self.random = random.Random(seed)

        # Order counter for generating unique IDs
        self._order_counter = 1

    def place_order(
        self,
        item: OrderItem,
        current_day: int,
        budget_remaining: float
    ) -> Tuple[Optional[PurchaseOrder], str]:
        """
        Places a purchase order with a supplier.

        Returns:
            (PurchaseOrder, "success") if valid
            (None, "error message") if invalid

        All the validation logic lives here.
        """
        # ── VALIDATION ─────────────────────────

        # Check supplier exists
        supplier = self.suppliers.get(item.supplier_id)
        if not supplier:
            return None, f"Supplier {item.supplier_id} not found"

        # Check supplier is active
        if supplier.status == SupplierStatus.BANKRUPT:
            return None, f"{supplier.name} is bankrupt - find another supplier"

        if supplier.status == SupplierStatus.INACTIVE:
            return None, f"{supplier.name} is currently inactive"

        # Check supplier sells this SKU
        if item.sku_id not in supplier.skus_supplied:
            return None, (
                f"{supplier.name} does not supply {item.sku_id}"
            )

        # Check SKU exists
        sku = self.skus.get(item.sku_id)
        if not sku:
            return None, f"SKU {item.sku_id} not found"

        # Check minimum order quantity
        if item.quantity < sku.min_order_qty:
            return None, (
                f"Minimum order for {item.sku_id} is "
                f"{sku.min_order_qty} units"
            )

        # Calculate cost
        unit_cost = sku.unit_cost * supplier.price_factor
        total_cost = unit_cost * item.quantity

        # Check minimum order value
        if total_cost < supplier.min_order_value:
            return None, (
                f"Order value ${total_cost:.2f} below "
                f"{supplier.name} minimum ${supplier.min_order_value:.2f}"
            )

        # Check budget
        if total_cost > budget_remaining:
            return None, (
                f"Insufficient budget. Need ${total_cost:.2f}, "
                f"have ${budget_remaining:.2f}"
            )

        # ── CREATE ORDER ───────────────────────

        # Calculate delivery day with possible delay
        lead_time = self._calculate_lead_time(supplier)
        expected_day = current_day + lead_time

        order = PurchaseOrder(
            order_id=f"PO-{self._order_counter:04d}",
            sku_id=item.sku_id,
            supplier_id=item.supplier_id,
            quantity=item.quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            order_day=current_day,
            expected_day=expected_day,
            status=OrderStatus.PENDING,
        )

        self._order_counter += 1
        return order, "success"

    def _calculate_lead_time(self, supplier: Supplier) -> int:
        """
        Calculates actual lead time with possible delays.

        reliability=0.95 means:
        95% of the time → normal lead time
        5% of the time  → delayed by 1-3 extra days

        Simulates real supplier unpredictability.
        """
        base_lead_time = supplier.lead_time_days

        # Check if delivery is on time
        if self.random.random() <= supplier.reliability:
            # On time!
            return base_lead_time
        else:
            # Delayed! Add 1-3 extra days
            delay = self.random.randint(1, 3)
            return base_lead_time + delay

    def process_daily_deliveries(
        self,
        pending_orders: List[PurchaseOrder],
        current_day: int
    ) -> Tuple[List[PurchaseOrder], Dict[str, int]]:
        """
        Checks which orders arrive today.
        Called every day by environment.

        Returns:
            updated_orders: all orders with updated status
            deliveries: {sku_id: quantity} that arrived today
        """
        deliveries: Dict[str, int] = {}
        updated_orders = []

        for order in pending_orders:
            if (order.expected_day <= current_day and
                    order.status == OrderStatus.PENDING):
                # This order arrives today!
                order.status = OrderStatus.DELIVERED
                order.actual_day = current_day

                # Add to deliveries
                if order.sku_id in deliveries:
                    deliveries[order.sku_id] += order.quantity
                else:
                    deliveries[order.sku_id] = order.quantity

            updated_orders.append(order)

        return updated_orders, deliveries

    def trigger_bankruptcy(self, supplier_id: str) -> bool:
        """
        Makes a supplier go bankrupt.
        Used in Task 3 (hard difficulty).

        All pending orders from this supplier
        are automatically cancelled.
        """
        if supplier_id in self.suppliers:
            self.suppliers[supplier_id].status = (
                SupplierStatus.BANKRUPT
            )
            return True
        return False

    def trigger_delay(
        self,
        supplier_id: str,
        extra_days: int
    ) -> bool:
        """
        Adds extra delay to a supplier.
        Used to simulate supply chain disruptions.
        """
        if supplier_id in self.suppliers:
            self.suppliers[supplier_id].lead_time_days += extra_days
            self.suppliers[supplier_id].status = (
                SupplierStatus.DELAYED
            )
            return True
        return False

    def get_active_suppliers(self) -> List[Supplier]:
        """Returns only suppliers that are currently active"""
        return [
            s for s in self.suppliers.values()
            if s.status == SupplierStatus.ACTIVE
        ]

    def get_suppliers_for_sku(
        self,
        sku_id: str
    ) -> List[Supplier]:
        """Returns all active suppliers that sell a specific SKU"""
        return [
            s for s in self.suppliers.values()
            if sku_id in s.skus_supplied
            and s.status == SupplierStatus.ACTIVE
        ]