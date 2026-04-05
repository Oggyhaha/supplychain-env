# env/suppliers.py
# ═══════════════════════════════════════════════
# SUPPLIER SYSTEM
# Fixed version - proper SKU coverage for all tasks
# ═══════════════════════════════════════════════

import random
from typing import Dict, List, Optional, Tuple
from env.models import (
    Supplier, PurchaseOrder, OrderItem,
    OrderStatus, SupplierStatus, SKU
)


def get_default_suppliers() -> List[Supplier]:
    """
    Returns suppliers for all tasks.
    Every SKU has at least 2 suppliers.
    Min order values are realistic for budgets.
    """
    return [
        Supplier(
            supplier_id="SUP-001",
            name="TechCorp Industries",
            lead_time_days=3,
            reliability=0.95,
            price_factor=1.0,
            skus_supplied=[
                "LAPTOP-001",
                "TABLET-002",
                "MONITOR-004",
                "KEYBOARD-009",
                "CHAIR-005",
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=100.0,
        ),
        Supplier(
            supplier_id="SUP-002",
            name="QuickShip Express",
            lead_time_days=1,
            reliability=0.90,
            price_factor=1.20,
            skus_supplied=[
                "LAPTOP-001",
                "TABLET-002",
                "CHAIR-005",
                "KEYBOARD-009",
                "MONITOR-004",
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=100.0,
        ),
        Supplier(
            supplier_id="SUP-003",
            name="BudgetParts Co",
            lead_time_days=5,
            reliability=0.82,
            price_factor=0.88,
            skus_supplied=[
                "LAPTOP-001",
                "TABLET-002",
                "KEYBOARD-009",
                "MONITOR-004",
                "CHAIR-005",
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=100.0,
        ),
        Supplier(
            supplier_id="SUP-004",
            name="GlobalStock Ltd",
            lead_time_days=4,
            reliability=0.88,
            price_factor=0.95,
            skus_supplied=[
                "CHAIR-005",
                "MONITOR-004",
                "KEYBOARD-009",
                "LAPTOP-001",
                "TABLET-002",
            ],
            status=SupplierStatus.ACTIVE,
            min_order_value=100.0,
        ),
    ]


class SupplierManager:
    """Manages all supplier interactions"""

    def __init__(
        self,
        suppliers: List[Supplier],
        skus: List[SKU],
        seed: Optional[int] = None
    ):
        self.suppliers: Dict[str, Supplier] = {
            s.supplier_id: s for s in suppliers
        }
        self.skus: Dict[str, SKU] = {
            s.sku_id: s for s in skus
        }
        self.random          = random.Random(seed)
        self._order_counter  = 1

    def place_order(
        self,
        item: OrderItem,
        current_day: int,
        budget_remaining: float
    ) -> Tuple[Optional[PurchaseOrder], str]:
        """Places a purchase order with validation"""

        supplier = self.suppliers.get(item.supplier_id)
        if not supplier:
            return None, f"Supplier {item.supplier_id} not found"

        if supplier.status == SupplierStatus.BANKRUPT:
            return None, f"{supplier.name} is bankrupt"

        if supplier.status == SupplierStatus.INACTIVE:
            return None, f"{supplier.name} is inactive"

        if item.sku_id not in supplier.skus_supplied:
            return None, f"{supplier.name} does not supply {item.sku_id}"

        sku = self.skus.get(item.sku_id)
        if not sku:
            return None, f"SKU {item.sku_id} not found"

        if item.quantity < sku.min_order_qty:
            return None, (
                f"Min order for {item.sku_id} is {sku.min_order_qty}"
            )

        unit_cost  = sku.unit_cost * supplier.price_factor
        total_cost = unit_cost * item.quantity

        if total_cost < supplier.min_order_value:
            return None, (
                f"Order ${total_cost:.0f} below "
                f"min ${supplier.min_order_value:.0f}"
            )

        if total_cost > budget_remaining:
            return None, (
                f"Need ${total_cost:.0f}, "
                f"have ${budget_remaining:.0f}"
            )

        lead_time    = self._calculate_lead_time(supplier)
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
        """Lead time with random delays based on reliability"""
        if self.random.random() <= supplier.reliability:
            return supplier.lead_time_days
        else:
            delay = self.random.randint(1, 2)
            return supplier.lead_time_days + delay

    def process_daily_deliveries(
        self,
        pending_orders: List[PurchaseOrder],
        current_day: int
    ) -> Tuple[List[PurchaseOrder], Dict[str, int]]:
        """Processes arriving orders for today"""
        deliveries:    Dict[str, int] = {}
        updated_orders = []

        for order in pending_orders:
            if (order.expected_day <= current_day
                    and order.status == OrderStatus.PENDING):
                order.status   = OrderStatus.DELIVERED
                order.actual_day = current_day

                if order.sku_id in deliveries:
                    deliveries[order.sku_id] += order.quantity
                else:
                    deliveries[order.sku_id] = order.quantity

            updated_orders.append(order)

        return updated_orders, deliveries

    def trigger_bankruptcy(self, supplier_id: str) -> bool:
        """Makes supplier go bankrupt"""
        if supplier_id in self.suppliers:
            self.suppliers[supplier_id].status = SupplierStatus.BANKRUPT
            return True
        return False

    def trigger_delay(self, supplier_id: str, extra_days: int) -> bool:
        """Adds delay to supplier"""
        if supplier_id in self.suppliers:
            self.suppliers[supplier_id].lead_time_days += extra_days
            self.suppliers[supplier_id].status = SupplierStatus.DELAYED
            return True
        return False

    def get_active_suppliers(self) -> List[Supplier]:
        """Returns active suppliers only"""
        return [
            s for s in self.suppliers.values()
            if s.status == SupplierStatus.ACTIVE
        ]

    def get_suppliers_for_sku(self, sku_id: str) -> List[Supplier]:
        """Returns active suppliers for a specific SKU"""
        return [
            s for s in self.suppliers.values()
            if sku_id in s.skus_supplied
            and s.status == SupplierStatus.ACTIVE
        ]