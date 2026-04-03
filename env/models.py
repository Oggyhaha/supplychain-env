# env/models.py
# ═══════════════════════════════════════════════
# DATA BLUEPRINTS FOR SUPPLYCHAIN-ENV
#
# Everything in our system has a defined shape.
# Pydantic enforces these shapes automatically.
# ═══════════════════════════════════════════════

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS — Fixed choice lists (like dropdowns)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class DemandPattern(str, Enum):
    """
    Types of customer demand our environment simulates.
    Each pattern behaves differently — agent must adapt.
    """
    STABLE   = "stable"    # Same amount every day
    SEASONAL = "seasonal"  # Spikes at certain times
    TRENDING = "trending"  # Slowly increasing over time
    RANDOM   = "random"    # Unpredictable each day
    SHOCK    = "shock"     # Sudden unexpected spike


class SupplierStatus(str, Enum):
    """Current status of a supplier"""
    ACTIVE    = "active"      # Working normally
    DELAYED   = "delayed"     # Taking longer than usual
    BANKRUPT  = "bankrupt"    # Gone out of business
    INACTIVE  = "inactive"    # Not available right now


class OrderStatus(str, Enum):
    """Status of a purchase order we placed"""
    PENDING   = "pending"     # Order placed, not shipped yet
    SHIPPED   = "shipped"     # On the way
    DELIVERED = "delivered"   # Arrived at warehouse
    CANCELLED = "cancelled"   # Order was cancelled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE DATA MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SKU(BaseModel):
    """
    SKU = Stock Keeping Unit = one specific product.

    Example:
        sku_id:           "LAPTOP-001"
        name:             "Business Laptop 15 inch"
        unit_cost:        800.0   (what WE pay supplier)
        selling_price:    1200.0  (what customer pays us)
        holding_cost:     2.0     (per unit per day storage)
        stockout_penalty: 50.0    (lost profit per missed sale)
        reorder_point:    20      (order when stock hits this)
        max_capacity:     500     (max units warehouse can hold)
    """
    sku_id:            str
    name:              str
    unit_cost:         float = Field(description="Cost per unit from supplier ($)")
    selling_price:     float = Field(description="Revenue per unit sold ($)")
    holding_cost:      float = Field(description="Storage cost per unit per day ($)")
    stockout_penalty:  float = Field(description="Penalty per unit of unmet demand ($)")
    reorder_point:     int   = Field(description="Order when inventory drops to this level")
    max_capacity:      int   = Field(description="Maximum units warehouse can store")
    min_order_qty:     int   = Field(default=1, description="Minimum order quantity")
    demand_pattern:    DemandPattern = DemandPattern.STABLE
    category:          str   = Field(default="general")


class Supplier(BaseModel):
    """
    A company we buy products from.

    Example:
        supplier_id:   "SUP-001"
        name:          "TechCorp Industries"
        lead_time:     3       (3 days to deliver)
        reliability:   0.95    (95% chance of on-time delivery)
        price_factor:  1.0     (1.0 = normal price, 1.2 = 20% more expensive)
        skus_supplied: ["LAPTOP-001", "TABLET-002"]
    """
    supplier_id:    str
    name:           str
    lead_time_days: int   = Field(description="Days from order to delivery")
    reliability:    float = Field(description="Probability of on-time delivery 0-1")
    price_factor:   float = Field(default=1.0, description="Price multiplier vs base cost")
    skus_supplied:  List[str] = Field(description="List of SKU IDs this supplier sells")
    status:         SupplierStatus = SupplierStatus.ACTIVE
    min_order_value: float = Field(default=0.0, description="Minimum order amount ($)")


class PurchaseOrder(BaseModel):
    """
    An order WE place to a supplier to restock inventory.

    Lifecycle: PENDING → SHIPPED → DELIVERED
    """
    order_id:       str
    sku_id:         str
    supplier_id:    str
    quantity:       int
    unit_cost:      float
    total_cost:     float
    order_day:      int   = Field(description="Which simulation day we placed order")
    expected_day:   int   = Field(description="Which day we expect delivery")
    actual_day:     Optional[int] = None
    status:         OrderStatus = OrderStatus.PENDING


class OrderItem(BaseModel):
    """
    One item in an agent's restock action.
    Agent says: "I want to order THIS product from THIS supplier"
    """
    sku_id:      str
    supplier_id: str
    quantity:    int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBSERVATION — What agent SEES each day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DailyMetrics(BaseModel):
    """
    Yesterday's performance summary.
    Helps agent learn from recent history.
    """
    units_sold:         Dict[str, int]   # SKU → units sold
    units_demanded:     Dict[str, int]   # SKU → units customers wanted
    stockouts:          List[str]        # SKUs that ran out
    orders_delivered:   List[str]        # Order IDs that arrived
    revenue:            float            # Money earned yesterday
    holding_costs:      float            # Storage costs yesterday
    stockout_losses:    float            # Lost revenue from stockouts


class WarehouseObservation(BaseModel):
    """
    WHAT THE AGENT SEES EVERY DAY.

    This is the agent's "eyes" — everything it needs
    to make a smart restock decision.
    """
    # Time info
    current_day:        int
    total_days:         int
    days_remaining:     int

    # Current inventory
    inventory:          Dict[str, int]
    # Example: {"LAPTOP-001": 45, "TABLET-002": 120}

    # Orders placed but not yet arrived
    pending_orders:     List[PurchaseOrder]

    # Money situation
    budget_remaining:   float
    total_budget:       float

    # Demand forecast for next 7 days
    demand_forecast:    Dict[str, List[float]]
    # Example: {"LAPTOP-001": [10,10,12,10,10,15,10]}

    # Available suppliers and their current status
    suppliers:          List[Supplier]

    # SKU catalog with costs and constraints
    skus:               List[SKU]

    # Yesterday's performance
    yesterday_metrics:  Optional[DailyMetrics]

    # Task context
    task_id:            str
    goal:               str

    # Episode info
    step_number:        int
    cumulative_score:   float


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACTION — What agent DOES each day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RestockAction(BaseModel):
    """
    THE AGENT'S DECISION FOR THE DAY.

    Agent can place multiple orders in one day.
    Or place no orders (empty list = wait today).

    Example:
        orders = [
            OrderItem(sku_id="LAPTOP-001",
                     supplier_id="SUP-001",
                     quantity=100),
            OrderItem(sku_id="TABLET-002",
                     supplier_id="SUP-002",
                     quantity=50)
        ]
        reasoning = "Laptop stock low, tablet has upcoming demand spike"
    """
    orders:    List[OrderItem] = Field(
        default=[],
        description="List of items to order. Empty = no orders today."
    )
    reasoning: str = Field(
        default="",
        description="Why agent made this decision (for logging)"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REWARD — Score agent receives each day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SupplyChainReward(BaseModel):
    """
    DETAILED SCORE BREAKDOWN.

    Not just a number — tells agent exactly WHY
    it got this score. Helps learning.
    """
    # Overall score this step (0.0 to 1.0)
    total_score:         float

    # Component scores
    service_level:       float  # % of demand fulfilled (0-1)
    inventory_health:    float  # How well inventory is balanced
    budget_efficiency:   float  # Smart use of budget
    cost_efficiency:     float  # Minimizing holding + stockout costs

    # Penalties (negative values)
    stockout_penalty:    float  # Lost sales penalty
    overstock_penalty:   float  # Too much inventory cost
    budget_penalty:      float  # Overspending penalty

    # Context
    stockouts_today:     List[str]  # Which SKUs stocked out
    feedback:            str        # Human readable explanation
    is_critical_failure: bool       # True if very bad decision


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP RESULT — What env.step() returns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StepResult(BaseModel):
    """
    Standard OpenEnv return format from step().
    """
    observation: WarehouseObservation
    reward:      SupplyChainReward
    done:        bool   # Is episode finished?
    info:        Dict   # Extra debugging info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENVIRONMENT STATE — Internal tracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EnvironmentState(BaseModel):
    """
    Everything the environment tracks internally.
    Returned by state() method.
    """
    task_id:              str
    difficulty:           TaskDifficulty
    current_day:          int
    total_days:           int
    inventory:            Dict[str, int]
    pending_orders:       List[PurchaseOrder]
    budget_remaining:     float
    total_budget:         float
    skus:                 List[SKU]
    suppliers:            List[Supplier]
    order_history:        List[PurchaseOrder]
    daily_metrics:        List[DailyMetrics]
    cumulative_score:     float
    total_stockout_days:  int
    total_revenue:        float
    total_costs:          float
    episode_done:         bool
    seed:                 Optional[int] = None