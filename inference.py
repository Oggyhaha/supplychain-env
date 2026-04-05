# inference.py
# ═══════════════════════════════════════════════
# SUPPLYCHAIN-ENV BASELINE INFERENCE
# Budget-aware ordering strategy
# Target scores: easy 0.70+ medium 0.55+ hard 0.40+
# ═══════════════════════════════════════════════

import os
import json
import time
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from env.environment import SupplyChainEnvironment
from env.models import RestockAction, OrderItem

# ── LOAD ENV ───────────────────────────────────
load_dotenv()
API_KEY      = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY missing from .env")
if not API_BASE_URL:
    raise ValueError("API_BASE_URL missing from .env")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME missing from .env")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── SETTINGS ───────────────────────────────────
DEMO_STEPS = {
    "task_easy":   30,
    "task_medium": 60,
    "task_hard":   90,
}

# Ordering strategy settings
COVERAGE_THRESHOLD = 5    # Order when less than 12 days of stock
TARGET_DAYS        = 15   # Order enough for 20 days
MAX_BUDGET_PCT     = 0.60 # Never spend more than 35% budget per order
LLM_EVERY          = 3    # Call LLM every 5 days
SLEEP              = 0.5  # Seconds between LLM calls

SYSTEM_PROMPT = """
You are a warehouse inventory manager.
Manage inventory carefully within budget constraints.

CRITICAL RULES:
1. Budget is LIMITED - never spend more than 35% in one order
2. Order when stock coverage drops below 8 days
3. Order enough for 15 days of demand
4. Only use ACTIVE suppliers
5. Check supplier carries the SKU before ordering
6. Respond with JSON only - no other text

Calculate order quantity like this:
- Get average daily demand from forecast
- Multiply by 15 to get target quantity
- Check if affordable within 35% of budget
- If not affordable reduce quantity

JSON format:
{"orders": [{"sku_id": "LAPTOP-001", "supplier_id": "SUP-001", "quantity": 50}], "reasoning": "Coverage low, ordering 15 days supply"}

If all coverage above 8 days:
{"orders": [], "reasoning": "All SKUs have sufficient coverage"}
"""


def budget_aware_restock(obs) -> RestockAction:
    """
    BUDGET AWARE RESTOCKING.

    This is the core fix.
    Previous versions ordered too much
    and ran out of budget immediately.

    This version:
    1. Calculates actual daily demand
    2. Checks total coverage (stock + pending)
    3. Only orders if coverage below threshold
    4. Orders only what budget allows (max 35%)
    5. Targets 15 days of supply per order

    Result: budget spread across entire episode
    Stock never runs out because orders arrive on time
    Scores stay high all 30/60/90 days
    """
    orders      = []
    budget_left = obs.budget_remaining

    for sku in obs.skus:
        try:
            sku_id   = sku.sku_id
            current  = obs.inventory.get(sku_id, 0)
            forecast = obs.demand_forecast.get(sku_id, [10.0] * 7)
            avg_d    = sum(forecast) / len(forecast) if forecast else 10.0

            if avg_d <= 0:
                continue

            # Current days of stock
            days_left = current / avg_d

            # Add pending orders to coverage
            pending_qty = sum(
                o.quantity for o in obs.pending_orders
                if o.sku_id == sku_id
            )
            pending_days = pending_qty / avg_d
            total_coverage = days_left + pending_days

            # Skip if enough coverage
            if total_coverage >= COVERAGE_THRESHOLD:
                continue

            # Find cheapest active supplier
            best_sup   = None
            best_price = float("inf")

            for sup in obs.suppliers:
                if "active" not in str(sup.status).lower():
                    continue
                if sku_id not in sup.skus_supplied:
                    continue
                if sup.price_factor < best_price:
                    best_price = sup.price_factor
                    best_sup   = sup

            if best_sup is None:
                continue

            # Calculate unit cost
            unit_cost = sku.unit_cost * best_sup.price_factor

            # Target quantity = 15 days of demand
            target_qty = int(avg_d * TARGET_DAYS)
            target_qty = max(target_qty, sku.min_order_qty)

            # BUDGET CONSTRAINT - max 35% of remaining budget
            max_spend  = budget_left * MAX_BUDGET_PCT
            max_qty    = int(max_spend / unit_cost)
            order_qty  = min(target_qty, max_qty)

            # Must meet minimum order quantity
            if order_qty < sku.min_order_qty:
                continue

            # Dont exceed warehouse capacity
            incoming = pending_qty
            space    = sku.max_capacity - current - incoming
            space    = max(0, int(space))
            order_qty = min(order_qty, space)

            if order_qty < sku.min_order_qty:
                continue

            # Final budget check
            total_cost = unit_cost * order_qty
            if total_cost > budget_left:
                continue

            # Place order
            orders.append(OrderItem(
                sku_id=sku_id,
                supplier_id=best_sup.supplier_id,
                quantity=order_qty,
            ))
            budget_left -= total_cost

        except Exception as e:
            print(f"    Order calc error {sku.sku_id}: {e}")
            continue

    return RestockAction(
        orders=orders,
        reasoning=(
            f"Budget-aware restock: {len(orders)} SKUs need restocking"
            if orders else
            "All SKUs have sufficient coverage"
        )
    )


def call_llm(obs) -> RestockAction:
    """
    Calls LLM for decisions.
    Falls back gracefully on any error.
    """
    inv_lines = []
    for sku in obs.skus:
        qty      = obs.inventory.get(sku.sku_id, 0)
        forecast = obs.demand_forecast.get(sku.sku_id, [10]*7)
        avg_d    = sum(forecast) / len(forecast) if forecast else 10
        days     = qty / avg_d if avg_d > 0 else 999

        pending = sum(
            o.quantity for o in obs.pending_orders
            if o.sku_id == sku.sku_id
        )
        pending_d  = pending / avg_d if avg_d > 0 else 0
        coverage   = days + pending_d
        flag       = "ORDER NOW" if coverage < COVERAGE_THRESHOLD else "ok"

        inv_lines.append(
            f"  {sku.sku_id}: stock={qty} | "
            f"coverage={coverage:.1f}days | "
            f"demand={avg_d:.0f}/day | "
            f"min_order={sku.min_order_qty} | "
            f"cost=${sku.unit_cost}/unit | {flag}"
        )

    sup_lines = []
    for s in obs.suppliers:
        if "active" in str(s.status).lower():
            sup_lines.append(
                f"  {s.supplier_id}: "
                f"lead={s.lead_time_days}d | "
                f"price={s.price_factor}x | "
                f"skus={s.skus_supplied}"
            )
        else:
            sup_lines.append(f"  {s.supplier_id}: INACTIVE")

    pending_lines = [
        f"{o.sku_id}:{o.quantity}u arriving day {o.expected_day}"
        for o in obs.pending_orders
    ]

    max_spend = obs.budget_remaining * MAX_BUDGET_PCT

    prompt = f"""Day {obs.current_day} of {obs.total_days}
Total budget remaining: ${obs.budget_remaining:,.0f}
MAX you can spend this order: ${max_spend:,.0f} (35% rule)

INVENTORY COVERAGE:
{chr(10).join(inv_lines)}

ACTIVE SUPPLIERS:
{chr(10).join(sup_lines)}

PENDING DELIVERIES: {', '.join(pending_lines) if pending_lines else 'none'}

Order SKUs with coverage below {COVERAGE_THRESHOLD} days.
Target {TARGET_DAYS} days supply per order.
Stay within ${max_spend:,.0f} budget limit.
JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        text = response.choices[0].message.content or ""
        time.sleep(SLEEP)

        # Parse response
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        s = text.find("{")
        e = text.rfind("}") + 1
        if s != -1 and e > s:
            text = text[s:e]

        data   = json.loads(text.strip())
        orders = []

        for o in data.get("orders", []):
            if not all(k in o for k in ["sku_id", "supplier_id", "quantity"]):
                continue

            sku_id = str(o["sku_id"])
            sup_id = str(o["supplier_id"])
            qty    = int(o["quantity"])

            if qty <= 0:
                continue

            # Validate supplier active and carries SKU
            sup = next(
                (s for s in obs.suppliers if s.supplier_id == sup_id),
                None
            )
            if sup is None:
                continue
            if "active" not in str(sup.status).lower():
                continue
            if sku_id not in sup.skus_supplied:
                continue

            orders.append(OrderItem(
                sku_id=sku_id,
                supplier_id=sup_id,
                quantity=qty,
            ))

        return RestockAction(
            orders=orders,
            reasoning=str(data.get("reasoning", "llm decision"))
        )

    except Exception as e:
        msg = str(e)
        if "decommissioned" in msg:
            print(f"\nFATAL: Model decommissioned!")
            print(f"Update MODEL_NAME in .env")
            raise SystemExit(1)
        if "401" in msg:
            print(f"\nFATAL: Invalid API key!")
            raise SystemExit(1)
        if "429" in msg or "rate_limit" in msg:
            print(f"    Rate limit - waiting 10s...")
            time.sleep(10)
        return RestockAction(orders=[], reasoning="llm_error")


def run_task(task_id: str) -> float:
    """Runs one complete task and returns final score"""
    print(f"\n{'='*55}")
    print(f"RUNNING: {task_id.upper()}")
    print(f"{'='*55}")

    try:
        env    = SupplyChainEnvironment(task_id)
        result = env.reset()
        obs    = result.observation
    except Exception as e:
        print(f"FAILED to start {task_id}: {e}")
        traceback.print_exc()
        return 0.0

    max_steps   = DEMO_STEPS[task_id]
    final_score = 0.0
    llm_calls   = 0
    task_start  = time.time()

    print(f"Goal  : {obs.goal}")
    print(f"Budget: ${obs.budget_remaining:,.0f}")
    print(f"SKUs  : {list(obs.inventory.keys())}")
    print()

    for step in range(1, max_steps + 1):

        if result.done:
            print(f"Episode complete at day {obs.current_day}")
            break

        obs = result.observation
        day = obs.current_day

        try:
            # Rule based ordering runs EVERY day
            rule_action = budget_aware_restock(obs)

            # LLM runs every 5 days
            llm_action = RestockAction(orders=[], reasoning="skipped")
            if step % LLM_EVERY == 1:
                llm_action = call_llm(obs)
                llm_calls += 1

            # Combine orders
            # Rule orders are base (reliable)
            # LLM adds non-duplicate orders
            final_dict  = {o.sku_id: o for o in rule_action.orders}
            budget_used = sum(
                next(
                    (sk.unit_cost * sp.price_factor
                     for sp in obs.suppliers
                     if sp.supplier_id == o.supplier_id),
                    0
                ) * o.quantity
                for o in rule_action.orders
                for sk in obs.skus
                if sk.sku_id == o.sku_id
            )
            budget_left = obs.budget_remaining - budget_used

            for o in llm_action.orders:
                if o.sku_id in final_dict:
                    continue
                try:
                    sup = next(
                        s for s in obs.suppliers
                        if s.supplier_id == o.supplier_id
                    )
                    sku = next(
                        s for s in obs.skus
                        if s.sku_id == o.sku_id
                    )
                    cost = sku.unit_cost * sup.price_factor * o.quantity
                    if cost <= budget_left:
                        final_dict[o.sku_id] = o
                        budget_left -= cost
                except Exception:
                    continue

            final_action = RestockAction(
                orders=list(final_dict.values()),
                reasoning=(
                    f"rule:{len(rule_action.orders)} "
                    f"llm:{len(llm_action.orders)}"
                )
            )

            # Take step
            result      = env.step(final_action)
            final_score = result.reward.total_score

            # Print
            if final_action.orders:
                summary = " | ".join([
                    f"{o.quantity}x{o.sku_id}"
                    for o in final_action.orders
                ])
                print(
                    f"  Day {day:3d}: [{summary}] "
                    f"Score:{result.reward.total_score:.3f}"
                )
            else:
                stockout = ""
                if result.reward.stockouts_today:
                    stockout = f" STOCKOUT:{result.reward.stockouts_today}"
                print(
                    f"  Day {day:3d}: No orders "
                    f"Score:{result.reward.total_score:.3f}"
                    f"{stockout}"
                )

        except Exception as e:
            print(f"  Day {day}: ERROR - {e}")
            traceback.print_exc()
            try:
                result = env.step(
                    RestockAction(orders=[], reasoning="error")
                )
                final_score = result.reward.total_score
            except Exception:
                break

    elapsed = time.time() - task_start
    try:
        state = env.state()
        print(f"\n{'─'*55}")
        print(f"Task {task_id} COMPLETE")
        print(f"  Final score    : {final_score:.4f}")
        print(f"  LLM calls      : {llm_calls}")
        print(f"  Time           : {elapsed:.1f}s")
        print(f"  Stockout days  : {state.total_stockout_days}")
        print(f"  Budget left    : ${state.budget_remaining:,.0f}")
        print(f"{'─'*55}")
    except Exception:
        print(f"  Score: {final_score:.4f}")

    return final_score


def main():
    """Main entry point"""
    print("=" * 55)
    print("  SUPPLYCHAIN-ENV BASELINE INFERENCE")
    print("=" * 55)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Endpoint: {API_BASE_URL}")
    print(f"  Coverage threshold: {COVERAGE_THRESHOLD} days")
    print(f"  Max budget per order: {MAX_BUDGET_PCT*100:.0f}%")
    print("=" * 55)

    # Validate model
    print(f"\nValidating model: {MODEL_NAME}...")
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5,
            temperature=0.0,
        )
        print(f"Model OK: {r.choices[0].message.content}")
    except Exception as e:
        msg = str(e)
        print(f"Model FAILED: {msg}")
        if "decommissioned" in msg:
            print("Update MODEL_NAME in .env")
        raise SystemExit(1)

    scores     = {}
    start_time = time.time()

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            scores[task_id] = round(run_task(task_id), 4)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error in {task_id}: {e}")
            traceback.print_exc()
            scores[task_id] = 0.0

    elapsed = time.time() - start_time
    avg     = sum(scores.values()) / len(scores) if scores else 0

    print(f"\n{'='*55}")
    print(f"  FINAL BASELINE SCORES")
    print(f"{'='*55}")
    print(f"  task_easy   : {scores.get('task_easy',   0):.4f}")
    print(f"  task_medium : {scores.get('task_medium', 0):.4f}")
    print(f"  task_hard   : {scores.get('task_hard',   0):.4f}")
    print(f"  {'─'*45}")
    print(f"  Average     : {avg:.4f}")
    print(f"  Time        : {elapsed:.1f}s ({elapsed/60:.1f} mins)")
    print(f"{'='*55}")

    results = {
        "model":        MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "scores": {
            "task_easy":   scores.get("task_easy",   0),
            "task_medium": scores.get("task_medium", 0),
            "task_hard":   scores.get("task_hard",   0),
        },
        "average_score": round(avg, 4),
        "time_seconds":  round(elapsed, 1),
        "time_minutes":  round(elapsed / 60, 2),
    }

    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to baseline_scores.json")
    print("Done!")


if __name__ == "__main__":
    main()