# inference.py
# ═══════════════════════════════════════════════
# BASELINE INFERENCE SCRIPT - OPTIMIZED VERSION
#
# Optimizations:
# 1. Reduced sleep time between steps
# 2. LLM called every 3 days (not every day)
# 3. Shorter demo episodes for speed
# 4. Parallel-ready structure
#
# Usage: python inference.py
# Expected runtime: under 5 minutes
# ═══════════════════════════════════════════════

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from env.environment import SupplyChainEnvironment
from env.models import RestockAction, OrderItem

# ── LOAD ENVIRONMENT VARIABLES ─────────────────
load_dotenv()

API_KEY      = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")

# ── VALIDATE CREDENTIALS ───────────────────────
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not API_BASE_URL:
    raise ValueError("API_BASE_URL not found in .env")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME not found in .env")

# ── CREATE CLIENT ──────────────────────────────
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

# ── CONFIGURATION ──────────────────────────────
# KEY OPTIMIZATION: Call LLM every N days
# Instead of every single day
LLM_CALL_INTERVAL = 3      # Call LLM every 3 days (can be adjusted for more speed)
SLEEP_BETWEEN_CALLS = 0.3  # Reduced from 1.5 to 0.3
MAX_RETRIES  = 2
TEMPERATURE  = 0.0
MAX_TOKENS   = 800         # Reduced from 1000

# Demo steps per task (reduced for speed)
# Full episodes used for training
# Shorter episodes for baseline demonstration
DEMO_STEPS = {
    "task_easy":   30,   # 30 days
    "task_medium": 60,   # 60 days
    "task_hard":   90,   # 90 days
}

# ── SYSTEM PROMPT ──────────────────────────────
SYSTEM_PROMPT = """
You are a supply chain manager AI.
Manage warehouse inventory optimally.

Respond with valid JSON only. No extra text.

When ordering:
{"orders": [{"sku_id": "LAPTOP-001", "supplier_id": "SUP-001", "quantity": 50}], "reasoning": "Stock low"}

When no orders needed:
{"orders": [], "reasoning": "Stock healthy"}

Rules:
- Only order from ACTIVE suppliers
- quantity must be positive integer
- Stay within budget
- Prevent stockouts by ordering before reorder point
"""


def build_compact_prompt(observation) -> str:
    """
    Builds a SHORTER prompt than before.
    Less tokens = faster API response = faster overall.
    """
    # Inventory summary - compact format
    inv_lines = []
    for sku in observation.skus:
        qty      = observation.inventory.get(sku.sku_id, 0)
        forecast = observation.demand_forecast.get(sku.sku_id, [10]*7)
        avg_d    = sum(forecast) / len(forecast) if forecast else 10
        days_left = qty / avg_d if avg_d > 0 else 999
        urgent   = "URGENT" if qty <= sku.reorder_point else "ok"

        inv_lines.append(
            f"{sku.sku_id}: {qty} units | "
            f"{days_left:.0f} days left | "
            f"reorder@{sku.reorder_point} | "
            f"forecast_avg={avg_d:.0f}/day | "
            f"status={urgent}"
        )

    # Supplier summary - compact format
    sup_lines = []
    for s in observation.suppliers:
        status = str(s.status).upper()
        if "active" in status.lower():
            sup_lines.append(
                f"{s.supplier_id}({s.name}): "
                f"lead={s.lead_time_days}d | "
                f"price={s.price_factor}x | "
                f"sells={s.skus_supplied}"
            )
        else:
            sup_lines.append(
                f"{s.supplier_id}: {status} - DO NOT USE"
            )

    # Pending orders - compact
    pending = []
    for o in observation.pending_orders:
        pending.append(
            f"{o.sku_id}: {o.quantity} units "
            f"arriving day {o.expected_day}"
        )

    prompt = f"""
DAY {observation.current_day}/{observation.total_days}
Budget: ${observation.budget_remaining:.0f} remaining

INVENTORY:
{chr(10).join(inv_lines)}

SUPPLIERS:
{chr(10).join(sup_lines)}

PENDING: {', '.join(pending) if pending else 'None'}

What orders to place today? JSON only.
"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Calls LLM with minimal retry logic.
    Returns response text.
    """
    fallback = '{"orders": [], "reasoning": "fallback"}'

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return completion.choices[0].message.content or fallback

        except Exception as e:
            error_msg = str(e)

            # Fatal errors - stop immediately
            if "decommissioned" in error_msg:
                print(f"\nFATAL: Model decommissioned!")
                print(f"Update MODEL_NAME in .env file")
                print(f"Use: llama-3.1-8b-instant")
                raise SystemExit(1)

            if "401" in error_msg or "invalid_api_key" in error_msg:
                print(f"\nFATAL: Invalid API key!")
                raise SystemExit(1)

            # Rate limit - wait
            if "rate_limit" in error_msg or "429" in error_msg:
                wait = 5 * attempt
                print(f"    Rate limit. Waiting {wait}s...")
                time.sleep(wait)
                continue

            # Other error
            if attempt == MAX_RETRIES:
                print(f"    LLM failed after {MAX_RETRIES} attempts")
                return fallback

            time.sleep(2)

    return fallback


def parse_action(text: str) -> RestockAction:
    """Parses LLM response into RestockAction"""
    try:
        text = text.strip()

        # Remove markdown if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Extract JSON object
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        data   = json.loads(text.strip())
        orders = []

        for o in data.get("orders", []):
            if not all(k in o for k in ["sku_id", "supplier_id", "quantity"]):
                continue
            qty = int(o["quantity"])
            if qty <= 0:
                continue
            orders.append(OrderItem(
                sku_id=str(o["sku_id"]),
                supplier_id=str(o["supplier_id"]),
                quantity=qty,
            ))

        return RestockAction(
            orders=orders,
            reasoning=str(data.get("reasoning", "")),
        )

    except Exception:
        return RestockAction(orders=[], reasoning="parse_error")


def make_rule_based_action(observation) -> RestockAction:
    """
    SMART RULE BASED FALLBACK.
    Used on days when LLM is NOT called.

    Simple logic:
    If stock below reorder point → order from cheapest supplier
    Otherwise → do nothing

    This is fast (no API call) and works well
    for days between LLM decisions.
    """
    orders = []

    for sku in observation.skus:
        qty = observation.inventory.get(sku.sku_id, 0)

        # Only order if below reorder point
        if qty <= sku.reorder_point:

            # Find cheapest active supplier for this SKU
            best_supplier = None
            best_price    = float("inf")

            for supplier in observation.suppliers:
                status = str(supplier.status).lower()
                if (sku.sku_id in supplier.skus_supplied
                        and "active" in status):
                    if supplier.price_factor < best_price:
                        best_price    = supplier.price_factor
                        best_supplier = supplier

            if best_supplier:
                # Order enough for 2 weeks
                order_qty = max(
                    sku.min_order_qty,
                    sku.reorder_point * 3
                )

                # Check budget
                cost = (
                    sku.unit_cost
                    * best_supplier.price_factor
                    * order_qty
                )
                if cost <= observation.budget_remaining:
                    orders.append(OrderItem(
                        sku_id=sku.sku_id,
                        supplier_id=best_supplier.supplier_id,
                        quantity=order_qty,
                    ))

    return RestockAction(
        orders=orders,
        reasoning="rule_based: reorder below threshold"
    )


def validate_model() -> bool:
    """Quick model validation before running"""
    print(f"Validating model: {MODEL_NAME}...")
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Reply: ok"}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        reply = r.choices[0].message.content
        print(f"Model OK! Response: {reply}")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"Model FAILED: {error_msg}")

        if "decommissioned" in error_msg:
            print(f"Update MODEL_NAME in .env")
            print(f"Recommended: llama-3.1-8b-instant")
        elif "401" in error_msg:
            print(f"Invalid API key in .env")

        return False


def run_task(task_id: str) -> float:
    """
    Runs one task episode.

    KEY OPTIMIZATION:
    LLM is called every LLM_CALL_INTERVAL days.
    Rule-based action used on other days.
    This reduces API calls by 3x.

    Returns final score (0.0 to 1.0)
    """
    print(f"\n{'='*55}")
    print(f"RUNNING: {task_id.upper()}")
    print(f"{'='*55}")

    env    = SupplyChainEnvironment(task_id)
    result = env.reset()
    obs    = result.observation

    max_steps  = DEMO_STEPS[task_id]
    llm_calls  = 0
    rule_calls = 0
    final_score = 0.0

    print(f"Goal   : {obs.goal}")
    print(f"Days   : {obs.total_days} (running {max_steps} for baseline)")
    print(f"Budget : ${obs.budget_remaining:,.0f}")
    print(f"SKUs   : {list(obs.inventory.keys())}")
    print(f"Strategy: LLM every {LLM_CALL_INTERVAL} days, rules otherwise")
    print()

    step_start = time.time()

    for step in range(1, max_steps + 1):

        if result.done:
            print(f"Episode done at step {step - 1}")
            break

        obs = result.observation
        day = obs.current_day

        # OPTIMIZATION: Only call LLM every N steps
        # Use rule-based action on other days
        if step % LLM_CALL_INTERVAL == 1:
            # Call LLM for decision
            prompt        = build_compact_prompt(obs)
            response_text = call_llm(prompt)
            action        = parse_action(response_text)
            llm_calls    += 1
            call_type     = "LLM"

            # Small sleep only after LLM call
            time.sleep(SLEEP_BETWEEN_CALLS)

        else:
            # Use fast rule-based action
            action     = make_rule_based_action(obs)
            rule_calls += 1
            call_type  = "rule"

        # Take step
        result = env.step(action)
        final_score = result.reward.total_score

        # Print step info
        if action.orders:
            order_summary = ", ".join([
                f"{o.quantity}x {o.sku_id}"
                for o in action.orders
            ])
            print(
                f"  Day {day:3d} [{call_type:4s}]: "
                f"Ordered [{order_summary}] | "
                f"Score: {result.reward.total_score:.3f}"
            )
        else:
            stockout = ""
            if result.reward.stockouts_today:
                stockout = f" | STOCKOUT:{result.reward.stockouts_today}"
            print(
                f"  Day {day:3d} [{call_type:4s}]: "
                f"No orders | "
                f"Score: {result.reward.total_score:.3f}"
                f"{stockout}"
            )

    # Task summary
    step_time = time.time() - step_start
    state     = env.state()

    print(f"\n{'─'*55}")
    print(f"Task {task_id} DONE")
    print(f"  Final score    : {final_score:.4f}")
    print(f"  LLM calls made : {llm_calls}")
    print(f"  Rule calls made: {rule_calls}")
    print(f"  Time taken     : {step_time:.1f} seconds")
    print(f"  Budget left    : ${state.budget_remaining:,.0f}")
    print(f"  Stockout days  : {state.total_stockout_days}")
    print(f"{'─'*55}")

    return final_score


def main():
    """
    Main function.
    Validates model → runs 3 tasks → saves scores.
    Target: complete in under 5 minutes.
    """
    print("=" * 55)
    print("  SUPPLYCHAIN-ENV BASELINE INFERENCE")
    print("  Optimized for speed")
    print("=" * 55)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Endpoint : {API_BASE_URL}")
    print(f"  LLM call : every {LLM_CALL_INTERVAL} steps")
    print(f"  Sleep    : {SLEEP_BETWEEN_CALLS}s between calls")
    print("=" * 55)

    # Validate model first
    if not validate_model():
        print("\nFix model configuration first.")
        raise SystemExit(1)

    print("\nStarting tasks...\n")

    scores     = {}
    start_time = time.time()

    # Run all 3 tasks
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            score          = run_task(task_id)
            scores[task_id] = round(score, 4)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error in {task_id}: {e}")
            scores[task_id] = 0.0

    # Final results
    elapsed = time.time() - start_time
    avg     = sum(scores.values()) / len(scores) if scores else 0

    print(f"\n{'='*55}")
    print(f"  FINAL BASELINE SCORES")
    print(f"{'='*55}")
    print(f"  task_easy   (Easy)  : {scores.get('task_easy',   0):.4f}")
    print(f"  task_medium (Medium): {scores.get('task_medium', 0):.4f}")
    print(f"  task_hard   (Hard)  : {scores.get('task_hard',   0):.4f}")
    print(f"  {'─'*45}")
    print(f"  Average Score       : {avg:.4f}")
    print(f"  Total Time          : {elapsed:.1f} seconds")
    print(f"  Total Time (mins)   : {elapsed/60:.1f} minutes")
    print(f"{'='*55}")

    # Save to JSON
    results = {
        "model":         MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "optimization": {
            "llm_call_interval":    LLM_CALL_INTERVAL,
            "sleep_between_calls":  SLEEP_BETWEEN_CALLS,
            "demo_steps":           DEMO_STEPS,
        },
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

    print(f"\nScores saved to baseline_scores.json")
    print(f"Inference complete in {elapsed/60:.1f} minutes!")

    # Warn if too slow
    if elapsed > 1200:
        print(f"\nWARNING: Took over 20 minutes!")
        print(f"Reduce DEMO_STEPS or LLM_CALL_INTERVAL")
    else:
        print(f"Well within 20 minute limit!")


if __name__ == "__main__":
    main()