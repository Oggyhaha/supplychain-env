# inference.py
# ═══════════════════════════════════════════════
# BASELINE INFERENCE SCRIPT
#
# Runs Groq LLM against SupplyChain-Env
# for all 3 tasks and reports scores.
#
# Usage:
#   python inference.py
#
# Required environment variables in .env:
#   OPENAI_API_KEY  = your Groq API key
#   API_BASE_URL    = https://api.groq.com/openai/v1
#   MODEL_NAME      = llama-3.1-8b-instant
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
    raise ValueError(
        "OPENAI_API_KEY not found in .env file\n"
        "Add this line to .env: OPENAI_API_KEY=gsk_xxx"
    )
if not API_BASE_URL:
    raise ValueError(
        "API_BASE_URL not found in .env file\n"
        "Add this line to .env: API_BASE_URL=https://api.groq.com/openai/v1"
    )
if not MODEL_NAME:
    raise ValueError(
        "MODEL_NAME not found in .env file\n"
        "Add this line to .env: MODEL_NAME=llama-3.1-8b-instant"
    )

# ── CREATE GROQ CLIENT ─────────────────────────
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

# ── CONFIGURATION ──────────────────────────────
MAX_STEPS        = 100    # Safety limit per episode
TEMPERATURE      = 0.0    # 0 = deterministic responses
MAX_TOKENS       = 1000   # Max response length
RETRY_WAIT       = 5      # Seconds to wait on rate limit
MAX_RETRIES      = 3      # Max retries per step

# ── SYSTEM PROMPT ──────────────────────────────
SYSTEM_PROMPT = """
You are an expert supply chain manager AI agent.
Your job is to manage warehouse inventory optimally.

Every day you must decide:
- Which products (SKUs) to reorder
- How many units to order
- Which supplier to order from

Your goals:
1. Never run out of stock (avoid stockouts)
2. Do not order too much (avoid overstock)
3. Stay within budget
4. Choose cheapest and fastest supplier wisely

You will receive current inventory levels,
demand forecasts, supplier options, and budget.

You must respond with a valid JSON object ONLY.
No extra text. No explanation outside JSON.
No markdown. Just the raw JSON.

Example response when ordering:
{"orders": [{"sku_id": "LAPTOP-001", "supplier_id": "SUP-001", "quantity": 50}], "reasoning": "Stock is low"}

Example response when no orders needed:
{"orders": [], "reasoning": "Stock levels are healthy today"}

Rules:
- quantity must be a positive whole number
- Only order from suppliers listed as ACTIVE
- Only order SKUs that supplier actually supplies
- Always respond with valid JSON only
"""


def build_user_prompt(observation) -> str:
    """
    Converts environment observation into
    a clear text prompt for the LLM.
    """
    # Build inventory status
    inventory_lines = []
    for sku in observation.skus:
        current_qty = observation.inventory.get(sku.sku_id, 0)
        forecast    = observation.demand_forecast.get(
            sku.sku_id, [0] * 7
        )
        avg_daily = (
            sum(forecast) / len(forecast)
            if forecast else 0
        )
        days_left = (
            current_qty / avg_daily
            if avg_daily > 0 else 999
        )

        status = "URGENT - ORDER NOW" if current_qty <= sku.reorder_point else "OK"

        inventory_lines.append(
            f"  {sku.sku_id} ({sku.name}):\n"
            f"    Status: {status}\n"
            f"    Current stock: {current_qty} units\n"
            f"    Reorder point: {sku.reorder_point} units\n"
            f"    Max capacity: {sku.max_capacity} units\n"
            f"    Min order qty: {sku.min_order_qty} units\n"
            f"    Avg daily demand: {avg_daily:.1f} units\n"
            f"    Days of stock left: {days_left:.1f}\n"
            f"    7-day forecast: {[round(x,1) for x in forecast]}\n"
            f"    Unit cost: ${sku.unit_cost}\n"
            f"    Stockout penalty: ${sku.stockout_penalty}/unit"
        )

    # Build supplier status
    supplier_lines = []
    for supplier in observation.suppliers:
        if supplier.status == "active":
            supplier_lines.append(
                f"  {supplier.supplier_id} ({supplier.name}):\n"
                f"    Status: ACTIVE\n"
                f"    Lead time: {supplier.lead_time_days} days\n"
                f"    Price factor: {supplier.price_factor}x base cost\n"
                f"    Min order value: ${supplier.min_order_value}\n"
                f"    Supplies: {supplier.skus_supplied}"
            )
        else:
            supplier_lines.append(
                f"  {supplier.supplier_id} ({supplier.name}):\n"
                f"    Status: {str(supplier.status).upper()} - DO NOT USE"
            )

    # Build pending orders
    pending_lines = []
    for order in observation.pending_orders:
        pending_lines.append(
            f"  {order.order_id}: "
            f"{order.quantity}x {order.sku_id} "
            f"from {order.supplier_id} "
            f"arriving day {order.expected_day}"
        )

    # Build yesterday metrics
    yesterday_section = ""
    if observation.yesterday_metrics:
        m = observation.yesterday_metrics
        yesterday_section = (
            f"\nYESTERDAY RESULTS:\n"
            f"  Revenue: ${m.revenue:.2f}\n"
            f"  Stockouts: {m.stockouts if m.stockouts else 'None - good job!'}\n"
            f"  Holding costs: ${m.holding_costs:.2f}\n"
            f"  Stockout losses: ${m.stockout_losses:.2f}"
        )

    prompt = f"""
DAY {observation.current_day} of {observation.total_days} 
({observation.days_remaining} days remaining)

TASK: {observation.goal}

BUDGET:
  Remaining: ${observation.budget_remaining:.2f} of ${observation.total_budget:.2f}

INVENTORY STATUS:
{chr(10).join(inventory_lines)}

AVAILABLE SUPPLIERS:
{chr(10).join(supplier_lines)}

PENDING ORDERS (already placed, not arrived yet):
{chr(10).join(pending_lines) if pending_lines else '  None'}
{yesterday_section}

CURRENT SCORE: {observation.cumulative_score:.4f}

Decide what to order today. Remember:
- Check which SKUs are below reorder point
- Account for lead time when ordering
- Stay within budget of ${observation.budget_remaining:.2f}
- Only use ACTIVE suppliers

Respond with JSON only.
"""
    return prompt


def validate_model() -> bool:
    """
    Tests if the configured model works
    before running full inference.
    Returns True if working, False if not.
    """
    print(f"Validating model: {MODEL_NAME}...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": 'Respond with this exact JSON: {"orders": [], "reasoning": "test"}'
                }
            ],
            max_tokens=50,
            temperature=0.0,
        )
        reply = response.choices[0].message.content
        print(f"Model validation passed! Response: {reply}")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"Model validation FAILED: {error_msg}")

        if "decommissioned" in error_msg:
            print(f"\nFATAL: Model '{MODEL_NAME}' is decommissioned!")
            print(f"Fix: Update MODEL_NAME in your .env file")
            print(f"Available models: https://console.groq.com/docs/models")
            print(f"Recommended: llama-3.1-8b-instant")

        elif "401" in error_msg or "invalid_api_key" in error_msg:
            print(f"\nFATAL: Invalid API key!")
            print(f"Fix: Update OPENAI_API_KEY in your .env file")
            print(f"Get key from: https://console.groq.com")

        return False


def call_llm_with_retry(user_prompt: str) -> str:
    """
    Calls LLM with retry logic.

    Handles:
    - Decommissioned model → stop immediately
    - Rate limit          → wait and retry
    - Network error       → retry up to 3 times
    - Other errors        → use fallback action

    Returns response text string.
    """
    fallback = '{"orders": [], "reasoning": "api_error_fallback"}'

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            # Success - return response
            return completion.choices[0].message.content or fallback

        except Exception as e:
            error_msg = str(e)

            # Model decommissioned - fatal, stop everything
            if "decommissioned" in error_msg:
                print(f"\nFATAL ERROR: Model '{MODEL_NAME}' is decommissioned!")
                print(f"Update MODEL_NAME in .env file")
                print(f"Recommended: llama-3.1-8b-instant")
                raise SystemExit(1)

            # Rate limit - wait then retry
            elif "rate_limit" in error_msg or "429" in error_msg:
                wait_time = RETRY_WAIT * attempt
                print(f"    Rate limit. Waiting {wait_time}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue

            # Auth error - fatal
            elif "401" in error_msg or "invalid_api_key" in error_msg:
                print(f"\nFATAL ERROR: Invalid API key!")
                print(f"Update OPENAI_API_KEY in .env file")
                raise SystemExit(1)

            # Network or other error - retry
            else:
                if attempt < MAX_RETRIES:
                    print(f"    Error (attempt {attempt}/{MAX_RETRIES}): {e}")
                    print(f"    Retrying in {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                else:
                    print(f"    All retries failed. Using fallback action.")
                    return fallback

    return fallback


def parse_action(response_text: str) -> RestockAction:
    """
    Parses LLM response text into RestockAction.

    Handles:
    - Perfect JSON
    - JSON wrapped in markdown code blocks
    - JSON with extra text around it
    - Completely invalid response (fallback)
    """
    try:
        text = response_text.strip()

        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Find JSON object in text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        # Parse JSON
        data = json.loads(text.strip())

        # Build order items
        orders = []
        for order_data in data.get("orders", []):
            # Validate required fields exist
            if not all(k in order_data for k in ["sku_id", "supplier_id", "quantity"]):
                continue

            qty = int(order_data["quantity"])
            if qty <= 0:
                continue

            orders.append(OrderItem(
                sku_id=str(order_data["sku_id"]),
                supplier_id=str(order_data["supplier_id"]),
                quantity=qty,
            ))

        return RestockAction(
            orders=orders,
            reasoning=str(data.get("reasoning", "")),
        )

    except Exception as e:
        # Return empty action as safe fallback
        return RestockAction(
            orders=[],
            reasoning=f"parse_error: {str(e)}"
        )


def run_task(task_id: str) -> float:
    """
    Runs one complete episode for a task.
    Returns final score (0.0 to 1.0)
    """
    print(f"\n{'='*55}")
    print(f"RUNNING TASK: {task_id.upper()}")
    print(f"{'='*55}")

    # Create and reset environment
    env    = SupplyChainEnvironment(task_id)
    result = env.reset()
    obs    = result.observation

    print(f"Goal      : {obs.goal}")
    print(f"Total Days: {obs.total_days}")
    print(f"Budget    : ${obs.budget_remaining:,.2f}")
    print(f"SKUs      : {list(obs.inventory.keys())}")
    print(f"\nStarting simulation...\n")

    final_score = 0.0
    step_count  = 0

    for step in range(1, MAX_STEPS + 1):

        # Check if episode is done
        if result.done:
            print(f"\nEpisode complete at day {step - 1}")
            break

        step_count = step
        obs        = result.observation

        # Build prompt for LLM
        user_prompt = build_user_prompt(obs)

        # Call LLM with retry logic
        response_text = call_llm_with_retry(user_prompt)

        # Parse response into action
        action = parse_action(response_text)

        # Print step summary
        day = obs.current_day
        if action.orders:
            order_summary = ", ".join([
                f"{o.quantity}x {o.sku_id} from {o.supplier_id}"
                for o in action.orders
            ])
            print(
                f"  Day {day:3d}: Ordered [{order_summary}]"
                f" | Score: {result.reward.total_score:.3f}"
            )
        else:
            stockout_warn = ""
            if result.reward.stockouts_today:
                stockout_warn = f" | STOCKOUT: {result.reward.stockouts_today}"
            print(
                f"  Day {day:3d}: No orders"
                f" | Score: {result.reward.total_score:.3f}"
                f"{stockout_warn}"
            )

        # Take step in environment
        result = env.step(action)

        # Track final score
        final_score = result.reward.total_score

        # Small delay to respect API rate limits
        time.sleep(1.5)

    # Final summary for this task
    state = env.state()
    print(f"\n{'─'*55}")
    print(f"Task {task_id} COMPLETE")
    print(f"  Steps taken      : {step_count}")
    print(f"  Final score      : {final_score:.4f}")
    print(f"  Budget remaining : ${state.budget_remaining:,.2f}")
    print(f"  Total revenue    : ${state.total_revenue:,.2f}")
    print(f"  Total stockout days: {state.total_stockout_days}")
    print(f"{'─'*55}")

    return final_score


def main():
    """
    Main function.
    1. Validates model works
    2. Runs all 3 tasks
    3. Prints and saves scores
    """
    print("=" * 55)
    print("  SUPPLYCHAIN-ENV BASELINE INFERENCE")
    print("=" * 55)
    print(f"  Model    : {MODEL_NAME}")
    print(f"  Endpoint : {API_BASE_URL}")
    print("=" * 55)

    # Validate model before running
    if not validate_model():
        print("\nCannot proceed. Fix model configuration first.")
        raise SystemExit(1)

    print("\nModel validated. Starting tasks...\n")

    # Store scores
    scores     = {}
    start_time = time.time()

    # Run all 3 tasks
    task_list = ["task_easy", "task_medium", "task_hard"]

    for task_id in task_list:
        try:
            score = run_task(task_id)
            scores[task_id] = round(score, 4)
        except SystemExit:
            raise
        except Exception as e:
            print(f"\nError running {task_id}: {e}")
            scores[task_id] = 0.0

    # Calculate results
    elapsed = time.time() - start_time
    avg     = sum(scores.values()) / len(scores) if scores else 0

    # Print final summary
    print(f"\n{'='*55}")
    print(f"  FINAL BASELINE SCORES")
    print(f"{'='*55}")
    print(f"  task_easy   (Easy)  : {scores.get('task_easy',   0):.4f}")
    print(f"  task_medium (Medium): {scores.get('task_medium', 0):.4f}")
    print(f"  task_hard   (Hard)  : {scores.get('task_hard',   0):.4f}")
    print(f"  {'─'*45}")
    print(f"  Average Score       : {avg:.4f}")
    print(f"  Total Time          : {elapsed:.1f} seconds")
    print(f"{'='*55}")

    # Save scores to JSON file
    results = {
        "model":          MODEL_NAME,
        "api_base_url":   API_BASE_URL,
        "scores": {
            "task_easy":   scores.get("task_easy",   0),
            "task_medium": scores.get("task_medium", 0),
            "task_hard":   scores.get("task_hard",   0),
        },
        "average_score":  round(avg, 4),
        "time_seconds":   round(elapsed, 1),
    }

    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nScores saved to: baseline_scores.json")
    print(f"Inference complete!")


if __name__ == "__main__":
    main()