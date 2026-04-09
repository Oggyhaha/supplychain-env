# inference.py
# ═══════════════════════════════════════════════
# SUPPLYCHAIN-ENV BASELINE INFERENCE
# Mandatory stdout format: [START][STEP][END]
# ═══════════════════════════════════════════════

import os
import json
import time
import sys
import traceback
from openai import OpenAI
from env.environment import SupplyChainEnvironment
from env.models import RestockAction, OrderItem

# ── LOAD ENV ───────────────────────────────────
# HF_TOKEN is the primary key (required by hackathon spec)
# Falls back to OPENAI_API_KEY for local dev
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# ── CLIENT ─────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── SETTINGS ───────────────────────────────────
BENCHMARK          = "supplychain-env"
COVERAGE_THRESHOLD = 5
TARGET_DAYS        = 15
MAX_BUDGET_PCT     = 0.60
LLM_EVERY          = 3
SLEEP              = 0.5

DEMO_STEPS = {
    "task_easy":   30,
    "task_medium": 60,
    "task_hard":   90,
}

SYSTEM_PROMPT = """
You are a warehouse inventory manager.
Prevent stockouts by ordering stock proactively.

Rules:
1. Order when stock coverage drops below 5 days
2. Order enough for 15 days of demand
3. Only use ACTIVE suppliers
4. Never exceed budget
5. Respond with JSON only

Format:
{"orders": [{"sku_id": "LAPTOP-001", "supplier_id": "SUP-001", "quantity": 150}], "reasoning": "Low stock"}

If stock healthy:
{"orders": [], "reasoning": "Stock levels OK"}
"""


def budget_aware_restock(obs) -> RestockAction:
    """Smart rule-based restocking that respects budget"""
    orders      = []
    budget_left = obs.budget_remaining

    for sku in obs.skus:
        try:
            sku_id   = sku.sku_id
            current  = obs.inventory.get(sku_id, 0)
            forecast = obs.demand_forecast.get(sku_id, [10.0]*7)
            avg_d    = sum(forecast)/len(forecast) if forecast else 10.0

            if avg_d <= 0:
                continue

            days_left   = current / avg_d
            pending_qty = sum(
                o.quantity for o in obs.pending_orders
                if o.sku_id == sku_id
            )
            pending_days   = pending_qty / avg_d
            total_coverage = days_left + pending_days

            if total_coverage >= COVERAGE_THRESHOLD:
                continue

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

            unit_cost  = sku.unit_cost * best_sup.price_factor
            target_qty = int(avg_d * TARGET_DAYS)
            target_qty = max(target_qty, sku.min_order_qty)

            max_spend  = budget_left * MAX_BUDGET_PCT
            max_qty    = int(max_spend / unit_cost) if unit_cost > 0 else 0
            order_qty  = min(target_qty, max_qty)

            if order_qty < sku.min_order_qty:
                continue

            incoming  = pending_qty
            space     = sku.max_capacity - current - incoming
            space     = max(0, int(space))
            order_qty = min(order_qty, space)

            if order_qty < sku.min_order_qty:
                continue

            total_cost = unit_cost * order_qty
            if total_cost > budget_left:
                continue

            orders.append(OrderItem(
                sku_id=sku_id,
                supplier_id=best_sup.supplier_id,
                quantity=order_qty,
            ))
            budget_left -= total_cost

        except Exception:
            continue

    return RestockAction(
        orders=orders,
        reasoning=(
            f"Restocking {len(orders)} SKUs"
            if orders else "Stock levels sufficient"
        )
    )


def call_llm(obs) -> RestockAction:
    """Calls LLM for strategic decisions — falls back to empty action on any error"""
    try:
        inv_lines = []
        for sku in obs.skus:
            qty      = obs.inventory.get(sku.sku_id, 0)
            forecast = obs.demand_forecast.get(sku.sku_id, [10]*7)
            avg_d    = sum(forecast)/len(forecast) if forecast else 10
            days     = qty/avg_d if avg_d > 0 else 999
            pending  = sum(
                o.quantity for o in obs.pending_orders
                if o.sku_id == sku.sku_id
            )
            coverage = days + pending/avg_d if avg_d > 0 else days
            flag     = "ORDER NOW" if coverage < COVERAGE_THRESHOLD else "ok"
            inv_lines.append(
                f"  {sku.sku_id}: stock={qty} coverage={coverage:.1f}d "
                f"demand={avg_d:.0f}/day cost={sku.unit_cost} {flag}"
            )

        sup_lines = []
        for s in obs.suppliers:
            if "active" in str(s.status).lower():
                sup_lines.append(
                    f"  {s.supplier_id}: lead={s.lead_time_days}d "
                    f"price={s.price_factor}x skus={s.skus_supplied}"
                )
            else:
                sup_lines.append(f"  {s.supplier_id}: INACTIVE")

        prompt = f"""Day {obs.current_day}/{obs.total_days}
Budget: ${obs.budget_remaining:,.0f}

INVENTORY:
{chr(10).join(inv_lines)}

SUPPLIERS:
{chr(10).join(sup_lines)}

Order SKUs with coverage below {COVERAGE_THRESHOLD} days.
JSON only."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        text = response.choices[0].message.content or ""
        time.sleep(SLEEP)

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
            qty = int(o["quantity"])
            if qty <= 0:
                continue
            sup = next(
                (s for s in obs.suppliers
                 if s.supplier_id == str(o["supplier_id"])),
                None
            )
            if sup is None:
                continue
            if "active" not in str(sup.status).lower():
                continue
            if str(o["sku_id"]) not in sup.skus_supplied:
                continue
            orders.append(OrderItem(
                sku_id=str(o["sku_id"]),
                supplier_id=str(o["supplier_id"]),
                quantity=qty,
            ))
        return RestockAction(
            orders=orders,
            reasoning=str(data.get("reasoning", "llm"))
        )

    except Exception as e:
        print(f"LLM call failed (using rule-based fallback): {e}", file=sys.stderr)
        return RestockAction(orders=[], reasoning="llm_error_fallback")


def action_to_str(action: RestockAction) -> str:
    """Converts action to string for [STEP] log"""
    if not action.orders:
        return "no_order"
    parts = [f"order_{o.sku_id}_{o.quantity}" for o in action.orders]
    return "+".join(parts)


def run_task(task_id: str) -> dict:
    """
    Runs one complete task.
    Emits mandatory [START][STEP][END] stdout format.
    Returns result dict with score and rewards.
    """
    rewards     = []
    steps_done  = 0
    final_score = 0.0
    success     = False

    try:
        env    = SupplyChainEnvironment(task_id)
        result = env.reset()

        # ── MANDATORY [START] LINE ──────────────
        print(
            f"[START] task={task_id} "
            f"env={BENCHMARK} "
            f"model={MODEL_NAME}",
            flush=True
        )

        max_steps = DEMO_STEPS[task_id]

        for step in range(1, max_steps + 1):

            if result.done:
                break

            obs    = result.observation
            err    = None
            reward = 0.0
            done   = False

            try:
                # Rule-based every day
                rule_action = budget_aware_restock(obs)

                # LLM every 3 days
                llm_action = RestockAction(orders=[], reasoning="skipped")
                if step % LLM_EVERY == 1:
                    llm_action = call_llm(obs)

                # Combine orders — rule-based takes priority
                final_dict  = {o.sku_id: o for o in rule_action.orders}
                budget_used = 0.0
                for o in rule_action.orders:
                    try:
                        sup = next(
                            s for s in obs.suppliers
                            if s.supplier_id == o.supplier_id
                        )
                        sku = next(
                            s for s in obs.skus
                            if s.sku_id == o.sku_id
                        )
                        budget_used += sku.unit_cost * sup.price_factor * o.quantity
                    except Exception:
                        pass

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
                        pass

                final_action = RestockAction(
                    orders=list(final_dict.values()),
                    reasoning="hybrid"
                )

                result      = env.step(final_action)
                reward      = result.reward.total_score
                done        = result.done
                final_score = reward
                steps_done  = step
                action_str  = action_to_str(final_action)

            except Exception as e:
                err        = str(e)
                reward     = 0.0
                done       = False
                action_str = "error"
                print(f"Step error: {e}", file=sys.stderr)

                try:
                    result = env.step(RestockAction(orders=[], reasoning="error"))
                    done   = result.done
                except Exception:
                    done = True

            rewards.append(reward)

            # ── MANDATORY [STEP] LINE ───────────
            print(
                f"[STEP] "
                f"step={step} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={'null' if err is None else err}",
                flush=True
            )

            if done:
                break

        success = final_score >= 0.5

    except Exception as e:
        print(f"Task {task_id} failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        final_score = 0.0
        success     = False

    finally:
        # ── MANDATORY [END] LINE ────────────────
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"
        print(
            f"[END] "
            f"success={'true' if success else 'false'} "
            f"steps={steps_done} "
            f"score={final_score:.2f} "
            f"rewards={rewards_str}",
            flush=True
        )

    return {
        "task_id": task_id,
        "score":   final_score,
        "success": success,
        "steps":   steps_done,
        "rewards": rewards,
    }


def main():
    """
    Main entry point.
    Runs all 3 tasks with mandatory stdout format.
    Saves baseline_scores.json.
    """
    results    = {}
    start_time = time.time()

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            result           = run_task(task_id)
            results[task_id] = result
        except Exception as e:
            print(f"Task {task_id} crashed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results[task_id] = {
                "task_id": task_id,
                "score":   0.0,
                "success": False,
                "steps":   0,
                "rewards": [],
            }

    elapsed = time.time() - start_time
    avg     = sum(r["score"] for r in results.values()) / len(results) if results else 0.0

    # Summary to stderr only (keep stdout clean for parser)
    print(f"\n=== FINAL SCORES ===", file=sys.stderr)
    for task_id, r in results.items():
        print(
            f"  {task_id}: {r['score']:.4f} "
            f"({'success' if r['success'] else 'failed'})",
            file=sys.stderr
        )
    print(f"  Average: {avg:.4f}", file=sys.stderr)
    print(f"  Time:    {elapsed:.1f}s", file=sys.stderr)

    # Save baseline scores
    output = {
        "model":         MODEL_NAME,
        "api_base_url":  API_BASE_URL,
        "scores": {
            task_id: r["score"]
            for task_id, r in results.items()
        },
        "average_score": round(avg, 4),
        "time_seconds":  round(elapsed, 1),
        "time_minutes":  round(elapsed / 60, 2),
    }

    try:
        with open("baseline_scores.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Saved baseline_scores.json", file=sys.stderr)
    except Exception as e:
        print(f"Could not save baseline_scores.json: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()