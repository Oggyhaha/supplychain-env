---
title: SupplyChain Env
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# SupplyChain-Env

A real-world OpenEnv environment for training
and evaluating AI agents on supply chain
inventory management decisions.

## Why This Environment Matters

Every retailer warehouse faces these decisions daily:
- When to reorder before stockout happens
- How many units to order from which supplier
- How to balance limited budget across products
- How to survive supplier bankruptcy and demand surges

Poor decisions cost billions annually.
This environment lets AI agents learn optimal
strategies through reinforcement learning.

## Environment Design

### State Space
Each day the agent observes:
- Current inventory per SKU (units)
- 7-day demand forecast per SKU
- Available suppliers with status and lead times
- Pending orders not yet delivered
- Budget remaining
- Yesterday performance metrics

### Action Space
Each day the agent decides:
- Which SKUs to reorder
- How many units to order
- Which supplier to order from
- Or do nothing if stock is healthy

### Reward Function
Dense reward signal every day:
- Service level achieved (50% weight)
- Inventory health score (25% weight)
- Budget efficiency (10% weight)
- Stockout penalties (proportional)
- Overstock penalties (minor)

## Three Tasks

| Task | Difficulty | Days | SKUs | Key Challenge |
|------|-----------|------|------|---------------|
| task_easy | Easy | 30 | 1 | Basic reorder timing |
| task_medium | Medium | 60 | 4 | Budget allocation |
| task_hard | Hard | 90 | 5 | Supplier bankruptcy + holiday surge |

## Baseline Scores

| Task | Random Agent | LLM Baseline | Perfect Agent |
|------|-------------|--------------|---------------|
| task_easy | ~0.30 | ~0.72 | ~0.95 |
| task_medium | ~0.20 | ~0.58 | ~0.90 |
| task_hard | ~0.15 | ~0.42 | ~0.85 |

## API Reference

### Reset Environment
POST /reset/{task_id}

Returns initial observation for the episode.

### Take Action Step
POST /step/{task_id}

Body:
```json
{
  "orders": [
    {
      "sku_id": "LAPTOP-001",
      "supplier_id": "SUP-001",
      "quantity": 50
    }
  ],
  "reasoning": "Stock below reorder point"
}
```

### Get Current State
GET /state/{task_id}

### Health Check
GET /health

## Quick Start

### Local Setup
```bash
git clone https://huggingface.co/spaces/Oggyis1/supplychain-env
cd supplychain-env
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python server.py
```

### Run Baseline Inference
```bash
cp .env.example .env
# Add your Groq API key to .env
python inference.py
```

### Docker
```bash
docker build -t supplychain-env .
docker run -p 7860:7860 supplychain-env
```

## PyTorch Reward Predictor

Includes a neural network that predicts
reward scores from warehouse state features.
Used to improve agent decision quality.
```bash
python -m pytorch.reward_predictor
```

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- Pydantic v2 typed models
- PyTorch neural network
- OpenEnv spec compliant
- Docker containerized
- Groq LLM baseline