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

A real-world supply chain inventory management
environment built for OpenEnv.

An AI agent manages warehouse inventory across
multiple products, suppliers, and demand patterns.

## What It Simulates

Every real warehouse faces these daily decisions:
- When to reorder products before stockout
- How many units to order from which supplier
- How to balance budget across multiple products
- How to handle supplier disruptions and demand spikes

## The 3 Tasks

### Task 1 - Easy - Single SKU Management
- 1 product over 30 days
- Stable demand pattern
- Budget: $50,000
- Expected score: 0.70 to 0.85

### Task 2 - Medium - Multi SKU Budget Management
- 4 products over 60 days
- Different demand patterns per product
- Budget: $150,000
- Expected score: 0.55 to 0.70

### Task 3 - Hard - Supplier Disruption Crisis
- 5 products over 90 days
- Supplier goes bankrupt on day 45
- Holiday demand surge on days 75 to 90
- Budget: $300,000
- Expected score: 0.40 to 0.55

## API Endpoints

Reset environment and start new episode:
POST /reset/{task_id}

Take one action step:
POST /step/{task_id}

Get current state:
GET /state/{task_id}

Check server health:
GET /health

## Action Space

The agent places purchase orders each day:
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

## Observation Space

Each day the agent receives:
- Current inventory levels per SKU
- 7-day demand forecast per SKU
- Available suppliers and their status
- Pending orders not yet delivered
- Budget remaining
- Yesterday performance metrics

## Reward Function

Daily rewards signal:
- Plus 0.5 times service level achieved today
- Plus 0.3 times inventory health score
- Minus 0.15 per SKU that stocked out
- Minus 0.05 per SKU that is overstocked
- Minus 0.3 if budget exceeded

## Setup Instructions

### Run Locally

Clone the repository:
git clone https://huggingface.co/spaces/Oggyis1/supplychain-env

Install dependencies:
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

Start server:
python server.py

Run baseline inference:
python inference.py

### Run With Docker

Build the container:
docker build -t supplychain-env .

Run the container:
docker run -p 7860:7860 supplychain-env

## Baseline Scores

| Task | Difficulty | Random Agent | LLM Baseline |
|------|------------|--------------|--------------|
| task_easy | Easy | 0.20 | 0.75 |
| task_medium | Medium | 0.25 | 0.62 |
| task_hard | Hard | 0.15 | 0.45 |

## Tech Stack

- Python 3.11
- FastAPI for environment server
- Pydantic for typed data models
- PyTorch for reward predictor neural network
- OpenEnv spec compliant
- Docker containerized
- Groq LLM for baseline inference