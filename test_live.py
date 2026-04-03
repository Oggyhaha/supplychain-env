# test_live.py
import httpx

USERNAME = "Oggyis1"
BASE_URL = f"https://{USERNAME}-supplychain-env.hf.space"

print(f"Testing: {BASE_URL}")
print()

try:
    # Test 1 - Root endpoint
    print("Test 1: Root endpoint...")
    r = httpx.get(BASE_URL, timeout=30)
    print(f"  Status : {r.status_code}")
    print(f"  Response: {r.json()}")
    print()

    # Test 2 - Health check
    print("Test 2: Health check...")
    r = httpx.get(f"{BASE_URL}/health", timeout=30)
    print(f"  Status  : {r.status_code}")
    print(f"  Response: {r.json()}")
    print()

    # Test 3 - Reset easy task
    print("Test 3: Reset task_easy...")
    r = httpx.post(f"{BASE_URL}/reset/task_easy", timeout=30)
    print(f"  Status: {r.status_code}")
    data = r.json()
    obs  = data["observation"]
    print(f"  Day      : {obs['current_day']}")
    print(f"  Budget   : ${obs['budget_remaining']}")
    print(f"  Inventory: {obs['inventory']}")
    print(f"  Goal     : {obs['goal']}")
    print()

    # Test 4 - Take one step
    print("Test 4: Step with order...")
    action = {
        "orders": [
            {
                "sku_id": "LAPTOP-001",
                "supplier_id": "SUP-001",
                "quantity": 50
            }
        ],
        "reasoning": "Testing deployment"
    }
    r    = httpx.post(
        f"{BASE_URL}/step/task_easy",
        json=action,
        timeout=30
    )
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Reward  : {data['reward']['total_score']}")
    print(f"  Feedback: {data['reward']['feedback']}")
    print(f"  Done    : {data['done']}")
    print()

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("Deployment is working correctly!")
    print("=" * 50)

except Exception as e:
    print(f"Test FAILED: {e}")
    print("Check if Space is running on HuggingFace")