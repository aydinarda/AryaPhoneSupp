import os
os.environ["ALLOCATION_MODE"] = "proportional"
os.environ["MARKET_SOFTMAX_TEMPERATURE"] = "1.0"

from app.matching_engine import run_market_matching
import json

# Test data: 2 users, 2 suppliers
users = [
    {
        "user_id": "u1",
        "choices": ["m1", "m2"],
        "utilities": {"m1": 8.0, "m2": 6.0},
        "price_sensitivity": 1.0,
        "sustainability_sensitivity": 0.5
    },
    {
        "user_id": "u2",
        "choices": ["m1", "m2"],
        "utilities": {"m1": 7.0, "m2": 9.0},
        "price_sensitivity": 0.5,
        "sustainability_sensitivity": 1.5
    }
]

market_options = [
    {
        "option_id": "m1",
        "capacity": 1.5,
        "priority": ["u2", "u1"],
        "price": 100.0,
        "sustainability": 4.0
    },
    {
        "option_id": "m2",
        "capacity": 1.5,
        "priority": ["u1", "u2"],
        "price": 50.0,
        "sustainability": 2.0
    }
]

result = run_market_matching(users, market_options)

print("=" * 60)
print("PROPORTIONAL ALLOCATION TEST")
print("=" * 60)
print(f"Allocation Mode: {result['meta']['allocation_mode']}")
print(f"User Count: {result['meta']['user_count']}")
print(f"Market Options: {result['meta']['market_option_count']}")
print()

print("FRACTIONAL ALLOCATIONS:")
print("-" * 60)
for user_id, allocations in result.get("fractional_allocations", {}).items():
    total = sum(allocations.values())
    print(f"\n{user_id}:")
    for market_id, fraction in sorted(allocations.items()):
        percentage = fraction * 100
        print(f"  {market_id}: {fraction:.4f} ({percentage:.1f}%)")
    print(f"  Total: {total:.4f}")

print()
print("PRIMARY MARKET (for backward compatibility):")
print("-" * 60)
for user_id, market_id in result["user_to_market"].items():
    print(f"{user_id} → {market_id}")

print()
print("MARKET LOADS:")
print("-" * 60)
for market_id, load in result["market_loads"].items():
    print(f"{market_id}: {load['assigned_count']} assigned / {load['capacity']} capacity")
