import os
os.environ["ALLOCATION_MODE"] = "proportional"
os.environ["MARKET_SOFTMAX_TEMPERATURE"] = "1.0"

from app.matching_engine import run_market_matching
import math

# Test data: 2 users, 3 suppliers (more varied)
users = [
    {
        "user_id": "u1",
        "choices": ["m1", "m2", "m3"],
        "utilities": {"m1": 8.0, "m2": 6.0, "m3": 5.0},
        "price_sensitivity": 1.0,
        "sustainability_sensitivity": 0.5
    },
    {
        "user_id": "u2",
        "choices": ["m1", "m2", "m3"],
        "utilities": {"m1": 7.0, "m2": 9.0, "m3": 8.0},
        "price_sensitivity": 0.5,
        "sustainability_sensitivity": 1.5
    }
]

market_options = [
    {
        "option_id": "m1",
        "capacity": 1.0,
        "priority": ["u2", "u1"],
        "price": 100.0,
        "sustainability": 4.0
    },
    {
        "option_id": "m2",
        "capacity": 1.2,
        "priority": ["u1", "u2"],
        "price": 50.0,
        "sustainability": 2.0
    },
    {
        "option_id": "m3",
        "capacity": 0.8,
        "priority": ["u1", "u2"],
        "price": 75.0,
        "sustainability": 3.5
    }
]

# Manual demand score calculation for reference
print("=" * 70)
print("DEMAND SCORE CALCULATIONS (for reference)")
print("=" * 70)
print()

for user in users:
    print(f"{user['user_id']}:")
    price_sens = user.get("price_sensitivity", 1.0)
    sust_sens = user.get("sustainability_sensitivity", 1.0)
    utilities = user.get("utilities", {})
    
    scores = {}
    for market in market_options:
        mid = market["option_id"]
        util = utilities.get(mid, 0.0)
        price = market.get("price", 0.0)
        sust = market.get("sustainability", 0.0)
        
        demand_score = util + (sust_sens * sust) - (price_sens * price)
        scores[mid] = demand_score
        print(f"  d({mid}) = {util:.1f} + {sust_sens:.1f}×{sust:.1f} - {price_sens:.1f}×{price:.1f} = {demand_score:.2f}")
    
    # Calculate softmax
    print(f"  Softmax probs:")
    max_score = max(scores.values())
    exps = {mid: math.exp(score - max_score) for mid, score in scores.items()}
    total_exp = sum(exps.values())
    probs = {mid: exp/total_exp for mid, exp in exps.items()}
    
    for mid in sorted(probs.keys()):
        print(f"    {mid}: {probs[mid]:.4f} ({probs[mid]*100:.1f}%)")
    print()

# Now run the allocation
print("=" * 70)
print("PROPORTIONAL ALLOCATION RESULT")
print("=" * 70)
print()

result = run_market_matching(users, market_options)

print(f"Allocation Mode: {result['meta']['allocation_mode']}")
print()

print("FRACTIONAL ALLOCATIONS:")
print("-" * 70)
for user_id in ["u1", "u2"]:
    allocations = result.get("fractional_allocations", {}).get(user_id, {})
    total = sum(allocations.values())
    print(f"\n{user_id}:")
    for market_id in sorted(allocations.keys()):
        fraction = allocations[market_id]
        percentage = fraction * 100
        print(f"  {market_id}: {fraction:.4f} ({percentage:.1f}%)")
    print(f"  Total: {total:.4f}")

print()
print("MARKET LOADS (after capacity scaling):")
print("-" * 70)
for market_id in ["m1", "m2", "m3"]:
    load = result["market_loads"][market_id]
    print(f"{market_id}: {load['assigned_count']:.1f} assigned / {load['capacity']:.1f} capacity")
