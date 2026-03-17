import os
os.environ["ALLOCATION_MODE"] = "proportional"
os.environ["MARKET_SOFTMAX_TEMPERATURE"] = "1.0"

from app.matching_engine import run_market_matching

# Test: Lower is better logic
# m1: GOOD suppliers (low scores) 
# m2: BAD suppliers (high scores)
users = [
    {
        "user_id": "u1",
        "choices": ["m1", "m2"],
        "utilities": {"m1": 1.0, "m2": 4.0},  # m1 is good (1), m2 is bad (4)
        "price_sensitivity": 1.0,
        "sustainability_sensitivity": 1.0
    }
]

market_options = [
    {
        "option_id": "m1",
        "capacity": 2.0,
        "priority": ["u1"],
        "price": 50.0,
        "sustainability": 1.0  # Good (low score)
    },
    {
        "option_id": "m2",
        "capacity": 2.0,
        "priority": ["u1"],
        "price": 100.0,
        "sustainability": 4.0  # Bad (high score)
    }
]

print("=" * 70)
print("LOWER IS BETTER TEST")
print("=" * 70)
print()

print("Market Characteristics:")
print("-" * 70)
print("m1: utility=1.0, sustainability=1.0, price=50  (GOOD)")
print("m2: utility=4.0, sustainability=4.0, price=100 (BAD)")
print()

print("Expected behavior: u1 should prefer m1 (lower scores = better)")
print()

result = run_market_matching(users, market_options)

print("RESULTS:")
print("-" * 70)
print()

print("Fractional Allocations:")
for uid in ["u1"]:
    allocs = result.get("fractional_allocations", {}).get(uid, {})
    for mid in sorted(allocs.keys()):
        frac = allocs[mid]
        if frac > 0.001:
            print(f"  {uid} -> {mid}: {frac:.1%}")

print()
print("Primary Market (backward compatibility):")
for uid in ["u1"]:
    mid = result["user_to_market"].get(uid)
    print(f"  {uid} -> {mid}")

print()
print("Analysis:")
print("-" * 70)
allocs = result.get("fractional_allocations", {}).get("u1", {})
m1_alloc = allocs.get("m1", 0.0)
m2_alloc = allocs.get("m2", 0.0)

if m1_alloc > m2_alloc:
    print("✓ CORRECT: u1 prefers m1 (good/low-score supplier)")
elif m2_alloc > m1_alloc:
    print("✗ WRONG: u1 prefers m2 (bad/high-score supplier)")
else:
    print("? EQUAL: Both markets equally preferred")
