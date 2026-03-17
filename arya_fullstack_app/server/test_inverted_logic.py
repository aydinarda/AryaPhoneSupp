import os

from app.matching_engine import run_market_matching

# Test: (5-x) inversion logic
# This mirrors what sessions.py does:
# - It calculates utilities as (5 - raw_score)
# - It calculates sustainability as (5-env + 5-social + ...) / 5
# - Price is actual value

users = [
    {
        "user_id": "u1",
        "choices": ["m1", "m2"],
        "utilities": {
            "m1": 4.0,  # High = good (sessions.py sends 5-raw_score)
            "m2": 1.0   # Low = bad
        },
        "price_sensitivity": 1.0,
        "sustainability_sensitivity": 1.0
    },
    {
        "user_id": "u2",
        "choices": ["m1", "m2"],
        "utilities": {
            "m1": 2.0,
            "m2": 3.0
        },
        "price_sensitivity": 0.5,
        "sustainability_sensitivity": 1.5
    }
]

market_options = [
    {
        "option_id": "m1",
        "capacity": 1.0,
        "priority": ["u1", "u2"],
        "price": 60.0,
        "sustainability": 4.0  # High = good (pre-inverted: (5-env+5-social+...)/5)
    },
    {
        "option_id": "m2",
        "capacity": 1.0,
        "priority": ["u1", "u2"],
        "price": 80.0,
        "sustainability": 2.0  # Low = bad
    }
]

print("=" * 70)
print("FORMULA VERIFICATION: (5-x) INVERSE LOGIC")
print("=" * 70)
print()

print("Utilities (pre-inverted by sessions.py, higher = good):")
print("  u1: m1=4.0 (GOOD), m2=1.0 (BAD)")
print("  u2: m1=2.0, m2=3.0")
print()

print("Sustainability (pre-inverted, higher = good):")
print("  m1: 4.0 (GOOD)")
print("  m2: 2.0 (BAD)")
print()

print("Price (lower = good):")
print("  m1: 60.0 (GOOD)")
print("  m2: 80.0 (BAD)")
print()

print("Expected behavior: Both users should prefer m1")
print()

# Test both modes
for mode in ["deterministic", "proportional"]:
    print("=" * 70)
    print(f"MODE: {mode.upper()}")
    print("=" * 70)
    
    os.environ["ALLOCATION_MODE"] = mode
    result = run_market_matching(users, market_options)
    
    print(f"Allocation result:")
    for uid in ["u1", "u2"]:
        if mode == "proportional":
            allocs = result.get("fractional_allocations", {}).get(uid, {})
            parts = []
            for mid in sorted(allocs.keys()):
                if allocs[mid] > 0.001:
                    parts.append(f"{mid}={allocs[mid]:.1%}")
            print(f"  {uid}: {', '.join(parts) if parts else 'No allocation'}")
        else:
            mid = result["user_to_market"].get(uid)
            print(f"  {uid} -> {mid}")
    
    print()
