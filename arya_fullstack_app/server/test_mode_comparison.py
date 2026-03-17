import os

from app.matching_engine import run_market_matching

users = [
    {
        "user_id": "u1",
        "choices": ["m1", "m2", "m3"],
        "utilities": {"m1": 8.0, "m2": 6.0, "m3": 5.0},
        "price_sensitivity": 0.2,  # Düşük fiyat duyarlılığı
        "sustainability_sensitivity": 1.0
    },
    {
        "user_id": "u2",
        "choices": ["m1", "m2", "m3"],
        "utilities": {"m1": 7.0, "m2": 9.0, "m3": 8.0},
        "price_sensitivity": 0.2,
        "sustainability_sensitivity": 1.0
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
        "price": 90.0,
        "sustainability": 2.0
    },
    {
        "option_id": "m3",
        "capacity": 0.8,
        "priority": ["u1", "u2"],
        "price": 95.0,
        "sustainability": 3.5
    }
]

print("=" * 75)
print("MODE KARŞILAŞTIRMASI")
print("=" * 75)
print()

# PROPORTIONAL MODE
print("1. PROPORTIONAL MODE (ALLOCATION_MODE=proportional)")
print("-" * 75)
os.environ["ALLOCATION_MODE"] = "proportional"
result_prop = run_market_matching(users, market_options)

print("\nFractional Allocations:")
for uid in ["u1", "u2"]:
    allocs = result_prop.get("fractional_allocations", {}).get(uid, {})
    print(f"  {uid}: ", end="")
    parts = []
    for mid in sorted(allocs.keys()):
        if allocs[mid] > 0.001:
            parts.append(f"{mid}={allocs[mid]:.1%}")
    print(", ".join(parts) if parts else "No allocation")

# DETERMINISTIC MODE
print()
print("2. DETERMINISTIC MODE (ALLOCATION_MODE=deterministic)")
print("-" * 75)
os.environ["ALLOCATION_MODE"] = "deterministic"
result_det = run_market_matching(users, market_options)

print("\nUser → Primary Market:")
for uid in ["u1", "u2"]:
    mid = result_det["user_to_market"].get(uid)
    print(f"  {uid} → {mid}")

print()
print("=" * 75)
print("MARKET LOADS KARŞILAŞTIRMASI:")
print("=" * 75)
print(f"{'Mode':<20} {'m1 Load':<15} {'m2 Load':<15} {'m3 Load':<15}")
print("-" * 75)

prop_loads = ", ".join([
    f"m1={result_prop['market_loads']['m1']['assigned_count']:.1f}/{result_prop['market_loads']['m1']['capacity']:.1f}",
    f"m2={result_prop['market_loads']['m2']['assigned_count']:.1f}/{result_prop['market_loads']['m2']['capacity']:.1f}",
    f"m3={result_prop['market_loads']['m3']['assigned_count']:.1f}/{result_prop['market_loads']['m3']['capacity']:.1f}"
])

det_loads = ", ".join([
    f"m1={result_det['market_loads']['m1']['assigned_count']}/{result_det['market_loads']['m1']['capacity']}",
    f"m2={result_det['market_loads']['m2']['assigned_count']}/{result_det['market_loads']['m2']['capacity']}",
    f"m3={result_det['market_loads']['m3']['assigned_count']}/{result_det['market_loads']['m3']['capacity']}"
])

print(f"{'Proportional':<20} {prop_loads:<50}")
print(f"{'Deterministic':<20} {det_loads:<50}")
