# Mathematical Models — Arya Phones Server

This document formalises the models implemented in the server.
Code references are consistent with the current implementation.

## 1. Notation

### Sets and indices
- `I = {1, ..., N}`: supplier index set
- `U`: full user/segment set loaded from the Excel workbook

### Supplier attributes (for each `i in I`)
- `e_i`: environmental risk (`env_risk`)
- `r_i`: social risk (`social_risk`)
- `c_i`: cost score (`cost_score`)
- Optional flags present in data but not used by the MILP:
  - `child_i` (`child_labor`), `chem_i` (`banned_chem`), `strategic_i`, `improvement_i`, `low_quality_i`

### User weights (for each `u in U`)
Used by the MILP and utility calculation:
- `w_env^u`, `w_soc^u` — quality preference weights
- `w_cost^u` — price sensitivity (used only in MNL demand model)

### Policy multipliers (`Policy` dataclass)
- `env_mult, social_mult, cost_mult >= 0`

### Game constants (`GameSettings`)
- `E_cap`: environmental cap (`env_cap`)
- `R_cap`: social cap (`social_cap`)
- `P`: price per user (`price_per_user`)
- `lambda`: cost scale (`cost_scale`)

### Decision variable (for a fixed `k`)
- `y_i in {0,1}`: 1 if supplier `i` is selected

---

## 2. Common Feasibility Structure (per fixed `k`)

### Cardinality
`sum_{i in I} y_i = k`

### Risk cap constraints
```
sum_{i in I} e_i * y_i  <=  E_cap * k
sum_{i in I} r_i * y_i  <=  R_cap * k
```

### Categorical mode (optional)
If the supplier table has a `category` column, exactly one supplier per category must be selected instead of free `k` enumeration.

---

## 3. Model A: Max Profit Benchmark

### 3.1 Objective per fixed `k`
```
min  sum_{i in I} c_i * y_i
```
Since `k` is fixed, minimising the sum is equivalent to minimising average cost.

### 3.2 Selection across all `k`
After solving all feasible `k`, pick the candidate with minimum `avg_cost = (1/k) * sum c_i y_i`.

### 3.3 Reported totals
```
profit_per_user = P - lambda * avg_cost
profit_total    = N * profit_per_user      (N = number of user segments)
```

---

## 4. Model B: Max Utility Benchmark

### 4.1 Utility coefficient per supplier
Define directional transforms:
```
env_ut_i  = 5 - e_i
soc_ut_i  = 5 - r_i
```

Density-weighted sums of user preferences:
```
W_env  = sum_{u in U} density(u) * w_env^u
W_soc  = sum_{u in U} density(u) * w_soc^u
```

Per-supplier utility coefficient (matches `util_num_coeff[i]` in `optimizer_common.py`):
```
g_i = W_env * env_mult * env_ut_i
    + W_soc * soc_mult * soc_ut_i
```

### 4.2 Objective per fixed `k`
```
max  sum_{i in I} g_i * y_i
```

### 4.3 Selection across all `k`
Pick the candidate with maximum `(1/k) * sum g_i y_i`.

### 4.4 Reported totals
`utility_total` is computed by `compute_utility_total` in `optimizer_common.py`:
```
utility_total = sum_{u in U} density(u) * [
    w_env^u * env_mult * (5 - avg_env)
  + w_soc^u * soc_mult * (5 - avg_social)
]
```
`profit_total` uses the same formula as Model A.

---

## 5. Manual Evaluation (No Optimizer)

For a user-provided supplier set `S_sel` (`manual_metrics` in `optimization_controller.py`):

1. Compute averages of selected suppliers.
2. Check feasibility: non-empty, avg env ≤ `E_cap`, avg social ≤ `R_cap`.
   In categorical mode: exactly one supplier per category required.
3. Build density-weighted `CustomerSegment` list (Beta(α, β) distribution over `w_cost`-sorted users).
4. Run `run_mnl_market` with a single `BuyerProfile` for this team (no competing buyers) to compute `profit_total` and `utility_total`.

Manual evaluation does not call Gurobi.

---

## 6. Why Enumerating `k` is Correct

Objectives depend on averages `(1/k) * sum ...`. If `k` were free, the objective would be fractional (non-linear). The implementation avoids this by:

1. Fixing `k`
2. Solving a linear MILP for that `k`
3. Comparing candidates over `k = 1..N` and taking the best

---

## 7. Implementation Mapping

| Concept | Location |
|---------|----------|
| Data loading | `load_supplier_user_tables` in `optimization_controller.py` |
| Column normalisation | `_normalize_supplier_columns`, `_normalize_user_columns` in `optimization_controller.py` |
| Manual evaluation | `manual_metrics` in `optimization_controller.py` |
| MILP core (k-enumeration + categorical) | `solve_best_over_k` in `optimizer_common.py` |
| Density-weighted utility | `compute_utility_total` in `optimizer_common.py` |
| Profit benchmark | `MinCostAgent.solve` in `mincost_optimizer.py` (`MaxProfitAgent` is a backward-compatible alias) |
| Utility benchmark | `MaxUtilityAgent.solve` in `max_utility_optimizer.py` (`MaxUtilAgent` is a backward-compatible alias) |

---

## 8. MNL Demand Model (`app/mnl_market.py`)

### 8.1 Overview

The Multinomial Logit (MNL) model determines how each customer segment splits
its demand among competing buyers in a single round.

### 8.2 Notation

- `S`: customer segments (rows of the User sheet, sorted by `w_cost`)
- `d_s`: density weight of segment `s` (from `BetaDensity`, normalised so `Σ d_s = 1`)
- `B`: competing buyers (teams) in the round
- `p_i`: price per user set by buyer `i`
- `delta`: global price sensitivity multiplier
- `quality_sensitivity`: global quality/sustainability multiplier

### 8.3 Quality utility (price-free)

Only environmental and social dimensions enter the quality score:

```
q_{i,s} = w_env_s   * (5 - avg_env_i)
         + w_social_s * (5 - avg_social_i)
```

### 8.4 MNL logit (includes price and affine transform)

Raw net utility before softmax:
```
raw_U_{i,s} = quality_sensitivity * q_{i,s}  -  delta * w_cost_s * price_i
```

An affine transform is applied for numerical stability and logit separation:
```
U_{i,s} = (raw_U_{i,s} + 50.0) * 1.2  +  utility_adjustment_i
```

`utility_adjustment` on `BuyerProfile` defaults to `0.0` and can be used to shift a buyer's logit.

### 8.5 MNL share for segment `s`

```
share_{i,s} = exp(U_{i,s}) / sum_{j in B} exp(U_{j,s})
```

Computed with numerically stable softmax (subtract max logit before `exp`).

Outside option: `u_outside = None` by default (all demand served by buyers).
Can be a `float` to model "no purchase" — its raw value is passed through the same affine transform before being added to the denominator.

### 8.6 Realised outcomes per buyer

```
demand_{i,s}        = (d_s / sum d) * share_{i,s}
realized_earnings_i = sum_s  price_i * demand_{i,s}
realized_utility_i  = sum_s  U_{i,s} * demand_{i,s}   (transformed logit)
```

`realized_utility` reflects the full transformed logit (quality + price effect).

### 8.7 Parallelisation

Each segment's MNL computation is independent. `run_mnl_market` dispatches all segments to a `ThreadPoolExecutor` and aggregates after all futures complete.

### 8.8 BuyerProfile fields

```python
@dataclass(frozen=True, slots=True)
class BuyerProfile:
    team_name: str
    price_per_user: float
    avg_env: float
    avg_social: float
    utility_adjustment: float = 0.0   # optional logit shift
```

Example:

```python
from app.mnl_market import BuyerProfile, run_mnl_market

profiles = [
    BuyerProfile("TeamA", price_per_user=90.0,  avg_env=1.5, avg_social=1.8),
    BuyerProfile("TeamB", price_per_user=110.0, avg_env=3.5, avg_social=3.0),
]

result = run_mnl_market(profiles, segments)   # segments: list[CustomerSegment]

for name, br in result.buyer_results.items():
    print(f"{name}: demand={br.total_demand:.3f}  "
          f"earnings={br.realized_earnings:.2f}  "
          f"utility={br.realized_utility:.4f}")
```

---

## 9. Validation Notes

- If Gurobi is unavailable: benchmark methods raise `RuntimeError("gurobipy is not available")`.
- If no feasible selection exists for any `k`: `solve_best_over_k` returns `None`; callers return `feasible=False` with zero metrics.
- MNL `total_demand` across all buyers sums to `≤ 1.0`; it is `< 1.0` when `u_outside` is set.
- Density weights are normalised inside `run_mnl_market`; raw Beta PDF values can be passed directly.
