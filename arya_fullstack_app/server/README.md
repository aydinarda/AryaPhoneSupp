# Mathematical Models for Supplier-Set Averaging Game

This document formalizes the two optimization models implemented in `app/mincost_agent.py`:
1. Max Profit benchmark
2. Max Utility benchmark

The formulations below are consistent with the current code path (`_solve_best_over_k`, `MaxProfitAgent`, `MaxUtilAgent`, and `manual_metrics`).

## 1. Notation

### Sets and indices
- `I = {1, ..., N}`: supplier index set
- `U`: user set from input table
- `U_s`: served users set used in utility calculation, where
  - `s = min(served_users, |U|)`
  - `U_s` is the last `s` rows of the user table (same as `_select_last_n_users` in code)

### Supplier attributes (for each `i in I`)
- `e_i`: environmental risk (`env_risk`)
- `r_i`: social risk (`social_risk`)
- `c_i`: cost score (`cost_score`)
- `t_i`: strategic importance (`strategic`)
- `m_i`: improvement potential (`improvement`)
- `q_i`: low product quality (`low_quality`)
- Optional binary-like flags in data:
  - `child_i` (`child_labor`)
  - `chem_i` (`banned_chem`)

### User weights (for each `u in U_s`)
- `w_env^u, w_soc^u, w_cost^u, w_str^u, w_imp^u, w_lq^u`

### Policy multipliers
- `a_env, a_soc, a_cost, a_str, a_imp, a_lq >= 0`
- Ban controls:
  - if `child_labor_penalty >= 0.5`, suppliers with `child_i >= 0.5` are forbidden
  - if `banned_chem_penalty >= 0.5`, suppliers with `chem_i >= 0.5` are forbidden

### Game constants
- `E_cap`: environmental cap (`env_cap`)
- `R_cap`: social cap (`social_cap`)
- `P`: price per user (`price_per_user`)
- `lambda`: cost scale (`cost_scale`)
- `s`: served user count as defined above

### Decision variable (for a fixed `k`)
- `y_i in {0,1}`: 1 if supplier `i` is selected
- Cardinality is not fixed globally. The algorithm enumerates `k = 1..N` and solves one MILP per `k`.

## 2. Common Feasibility Structure (per fixed `k`)

For each `k in {1, ..., N}`:

### Cardinality
`sum_{i in I} y_i = k`

### Risk cap constraints on averages
`(1/k) * sum_{i in I} e_i y_i <= E_cap`
`(1/k) * sum_{i in I} r_i y_i <= R_cap`

Equivalent linear form used in solver:
`sum_{i in I} e_i y_i <= E_cap * k`
`sum_{i in I} r_i y_i <= R_cap * k`

### Policy-based banning constraints
If active:
- `y_i = 0` for all `i` with `child_i >= 0.5`
- `y_i = 0` for all `i` with `chem_i >= 0.5`

## 3. Model A: Max Profit Benchmark

### 3.1 Fixed-`k` optimization problem
Code uses:
`max -sum_{i in I} c_i y_i`
which is equivalent to:
`min sum_{i in I} c_i y_i`
for fixed `k`.

Since `k` is fixed in each subproblem, minimizing `sum c_i y_i` is equivalent to minimizing average cost `(1/k) * sum c_i y_i`.

### 3.2 Selection across all `k`
After solving all feasible `k`, the algorithm picks the candidate with minimum
`avg_cost = (1/k) * sum_{i in I} c_i y_i`.

### 3.3 Reported totals
For the chosen set:
- `profit_per_user = P - lambda * avg_cost`
- `profit_total = s * profit_per_user`

Utility is also reported from the same chosen set using Model B utility formula (for dashboard completeness).

## 4. Model B: Max Utility Benchmark

### 4.1 Utility coefficient per supplier
For each supplier `i`, define:

`g_i = sum_{u in U_s} [`
`  w_env^u * a_env * e_i`
`+ w_soc^u * a_soc * r_i`
`+ w_cost^u * a_cost * c_i`
`+ w_str^u * a_str * t_i`
`+ w_imp^u * a_imp * m_i`
`+ w_lq^u  * a_lq  * q_i`
`]`

This exactly matches `util_num_coeff[i]` in code.

### 4.2 Fixed-`k` optimization problem
True utility for a selected set is:
`utility_total = (1/k) * sum_{i in I} g_i y_i`

For fixed `k`, denominator is constant, so solver maximizes numerator:
`max sum_{i in I} g_i y_i`

### 4.3 Selection across all `k`
After solving all feasible `k`, the algorithm compares candidates by:
`(1/k) * sum_{i in I} g_i y_i`
and picks the maximum.

### 4.4 Reported totals
For the chosen set, code recomputes:
- `utility_total = sum_{u in U_s} utility_per_user(u)` using averaged attributes
- `profit_total = s * (P - lambda * avg_cost)`

## 5. Manual Evaluation (No Optimizer)

For a user-provided supplier set `S_sel`:
- Compute averages of selected suppliers
- Check feasibility:
  - non-empty selection
  - average environmental risk <= `E_cap`
  - average social risk <= `R_cap`
- Compute `profit_total` and `utility_total` using same formulas

Important: manual evaluation does not call Gurobi, by design.

## 6. Why Enumerating `k` is Correct Here

Objectives depend on averages, which include division by `k` (selected count). If `k` were also variable in one model, objective/constraints become fractional.

Current implementation avoids nonlinear complexity by:
1. fixing `k`
2. solving a linear MILP for that `k`
3. taking best solution over `k = 1..N`

This is mathematically consistent with the implemented benchmark behavior.

## 7. Implementation Mapping

- Data normalization/loading: `load_supplier_user_tables`, `_normalize_supplier_columns`, `_normalize_user_columns`
- Manual model: `manual_metrics`
- MILP core: `_solve_best_over_k`
- Profit benchmark wrapper: `MaxProfitAgent.solve`
- Utility benchmark wrapper: `MaxUtilAgent.solve`

## 8. Validation Notes

- If Gurobi is unavailable: benchmark methods raise `RuntimeError("gurobipy is not available")`.
- If no feasible selection exists for all `k`: returns `feasible=False` with zero metrics.
- Ban constraints are active only when corresponding policy penalty >= 0.5.
