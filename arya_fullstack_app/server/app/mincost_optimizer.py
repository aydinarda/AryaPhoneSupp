from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .optimizer_common import GUROBI_AVAILABLE, compute_utility_total, solve_best_over_k


class MinCostAgent:
    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Any, cfg: Any):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available")
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

    def solve(self) -> Dict[str, Any]:
        cfg = self.cfg

        best = solve_best_over_k(
            self.suppliers,
            self.users,
            self.policy,
            served_users=int(cfg.served_users),
            env_cap=float(cfg.env_cap),
            social_cap=float(cfg.social_cap),
            output_flag=int(cfg.output_flag),
            objective_mode="profit",
        )

        if best is None:
            return {
                "feasible": False,
                "metrics": {
                    "k": 0.0,
                    "avg_env": 0.0,
                    "avg_social": 0.0,
                    "avg_cost": 0.0,
                    "profit_total": 0.0,
                    "utility_total": 0.0,
                },
            }

        served = min(int(cfg.served_users), int(len(self.users)))
        profit_per_user = float(cfg.price_per_user) - float(cfg.cost_scale) * float(best["avg_cost"])
        profit_total = served * profit_per_user

        utility_total = compute_utility_total(self.users, served, self.policy, best)

        feasible = (float(best["avg_env"]) <= float(cfg.env_cap) + 1e-12) and (
            float(best["avg_social"]) <= float(cfg.social_cap) + 1e-12
        )

        return {
            "feasible": bool(feasible),
            "metrics": {
                "k": float(best["k"]),
                "avg_env": float(best["avg_env"]),
                "avg_social": float(best["avg_social"]),
                "avg_cost": float(best["avg_cost"]),
                "profit_total": float(profit_total),
                "utility_total": float(utility_total),
            },
        }


# Backward-compatible alias
MaxProfitAgent = MinCostAgent
