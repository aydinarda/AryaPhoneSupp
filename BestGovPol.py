"""
BestGovPol.py (REWRITTEN)

This project previously contained a "Best governmental policy" page that generated
/ sampled policy candidates (often from policy_pool.xlsx) and solved a nested MILP.

The course flow has changed:
- Policy is now fixed (chosen in the Policy tab).
- Students solve a supplier-selection MILP under that fixed policy.

This file is kept only to avoid import errors if older UI code still imports it.
"""

import streamlit as st


def render_best_governmental_policy() -> None:
    st.markdown("### Best governmental policy (deprecated)")
    st.info(
        "This page is no longer used.\n\n"
        "✅ **New workflow:** fix a policy in the **Policy** tab, then choose suppliers in **Supplier Selection**.\n\n"
        "If you still see this tab in your UI, remove it from `UI.py`."
    )
