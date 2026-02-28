"""
MaxUtilAgent.py (REWRITTEN)

Compatibility wrapper.
The real implementation now lives in MinCostAgent.py so the whole app shares:
- the same Policy class
- the same Excel loader/normalization
- the same utility definition

This file is kept so old imports don't break.
"""

from MinCostAgent import (  # noqa: F401
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    Policy,
    MaxUtilConfig,
    MaxUtilAgent,
    load_supplier_user_tables,
)
