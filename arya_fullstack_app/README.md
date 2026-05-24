# Arya Phones — Supplier Selection Game

A classroom simulation game in which competing teams select smartphone component suppliers to maximise profit or utility, subject to environmental and social risk caps.  Facilitators manage sessions and rounds; results are shown on a live leaderboard updated in real time.

## Architecture

```
arya_fullstack_app/
├── server/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, all HTTP & WS routes
│   │   ├── service.py              # Business logic (evaluation, benchmarks)
│   │   ├── matching_engine.py      # MNL market matching
│   │   ├── optimization_controller.py  # Gurobi-backed MILP benchmarks
│   │   ├── mnl_market.py           # Multinomial Logit demand model
│   │   ├── session_service.py      # Session / round management
│   │   ├── db.py                   # Supabase queries
│   │   ├── live_state.py           # In-memory state (round submissions)
│   │   ├── ws_manager.py           # WebSocket broadcast manager
│   │   ├── audit.py                # Supplier audit logic
│   │   ├── error_notifier.py       # E-mail error alerts
│   │   ├── settings.py             # Game constants (caps, prices, etc.)
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── routers/sessions.py     # Session & round API endpoints
│   ├── tests/                      # Pytest test suite + simulation notebooks
│   ├── scripts/                    # Utility scripts
│   └── requirements.txt
├── client/                         # Vanilla JS / HTML / CSS frontend
│   ├── index.html                  # Main game UI
│   ├── game-finish.html            # Post-game results page
│   ├── app.js / state.js / ws.js   # Core app logic
│   ├── round.js / lobby.js / suppliers.js / leaderboard.js
│   └── api.js / benchmark.js / distribution.js
├── database_schema.sql             # Supabase table definitions
├── Arya_Phones_Supplier_Selection.xlsx  # Supplier & user data
└── .env.example                    # Environment variable template
```

## Setup

### 1. Database

1. Create a [Supabase](https://supabase.com) project.
2. In the SQL Editor, run `database_schema.sql` to create all tables.
3. Note your **Project URL** and **anon/public key** from *Settings → API*.

### 2. Credentials

Copy `.env.example` to `.env` (or create `secrets.toml` in the project root):

```toml
# secrets.toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
```

The server checks environment variables first, then falls back to `secrets.toml`.

### 3. Backend

```bash
cd arya_fullstack_app/server
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — the frontend is served as static files by the same process.

> **Gurobi note:** the `/api/benchmark` endpoint requires a Gurobi licence.  Manual evaluation and all classroom gameplay work without it.

## How to Play

### Player flow

1. Enter a **Team Name** and **Player Name**, then join a session with the session code provided by the facilitator.
2. Wait in the lobby until the facilitator starts the round.
3. Select one or more suppliers from the table.
4. Set your **price per user** (affects MNL market share in competitive mode).
5. Click **Evaluate** to preview metrics, then **Submit** to lock in your choice.
6. Watch the live leaderboard update as teammates submit.

### Constraints (checked automatically)

| Metric | Cap |
|--------|-----|
| Avg Environmental Risk | ≤ 3.25 |
| Avg Social Risk | ≤ 3.5 |
| Min suppliers selected | 1 |

Only **feasible** submissions count in the final ranking.

### Scoring

- **Profit** = `served_users × (price_per_user − cost_scale × avg_cost)`
- **Utility** = demand-weighted quality score across customer segments (MNL model)

## Session Management (Facilitator)

Sessions are created at `/api/sessions`.  Each session has a short **session code** players use to join.  Rounds are started and closed through the sessions API; the frontend lobby page reflects round state in real time via WebSocket.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create a new session |
| `POST` | `/api/sessions/{code}/rounds` | Start a new round |
| `GET`  | `/api/sessions/{code}/rounds/active` | Active round info |
| `GET`  | `/api/sessions/{code}/submissions` | All submissions for a session |
| `WS`   | `/api/sessions/{code}/ws` | Real-time push (submissions, round changes) |

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check |
| `GET`  | `/api/config` | Game constants |
| `GET`  | `/api/suppliers` | Supplier list with attributes |
| `POST` | `/api/manual-eval` | Evaluate a supplier selection |
| `POST` | `/api/submit` | Submit to the leaderboard |
| `GET`  | `/api/leaderboard` | All submissions (filterable) |
| `GET`  | `/api/benchmarks/both` | Max-profit & max-utility benchmarks (requires Gurobi) |
| `POST` | `/api/benchmark` | Run a single benchmark objective |
| `POST` | `/api/matching` | Run MNL market matching |

## Tests

```bash
cd arya_fullstack_app/server
pytest tests/
```

The `tests/` directory also contains Jupyter notebooks (`market_simulation.ipynb`, `round_simulation.ipynb`) for offline scenario exploration.  Run them with Jupyter; output files are git-ignored.

## Mathematical Background

See [server/README.md](server/README.md) for the full formal specification of:
- Max-Profit MILP benchmark
- Max-Utility MILP benchmark
- Multinomial Logit (MNL) demand model
