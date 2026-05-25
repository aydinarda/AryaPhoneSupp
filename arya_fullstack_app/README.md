# Arya Phones — Supplier Selection Game

A classroom simulation game where competing teams select smartphone component suppliers to maximise profit or utility, subject to environmental and social risk caps. Facilitators manage sessions and rounds; results appear on a live leaderboard updated in real time.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Frontend | Vanilla JS / HTML / CSS (no build step) |
| Database | PostgreSQL via [Supabase](https://supabase.com) |
| Real-time | WebSockets (built into FastAPI) |
| Charts | [Plotly](https://plotly.com/javascript/) (CDN) |
| Optimisation | [Gurobi](https://www.gurobi.com) MILP (optional) |

The frontend is served as static files by the same FastAPI process — there is no separate frontend server.

---

## External Dependencies

### Required
- **Supabase** — hosted PostgreSQL database. A free-tier project is sufficient.
  - Used for: sessions, rounds, submissions, matching results.
  - Credentials needed: `SUPABASE_URL`, `SUPABASE_ANON_KEY`.

### Optional
- **Gurobi** — commercial MILP solver.
  - Used only for `/api/benchmark` and `/api/benchmarks/both` endpoints.
  - All classroom gameplay (manual evaluation, session management, real-time leaderboard) works **without** a Gurobi licence.
  - Free academic licences available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

### CDN (no installation needed)
- `plotly-2.35.2.min.js` — charts and scatter plots in the frontend.
- Google Fonts (Inter) — UI typography.

---

## Project Structure

```
arya_fullstack_app/
├── server/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point, HTTP & WebSocket routes
│   │   ├── routers/sessions.py      # Session & round endpoints
│   │   ├── db.py                    # Supabase queries
│   │   ├── service.py               # Business logic & evaluation
│   │   ├── session_service.py       # Session / round lifecycle
│   │   ├── matching_engine.py       # MNL market matching orchestration
│   │   ├── mnl_market.py            # Multinomial Logit demand model
│   │   ├── optimization_controller.py  # Gurobi MILP wrapper
│   │   ├── live_state.py            # In-memory round state
│   │   ├── ws_manager.py            # WebSocket broadcast manager
│   │   ├── audit.py                 # Supplier audit logic
│   │   ├── settings.py              # Game constants (caps, prices, scales)
│   │   └── schemas.py               # Pydantic models
│   ├── tests/                       # Pytest suite + Jupyter simulation notebooks
│   ├── scripts/                     # Utility scripts
│   └── requirements.txt
├── client/                          # Vanilla JS / HTML / CSS frontend
│   ├── index.html                   # Main game UI
│   ├── game-finish.html             # Post-game results
│   └── *.js / styles.css
├── database_schema.sql              # All Supabase table definitions
├── Arya_Phones_Supplier_Selection.xlsx  # Supplier & user segment data
├── run_server.py                    # Convenience startup script
└── .env.example                     # Environment variable template
```

---

## Local Setup

### 1. Database

1. Create a free project on [supabase.com](https://supabase.com).
2. Open the **SQL Editor** and run the entire contents of `database_schema.sql`.
3. Copy your **Project URL** and **anon/public key** from *Settings → API*.

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

Alternatively, create a `secrets.toml` file in the project root:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
```

The server checks environment variables first, then falls back to `secrets.toml`.

### 3. Install & Run

```bash
cd server
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — the frontend loads automatically.

---

## Deploying to Render (Free Tier)

> Render's free tier spins the service down after ~15 minutes of inactivity. The first request after a cold start takes a few seconds.

1. Push the repository to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repository.
4. Configure the service:

   | Setting | Value |
   |---------|-------|
   | **Environment** | Python 3 |
   | **Root Directory** | `server` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

5. Add environment variables under **Environment → Add Environment Variable**:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`

6. Click **Deploy**. Render builds and starts the service; your app URL is shown in the dashboard.

> Note: Gurobi is not available on Render's free tier. Benchmark endpoints will return an error, but all gameplay features work fine.

---

## How to Play

### Facilitator

1. Open the app and select **Admin**.
2. Create a session — you'll receive a short **session code**.
3. Share the code with players.
4. Start a round (optionally timed). Players can now submit.
5. After submissions close, run **Market Matching** to allocate demand.
6. Review results on the leaderboard, then start the next round.

### Player

1. Select **Player**, enter a team name and player name, then enter the session code.
2. Wait in the lobby until the facilitator starts the round.
3. Select one or more suppliers from the table.
4. Set your **price per user** (affects market share in competitive mode).
5. Click **Evaluate** to preview metrics, then **Submit** to lock in.

### Constraints

| Metric | Cap |
|--------|-----|
| Avg Environmental Risk | ≤ 3.25 |
| Avg Social Risk | ≤ 3.5 |
| Min suppliers selected | 1 |

Only feasible submissions appear in the final ranking.

---

## API Reference

### Core

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config` | Game constants |
| `GET` | `/api/suppliers` | Supplier list with attributes |
| `POST` | `/api/manual-eval` | Evaluate a selection |
| `POST` | `/api/submit` | Submit to leaderboard |
| `GET` | `/api/leaderboard` | All submissions (filterable) |
| `POST` | `/api/matching` | Run MNL market matching |
| `GET` | `/api/benchmarks/both` | Max-profit & max-utility benchmarks *(Gurobi required)* |

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create session |
| `GET` | `/api/sessions/{code}` | Session details |
| `POST` | `/api/sessions/{code}/rounds` | Start a round |
| `GET` | `/api/sessions/{code}/rounds/active` | Active round info |
| `POST` | `/api/sessions/{code}/rounds/{n}/match` | Run matching for a round |
| `GET` | `/api/sessions/{code}/submissions` | Session submissions |
| `WS` | `/api/sessions/{code}/ws` | Real-time push (submissions, round changes) |

---

## Running Tests

```bash
cd server
pytest tests/
```

Simulation notebooks (`market_simulation.ipynb`, `round_simulation.ipynb`) in `tests/` can be run with Jupyter for offline scenario analysis.

---

## Mathematical Background

See [server/README.md](server/README.md) for the formal specification of:
- Max-Profit MILP benchmark
- Max-Utility MILP benchmark
- Multinomial Logit (MNL) demand model
