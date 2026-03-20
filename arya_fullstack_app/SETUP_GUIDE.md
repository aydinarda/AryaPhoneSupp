# Arya Phones Full Stack App - Setup Guide

## 🎯 Features

### Single-Player Leaderboard System
- ✅ Everyone plays at the same time (independently)
- ✅ Everyone selects from the same supplier pool
- ✅ Everyone is matched with the same user pool
- ✅ Feasibility is checked against the constraints
- ✅ Real-time leaderboard
- ✅ Ranking by Profit and Utility
- ✅ Filter to show only feasible results

## 📦 Setup Steps

### 1. Supabase Database Setup

1. Create a [Supabase](https://supabase.com) account
2. Create a new project
3. Go to the SQL Editor
4. Run the SQL from the `database_schema.sql` file

### 2. Environment Variables

Create a `secrets.toml` file in the project root:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key-here"
```

**Where can you find your Supabase credentials?**
- Supabase Dashboard > Settings > API
- URL: Project URL
- Key: `anon` / `public` key

### 3. Backend Server

```powershell
cd arya_fullstack_app/server
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Frontend (Static Files)

The frontend is automatically served on the `/` route.
After the server starts, open: **http://localhost:8000**

## 🎮 How to Play

### Player Side

1. Enter **Team Name** and **Player Name**
2. **Select suppliers** (as many as you want)
3. Choose an **Objective** (Max Profit or Max Utility)
4. Use **Manual Evaluate** to see the results
5. Use **Submit** to save your result to the leaderboard

### Leaderboard

- **Sort by Profit/Utility**: Choose which metric to rank by
- **Show only feasible**: Show only results that satisfy the constraints
- **Top 10**: Displays the best 10 results
- **Color Coding**:
  - 🟢 Green = Feasible (constraints satisfied)
  - 🔴 Red = Infeasible (constraint violation)

## 🔧 Constraints

The following constraints are checked automatically:

- **Avg Environmental Risk** ≤ 2.75
- **Avg Social Risk** ≤ 3.0
- At least **1 supplier** must be selected

## 📊 Metrics

Calculated for each submission:

- ✅ **Feasibility** (whether the constraints are satisfied)
- 💰 **Profit** = served_users × (price_per_user - cost_scale × avg_cost)
- 😊 **Utility** = total benefit based on user weights
- 📈 **Averages**: env, social, cost, strategic, improvement, low_quality

## 🚀 Deployment (Future)

The app has not been deployed yet; it currently runs only locally.

For deployment:
- Backend: Railway, Render, Heroku
- Frontend: Netlify, Vercel
- Database: Supabase (already cloud-hosted)

## 📝 API Endpoints

- `GET /api/config` - Game settings
- `GET /api/suppliers` - Supplier list
- `POST /api/manual-eval` - Manual evaluation
- `POST /api/submit` - Submit to leaderboard
- `GET /api/leaderboard?sort_by=profit&feasible_only=false` - Leaderboard

## 🎯 Game Mechanics

### Single-Player Mode
- Each player plays independently
- Everyone uses the same supplier and user pool
- Everyone is subject to the same constraints
- There is competition on the leaderboard, but players do not affect one another

### How to Win
- **Max Profit Mode**: The player with the highest profit wins
- **Max Utility Mode**: The player with the highest utility wins
- **IMPORTANT**: Only **feasible** results are valid!

## 🐛 Troubleshooting

**Supabase connection error?**
- Check the `secrets.toml` file
- Make sure your Supabase credentials are correct

**Gurobi error?**
- Gurobi-backed optimization remains on the backend for internal optimization flows
- Manual evaluation endpoint does not require running benchmark endpoints

**Leaderboard is empty?**
- No one may have submitted yet
- Check the database connection

