-- Arya Phones Supplier Selection Game - Supabase Schema
-- Bu SQL'i Supabase Dashboard > SQL Editor'de çalıştır

CREATE TABLE IF NOT EXISTS submissions (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  team TEXT NOT NULL DEFAULT '(anonymous)',
  player_name TEXT NOT NULL DEFAULT '(anonymous)',
  selected_suppliers TEXT,
  objective TEXT,
  comment TEXT,
  feasible BOOLEAN DEFAULT FALSE,
  num_suppliers INTEGER DEFAULT 0,
  profit FLOAT8 DEFAULT 0.0,
  utility FLOAT8 DEFAULT 0.0,
  env_avg FLOAT8 DEFAULT 0.0,
  social_avg FLOAT8 DEFAULT 0.0,
  cost_avg FLOAT8 DEFAULT 0.0,
  strategic_avg FLOAT8 DEFAULT 0.0,
  improvement_avg FLOAT8 DEFAULT 0.0,
  low_quality_avg FLOAT8 DEFAULT 0.0
);

-- Index'ler (performans için)
CREATE INDEX IF NOT EXISTS idx_submissions_team ON submissions(team);
CREATE INDEX IF NOT EXISTS idx_submissions_created_at ON submissions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_profit ON submissions(profit DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_utility ON submissions(utility DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_feasible ON submissions(feasible);

-- RLS (Row Level Security) - herkes okuyabilir, herkes yazabilir
ALTER TABLE submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable read access for all users" ON submissions
  FOR SELECT USING (true);

CREATE POLICY "Enable insert for all users" ON submissions
  FOR INSERT WITH CHECK (true);
