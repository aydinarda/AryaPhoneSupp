-- Arya Phones Supplier Selection Game - Supabase Schema
-- Run this SQL in Supabase Dashboard > SQL Editor

CREATE TABLE IF NOT EXISTS submissions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  team TEXT NOT NULL,
  player_name TEXT,
  selected_suppliers TEXT NOT NULL,
  objective TEXT NOT NULL DEFAULT 'manual',
  comment TEXT,
  profit DOUBLE PRECISION,
  utility DOUBLE PRECISION,
  env_avg DOUBLE PRECISION,
  social_avg DOUBLE PRECISION,
  cost_avg DOUBLE PRECISION,
  strategic_avg DOUBLE PRECISION,
  improvement_avg DOUBLE PRECISION,
  low_quality_avg DOUBLE PRECISION
);

-- Indexes (for performance)
CREATE INDEX IF NOT EXISTS submissions_team_created_at_idx ON submissions(team, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_profit ON submissions(profit DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_utility ON submissions(utility DESC);

-- RLS (Row Level Security) - everyone can read, everyone can write
ALTER TABLE submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable read access for all users" ON submissions
  FOR SELECT USING (true);

CREATE POLICY "Enable insert for all users" ON submissions
  FOR INSERT WITH CHECK (true);

-- Persistent game sessions
CREATE TABLE IF NOT EXISTS game_sessions (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  session_code TEXT NOT NULL UNIQUE,
  session_token TEXT NOT NULL UNIQUE,
  game_name TEXT NOT NULL,
  admin_name TEXT NOT NULL DEFAULT 'Admin',
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS session_players (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  session_token TEXT NOT NULL REFERENCES game_sessions(session_token) ON DELETE CASCADE,
  team_name TEXT NOT NULL,
  team_name_normalized TEXT NOT NULL,
  UNIQUE (session_token, team_name_normalized)
);

CREATE INDEX IF NOT EXISTS idx_game_sessions_code ON game_sessions(session_code);
CREATE INDEX IF NOT EXISTS idx_game_sessions_active ON game_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_session_players_token ON session_players(session_token);

ALTER TABLE game_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_players ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable read access for all users (game_sessions)" ON game_sessions
  FOR SELECT USING (true);

CREATE POLICY "Enable insert for all users (game_sessions)" ON game_sessions
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable read access for all users (session_players)" ON session_players
  FOR SELECT USING (true);

CREATE POLICY "Enable insert for all users (session_players)" ON session_players
  FOR INSERT WITH CHECK (true);
