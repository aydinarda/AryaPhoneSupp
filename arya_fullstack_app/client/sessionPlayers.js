import { state, el } from "./state.js";

function normalizePlayers(players) {
  const seen = new Set();
  const normalized = [];
  for (const raw of Array.isArray(players) ? players : []) {
    const name = String(raw || "").trim();
    const key = name.toLowerCase();
    if (!name || seen.has(key)) continue;
    seen.add(key);
    normalized.push(name);
  }
  return normalized;
}

export function setJoinedTeams(players) {
  state.joinedTeams = normalizePlayers(players);
  renderSessionPlayers();
}

export function clearJoinedTeams() {
  state.joinedTeams = [];
  renderSessionPlayers();
}

export function renderSessionPlayers() {
  if (!el.sessionPlayersCard || !el.sessionPlayersList || !el.sessionPlayersHint) return;

  const isAdmin = state.role === "admin";
  el.sessionPlayersCard.classList.toggle("hidden", !isAdmin);
  if (!isAdmin) return;

  const players = Array.isArray(state.joinedTeams) ? state.joinedTeams : [];
  if (!players.length) {
    el.sessionPlayersHint.textContent = "No teams joined yet.";
    el.sessionPlayersList.innerHTML = "";
    return;
  }

  el.sessionPlayersHint.textContent = `${players.length} team${players.length === 1 ? "" : "s"} joined this session.`;
  el.sessionPlayersList.innerHTML = players
    .map((team) => `<li class="session-player-item">${team}</li>`)
    .join("");
}
