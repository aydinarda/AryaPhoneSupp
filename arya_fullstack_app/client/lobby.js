import { state, el, LOBBY_STORAGE_KEY } from "./state.js";
import { api } from "./api.js";
import {
  clearRoundSync,
  clearRoundTimer,
  renderAdminControls,
  renderRoundSummary,
  resetBetaInputsInitialized,
  startRoundSync,
} from "./round.js";
import { connectWS, disconnectWS } from "./ws.js";
import { loadBenchmarkSummary } from "./benchmark.js";
import { renderDistributionChart } from "./distribution.js";
import { clearJoinedTeams, renderSessionPlayers } from "./sessionPlayers.js";

export function saveLobbyState() {
  const selected = state.selected instanceof Set ? [...state.selected] : state.selected;
  const payload = {
    role: state.role,
    adminPlays: state.adminPlays,
    gameCode: state.gameCode,
    gameName: state.gameName,
    totalRounds: state.totalRounds,
    teamName: (el.teamName.value || "").trim(),
    playerName: (el.teamName.value || "").trim(),
    selected,
  };
  localStorage.setItem(LOBBY_STORAGE_KEY, JSON.stringify(payload));
}

export function loadLobbyState() {
  try {
    const raw = localStorage.getItem(LOBBY_STORAGE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw);
    if (saved.gameName) el.adminGameName.value = saved.gameName;
    if (saved.totalRounds) el.adminNumberOfRounds.value = saved.totalRounds;
    if (saved.gameCode) el.playerJoinCode.value = saved.gameCode;
    if (saved.teamName) el.playerTeamName.value = saved.teamName;
    if (Array.isArray(saved.selected)) {
      state.selected = new Set(saved.selected.map(String));
    } else if (saved.selected && typeof saved.selected === "object") {
      state.selected = saved.selected;
    }
    return saved;
  } catch (_err) {
    localStorage.removeItem(LOBBY_STORAGE_KEY);
    return null;
  }
}

export function restoreSavedGame(saved) {
  if (!saved || !saved.role || !saved.gameCode) return false;
  const role = saved.role === "admin" ? "admin" : saved.role === "player" ? "player" : null;
  if (!role) return false;

  state.role = role;
  state.adminPlays = Boolean(saved.adminPlays);
  state.gameCode = String(saved.gameCode || "").trim().toUpperCase();
  state.gameName = String(saved.gameName || "").trim();
  state.totalRounds = Number.isFinite(Number(saved.totalRounds)) ? Number(saved.totalRounds) : state.totalRounds;

  el.playerJoinCode.value = state.gameCode;
  if (role === "admin") {
    el.adminGameName.value = state.gameName;
    el.teamName.value = String(saved.teamName || state.gameName || "Admin").trim();
    el.playerTeamName.value = el.teamName.value;
  } else {
    el.teamName.value = String(saved.teamName || saved.playerName || "").trim();
    el.playerTeamName.value = el.teamName.value;
  }

  el.teamName.readOnly = true;
  el.teamName.style.opacity = "0.65";
  showGameScreen();
  return true;
}

export function renderSessionSummary() {
  if (!el.sessionSummary) return;
  if (!state.role) {
    el.sessionSummary.textContent = "";
    return;
  }
  const roleLabel = state.role === "admin" ? "Admin" : "Player";
  const gameLabel = state.gameName || "Untitled Game";
  const codeLabel = state.gameCode || "------";
  const roundsLabel = state.totalRounds ? ` | Rounds: ${state.totalRounds}` : "";
  el.sessionSummary.textContent = `${roleLabel} | ${gameLabel} | Code: ${codeLabel}${roundsLabel}`;
}

export function showGameScreen() {
  el.lobbyScreen.classList.add("hidden");
  el.gameScreen.classList.remove("hidden");
  renderSessionSummary();
  renderAdminControls();
  renderSessionPlayers();
  renderRoundSummary();
  renderDistributionChart();
  connectWS(state.gameCode);
  startRoundSync();
  loadBenchmarkSummary();
}

export function showLobbyScreen() {
  disconnectWS();
  clearRoundSync();
  clearRoundTimer();
  clearJoinedTeams();
  el.gameScreen.classList.add("hidden");
  el.lobbyScreen.classList.remove("hidden");
  el.teamName.readOnly = false;
  el.teamName.style.opacity = "";
}

export function clearLobbyHints() {
  el.adminHint.textContent = "";
  el.playerHint.textContent = "";
}

export async function enterAsAdmin() {
  clearLobbyHints();
  const gameName = (el.adminGameName.value || "").trim();
  const adminName = (el.adminName.value || "").trim();
  const roundsRaw = Number(el.adminNumberOfRounds?.value || 5);
  const numberOfRounds = Number.isFinite(roundsRaw) && roundsRaw >= 1 ? Math.floor(roundsRaw) : 5;

  if (!gameName) {
    el.adminHint.textContent = "Please enter a game name.";
    return;
  }

  try {
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        game_name: gameName,
        admin_name: adminName || "Admin",
        number_of_rounds: numberOfRounds,
      }),
    });

    resetBetaInputsInitialized();
    state.role = "admin";
    state.adminPlays = false;
    state.gameName = session.game_name;
    state.gameCode = session.code;
    state.totalRounds = Number.isFinite(Number(session.number_of_rounds)) ? Number(session.number_of_rounds) : numberOfRounds;
    clearJoinedTeams();
    el.teamName.value = session.game_name;
    el.teamName.readOnly = true;
    el.teamName.style.opacity = "0.65";
    el.playerJoinCode.value = session.code;
    el.playerTeamName.value = session.game_name;
    saveLobbyState();
    showGameScreen();
  } catch (e) {
    el.adminHint.textContent = e.message || "Could not create session.";
  }
}

export async function enterAsPlayer() {
  clearLobbyHints();
  const joinCode = (el.playerJoinCode.value || "").trim().toUpperCase();
  const teamName = (el.playerTeamName.value || "").trim();

  if (!joinCode) {
    el.playerHint.textContent = "A game code is required to join as a player.";
    return;
  }
  if (!teamName) {
    el.playerHint.textContent = "Please enter a team name.";
    return;
  }

  try {
    const session = await api(`/api/sessions/${joinCode}/join`, {
      method: "POST",
      body: JSON.stringify({ team_name: teamName }),
    });
    resetBetaInputsInitialized();
    state.role = "player";
    state.adminPlays = false;
    state.gameCode = session.code;
    state.gameName = session.game_name;
    state.totalRounds = Number.isFinite(Number(session.number_of_rounds)) ? Number(session.number_of_rounds) : state.totalRounds;
    clearJoinedTeams();
    el.teamName.value = teamName;
    el.teamName.readOnly = true;
    el.teamName.style.opacity = "0.65";
    saveLobbyState();
    showGameScreen();
  } catch (e) {
    el.playerHint.textContent = e.message || "Session code not found.";
  }
}
