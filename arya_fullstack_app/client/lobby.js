import { state, el, LOBBY_STORAGE_KEY } from "./state.js";
import { api } from "./api.js";
import {
  clearRoundTimer,
  clearRoundSync,
  renderAdminControls,
  renderRoundSummary,
  startRoundSync,
  loadCurrentRound,
  loadLatestMatch,
  resetBetaInputsInitialized,
} from "./round.js";
import { loadBenchmarkSummary } from "./benchmark.js";
import { renderDistributionChart } from "./distribution.js";

export function saveLobbyState() {
  const payload = {
    role: state.role,
    gameCode: state.gameCode,
    gameName: state.gameName,
    totalRounds: state.totalRounds,
    teamName: (el.teamName.value || "").trim(),
    playerName: (el.playerName.value || "").trim(),
  };
  localStorage.setItem(LOBBY_STORAGE_KEY, JSON.stringify(payload));
}

export function loadLobbyState() {
  try {
    const raw = localStorage.getItem(LOBBY_STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.gameName) el.adminGameName.value = saved.gameName;
    if (saved.totalRounds) el.adminNumberOfRounds.value = saved.totalRounds;
    if (saved.gameCode) el.playerJoinCode.value = saved.gameCode;
    if (saved.teamName) el.playerTeamName.value = saved.teamName;
    if (saved.playerName) {
      el.adminName.value = saved.playerName;
    }
  } catch (_err) {
    localStorage.removeItem(LOBBY_STORAGE_KEY);
  }
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
  renderRoundSummary();
  renderDistributionChart();
  startRoundSync();
  loadBenchmarkSummary();
}

export function showLobbyScreen() {
  clearRoundSync();
  clearRoundTimer();
  el.gameScreen.classList.add("hidden");
  el.lobbyScreen.classList.remove("hidden");
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
    state.gameName = session.game_name;
    state.gameCode = session.code;
    state.totalRounds = Number.isFinite(Number(session.number_of_rounds)) ? Number(session.number_of_rounds) : numberOfRounds;
    el.teamName.value = session.game_name;
    el.playerName.value = session.admin_name || "Admin";
    el.playerJoinCode.value = session.code;
    el.playerTeamName.value = session.game_name;
    saveLobbyState();
    showGameScreen();
    await loadCurrentRound();
    await loadLatestMatch();
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
    state.gameCode = session.code;
    state.gameName = session.game_name;
    state.totalRounds = Number.isFinite(Number(session.number_of_rounds)) ? Number(session.number_of_rounds) : state.totalRounds;
    el.teamName.value = teamName;
    el.playerName.value = "Player";
    saveLobbyState();
    showGameScreen();
    await loadCurrentRound();
    await loadLatestMatch();
  } catch (e) {
    el.playerHint.textContent = e.message || "Session code not found.";
  }
}
