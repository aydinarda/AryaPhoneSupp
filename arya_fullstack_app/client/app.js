import { state, el } from "./state.js";
import { api } from "./api.js";
import { loadLobbyState, showLobbyScreen, enterAsAdmin, enterAsPlayer, saveLobbyState } from "./lobby.js";
import { startRound, runMatchingNow, renderRoundSummary } from "./round.js";
import { loadConfigAndSuppliers, runManual, submit } from "./suppliers.js?v=20260411-1";
import { loadLeaderboard, loadRoundHistory, renderLeaderboardScatter, ensureLeaderboardPlotUI } from "./leaderboard.js";
import { renderDistributionChart } from "./distribution.js";

function showSelectionPanel() {
  const panelGame = document.getElementById("panel-game");
  const panelLeaderboard = document.getElementById("panel-leaderboard");
  const leaderboardTab = document.querySelector('.tab[data-tab="leaderboard"]');
  panelLeaderboard.classList.add("hidden");
  panelGame.classList.remove("hidden");
  if (leaderboardTab) leaderboardTab.classList.remove("active");
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panelGame = document.getElementById("panel-game");
  const panelLeaderboard = document.getElementById("panel-leaderboard");

  tabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
      const key = tab.dataset.tab;

      if (key === "leaderboard") {
        const isShowingLeaderboard = !panelLeaderboard.classList.contains("hidden");
        if (isShowingLeaderboard) {
          showSelectionPanel();
          return;
        }

        panelGame.classList.add("hidden");
        panelLeaderboard.classList.remove("hidden");
        tab.classList.add("active");
        await loadLeaderboard();
        await loadRoundHistory();
        return;
      }
    });
  });
}

function readNumberInput(input, fallback) {
  if (!input) return fallback;
  if (Number.isFinite(input.valueAsNumber)) return input.valueAsNumber;

  const raw = String(input.value ?? "").trim().replace(",", ".");
  if (!raw) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatProbability(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  const pct = Math.round(numeric * 1000) / 10;
  return `${numeric} (${pct}%)`;
}

async function applyBetaDistribution() {
  const a = readNumberInput(el.betaAlpha, NaN);
  const b = readNumberInput(el.betaBeta, NaN);
  const d = readNumberInput(el.deltaInput, NaN);
  const auditProbability = readNumberInput(el.auditProbabilityInput, NaN);
  const catchProbability = readNumberInput(el.catchProbabilityInput, NaN);
  if (!Number.isFinite(a) || a <= 0 || !Number.isFinite(b) || b <= 0) {
    if (el.adminRoundHint) el.adminRoundHint.textContent = "Invalid α/β values.";
    return;
  }
  if (
    !Number.isFinite(auditProbability) ||
    auditProbability < 0 ||
    auditProbability > 1 ||
    !Number.isFinite(catchProbability) ||
    catchProbability < 0 ||
    catchProbability > 1
  ) {
    if (el.adminRoundHint) el.adminRoundHint.textContent = "Invalid investigation values. Use numbers between 0 and 1.";
    return;
  }
  state.betaAlpha = a;
  state.betaBeta = b;
  if (Number.isFinite(d) && d > 0) state.delta = d;
  state.auditProbability = auditProbability;
  state.catchProbability = catchProbability;
  renderDistributionChart();

  if (!state.gameCode) return;
  try {
    const body = {
      beta_alpha: a,
      beta_beta: b,
      audit_probability: auditProbability,
      catch_probability: catchProbability,
    };
    if (Number.isFinite(d) && d > 0) body.delta = d;
    const saved = await api(`/api/sessions/${state.gameCode}/config`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
    const savedAuditProbability = Number(saved.audit_probability ?? auditProbability);
    const savedCatchProbability = Number(saved.catch_probability ?? catchProbability);
    if (Number.isFinite(savedAuditProbability)) state.auditProbability = savedAuditProbability;
    if (Number.isFinite(savedCatchProbability)) state.catchProbability = savedCatchProbability;
    if (el.adminRoundHint) {
      const dStr = Number.isFinite(d) && d > 0 ? ` δ=${d}` : "";
      el.adminRoundHint.textContent = `Distribution applied (α=${a}, β=${b}${dStr}; investigation=${formatProbability(savedAuditProbability)}, detection=${formatProbability(savedCatchProbability)}).`;
    }
  } catch (e) {
    if (el.adminRoundHint) el.adminRoundHint.textContent = `Failed to apply: ${e.message}`;
  }
}

function setupEvents() {
  el.btnEnterAdmin.addEventListener("click", enterAsAdmin);
  el.btnEnterPlayer.addEventListener("click", enterAsPlayer);
  el.btnBackLobby.addEventListener("click", showLobbyScreen);
  if (el.btnStartRound) el.btnStartRound.addEventListener("click", startRound);
  if (el.btnRunMatch) el.btnRunMatch.addEventListener("click", runMatchingNow);

  if (el.btnBackToSelection) {
    el.btnBackToSelection.addEventListener("click", showSelectionPanel);
  }

  if (el.historyMetricSelect) {
    el.historyMetricSelect.addEventListener("change", (e) => {
      state.historyMetric = e.target.value;
      loadRoundHistory();
    });
  }

  // Beta distribution: applied explicitly via button, not on every keystroke
  if (el.btnApplyDistribution) el.btnApplyDistribution.addEventListener("click", applyBetaDistribution);

  el.teamName.addEventListener("change", saveLobbyState);
  el.playerName.addEventListener("change", saveLobbyState);
  el.playerJoinCode.addEventListener("change", () => {
    el.playerJoinCode.value = (el.playerJoinCode.value || "").toUpperCase();
  });

  document.getElementById("btnManual").addEventListener("click", runManual);
  document.getElementById("btnSubmit").addEventListener("click", submit);
  document.getElementById("btnRefreshLeaderboard").addEventListener("click", loadLeaderboard);

  ensureLeaderboardPlotUI();
  if (el.plotXSelect) {
    el.plotXSelect.addEventListener("change", (e) => {
      state.plotX = e.target.value;
      renderLeaderboardScatter(state.latestRows);
    });
  }

  if (el.plotYSelect) {
    el.plotYSelect.addEventListener("change", (e) => {
      state.plotY = e.target.value;
      renderLeaderboardScatter(state.latestRows);
    });
  }
}

async function init() {
  loadLobbyState();
  setupTabs();
  setupEvents();
  showLobbyScreen();
  renderRoundSummary();
  await loadConfigAndSuppliers();
  renderDistributionChart();
  await runManual();
  await loadLeaderboard();
}

init();
