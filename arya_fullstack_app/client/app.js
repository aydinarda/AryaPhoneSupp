import { state, el } from "./state.js";
import { loadLobbyState, showLobbyScreen, enterAsAdmin, enterAsPlayer, saveLobbyState } from "./lobby.js";
import { startRound, runMatchingNow, renderRoundSummary } from "./round.js";
import { loadConfigAndSuppliers, runManual, submit } from "./suppliers.js";
import { loadLeaderboard, renderLeaderboardScatter, ensureLeaderboardPlotUI } from "./leaderboard.js";
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
        return;
      }
    });
  });
}

function syncBetaFromInputs() {
  const a = parseFloat(el.betaAlpha?.value);
  const b = parseFloat(el.betaBeta?.value);
  if (Number.isFinite(a) && a > 0) state.betaAlpha = a;
  if (Number.isFinite(b) && b > 0) state.betaBeta = b;
  renderDistributionChart();

  if (state.gameCode) {
    api(`/api/sessions/${state.gameCode}/config`, {
      method: "PATCH",
      body: JSON.stringify({ beta_alpha: state.betaAlpha, beta_beta: state.betaBeta }),
    }).catch(() => {});
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

  // Beta distribution controls (admin only, but chart is visible to all)
  if (el.betaAlpha) el.betaAlpha.addEventListener("input", syncBetaFromInputs);
  if (el.betaBeta)  el.betaBeta.addEventListener("input", syncBetaFromInputs);

  el.teamName.addEventListener("change", saveLobbyState);
  el.playerName.addEventListener("change", saveLobbyState);
  el.playerJoinCode.addEventListener("change", () => {
    el.playerJoinCode.value = (el.playerJoinCode.value || "").toUpperCase();
  });

  document.getElementById("btnManual").addEventListener("click", runManual);
  document.getElementById("btnSubmit").addEventListener("click", submit);
  document.getElementById("btnRefreshLeaderboard").addEventListener("click", loadLeaderboard);

  el.sortSelect.addEventListener("change", (e) => {
    state.sortBy = e.target.value;
    loadLeaderboard();
  });

  el.feasibleOnly.addEventListener("change", (e) => {
    state.feasibleOnly = e.target.checked;
    loadLeaderboard();
  });

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
  renderDistributionChart();
  await loadConfigAndSuppliers();
  await runManual();
  await loadLeaderboard();
}

init();
