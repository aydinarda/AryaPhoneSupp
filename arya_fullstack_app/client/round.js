import { state, el, ROUND_SYNC_INTERVAL_MS } from "./state.js";
import { api } from "./api.js";
import { renderDistributionChart } from "./distribution.js";

export function clearRoundTimer() {
  if (state.roundTimerId) {
    clearInterval(state.roundTimerId);
    state.roundTimerId = null;
  }
}

export function clearRoundSync() {
  if (state.roundSyncId) {
    clearInterval(state.roundSyncId);
    state.roundSyncId = null;
  }
}

export function renderRoundSummary() {
  if (!el.roundSummary) return;
  if (!state.roundNo) {
    const suffix = state.totalRounds ? ` / ${state.totalRounds}` : "";
    el.roundSummary.textContent = `Round: not started${suffix}`;
    return;
  }

  if (state.roundEndsAt) {
    const diffMs = new Date(state.roundEndsAt).getTime() - Date.now();
    const sec = Math.max(0, Math.floor(diffMs / 1000));
    const mm = String(Math.floor(sec / 60)).padStart(2, "0");
    const ss = String(sec % 60).padStart(2, "0");
    const suffix = state.totalRounds ? `/${state.totalRounds}` : "";
    el.roundSummary.textContent = `Round ${state.roundNo}${suffix} | Remaining: ${mm}:${ss}`;
    return;
  }

  const suffix = state.totalRounds ? `/${state.totalRounds}` : "";
  el.roundSummary.textContent = `Round ${state.roundNo}${suffix} | No timer`;
}

export function startRoundCountdown() {
  clearRoundTimer();
  if (!state.roundEndsAt) {
    renderRoundSummary();
    return;
  }

  state.roundTimerId = setInterval(() => {
    const diffMs = new Date(state.roundEndsAt).getTime() - Date.now();
    if (diffMs <= 0) {
      clearRoundTimer();
      state.roundEndsAt = null;
      renderRoundSummary();
      if (el.adminRoundHint) {
        el.adminRoundHint.textContent = "Timer ended. You can run Match now.";
      }
      return;
    }
    renderRoundSummary();
  }, 1000);
}

export function renderAdminControls() {
  if (!el.adminControls) return;
  const isAdmin = state.role === "admin";
  el.adminControls.classList.toggle("hidden", !isAdmin);
  if (el.matchingResultCard) {
    el.matchingResultCard.classList.toggle("hidden", !isAdmin);
  }
}

export function renderMatchingResult(payload) {
  if (!el.matchingResultText || !el.matchingTableBody) return;
  if (!payload || !payload.market_to_users) {
    el.matchingResultText.textContent = "No matching result yet.";
    el.matchingTableBody.innerHTML = "";
    return;
  }

  const meta = payload.meta || {};
  const excluded = payload.excluded_infeasible_users || [];
  const eligibleTeams = meta.eligible_team_count ?? 0;
  const usersInPool = meta.user_pool_count ?? meta.user_count ?? 0;
  el.matchingResultText.textContent = `Solver: ${meta.solver || "-"} | Team products: ${eligibleTeams} | Users in pool: ${usersInPool} | Matched users: ${meta.matched_count ?? 0} | Excluded infeasible teams: ${meta.infeasible_excluded_count ?? excluded.length}`;

  const entries = Object.entries(payload.market_to_users);
  const loads = payload.market_loads || {};
  el.matchingTableBody.innerHTML = entries
    .map(([marketId, users]) => {
      const load = loads[marketId] || {};
      const countLabel = `${load.assigned_count ?? (users || []).length}/${load.capacity ?? "-"}`;
      return `<tr><td>${marketId} (${countLabel})</td><td>${(users || []).join(", ") || "-"}</td></tr>`;
    })
    .join("");
}

function _applyBetaFromData(data) {
  const a = Number(data.beta_alpha);
  const b = Number(data.beta_beta);
  if (Number.isFinite(a) && a > 0) state.betaAlpha = a;
  if (Number.isFinite(b) && b > 0) state.betaBeta = b;
  if (el.betaAlpha) el.betaAlpha.value = state.betaAlpha;
  if (el.betaBeta)  el.betaBeta.value  = state.betaBeta;
  renderDistributionChart();
}

export async function loadCurrentRound() {
  if (!state.gameCode) return;
  try {
    const data = await api(`/api/sessions/${state.gameCode}/rounds/current`);
    const r = data.round;
    state.totalRounds = Number.isFinite(Number(data.total_rounds)) ? Number(data.total_rounds) : state.totalRounds;
    _applyBetaFromData(data);
    if (!r) {
      state.roundNo = null;
      state.roundEndsAt = null;
      renderRoundSummary();
      clearRoundTimer();
      return;
    }

    state.roundNo = r.round_no;
    state.roundEndsAt = r.ends_at || null;
    renderRoundSummary();
    startRoundCountdown();
  } catch (e) {
    if (el.adminRoundHint) {
      el.adminRoundHint.textContent = e.message || "Could not load round.";
    }
  }
}

export async function loadLatestMatch() {
  if (state.role !== "admin" || !state.gameCode) {
    renderMatchingResult(null);
    return;
  }

  try {
    const data = await api(`/api/sessions/${state.gameCode}/match/latest`);
    const latest = data.match?.result || null;
    renderMatchingResult(latest);
  } catch (_e) {
    renderMatchingResult(null);
  }
}

export function startRoundSync() {
  clearRoundSync();
  if (!state.gameCode) return;

  state.roundSyncId = setInterval(async () => {
    if (!state.gameCode || el.gameScreen.classList.contains("hidden")) {
      return;
    }

    try {
      await loadCurrentRound();
      if (state.role === "admin") {
        await loadLatestMatch();
      }
    } catch (_err) {
      // Ignore transient polling failures; explicit user actions still show errors.
    }
  }, ROUND_SYNC_INTERVAL_MS);
}

export async function startRound() {
  if (state.role !== "admin") return;
  if (!state.gameCode) return;
  if (el.adminRoundHint) el.adminRoundHint.textContent = "";

  const durationRaw = Number(el.roundTimerSeconds?.value || 0);
  const durationSeconds = Number.isFinite(durationRaw) && durationRaw > 0 ? Math.floor(durationRaw) : null;

  try {
    const data = await api(`/api/sessions/${state.gameCode}/rounds/start`, {
      method: "POST",
      body: JSON.stringify({
        duration_seconds: durationSeconds,
      }),
    });
    state.roundNo = data.round_no;
    state.totalRounds = Number.isFinite(Number(data.total_rounds)) ? Number(data.total_rounds) : state.totalRounds;
    state.roundEndsAt = data.ends_at || null;
    renderRoundSummary();
    startRoundCountdown();
    if (el.adminRoundHint) {
      const remaining = Number.isFinite(Number(data.remaining_rounds)) ? Number(data.remaining_rounds) : null;
      const suffix = remaining === null ? "" : ` Remaining rounds: ${remaining}.`;
      el.adminRoundHint.textContent = `Round ${data.round_no} started.${suffix}`;
    }
  } catch (e) {
    if (el.adminRoundHint) {
      el.adminRoundHint.textContent = e.message || "Could not start round.";
    }
  }
}

export async function runMatchingNow() {
  if (state.role !== "admin") return;
  if (!state.gameCode) return;
  if (el.adminRoundHint) el.adminRoundHint.textContent = "";

  try {
    const data = await api(`/api/sessions/${state.gameCode}/match`, {
      method: "POST",
      body: JSON.stringify({ round_no: state.roundNo }),
    });
    renderMatchingResult(data.matching);
    if (el.adminRoundHint) {
      el.adminRoundHint.textContent = `Matching completed for round ${data.round_no}.`;
    }
  } catch (e) {
    if (el.adminRoundHint) {
      el.adminRoundHint.textContent = e.message || "Matching failed.";
    }
  }
}
