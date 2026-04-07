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
  const excluded = payload.excluded_infeasible_teams || [];
  const financials = payload.round_financials || {};
  const teamFinancials = financials.team_financials || [];

  el.matchingResultText.textContent = [
    `Solver: ${meta.solver || "-"}`,
    `Teams: ${meta.eligible_team_count ?? 0}`,
    `Users: ${meta.user_pool_count ?? 0}`,
    `δ: ${financials.delta ?? "-"}`,
    excluded.length ? `Excluded: ${excluded.join(", ")}` : null,
  ].filter(Boolean).join("  |  ");

  // Update table header for MNL
  const thead = el.matchingTableBody.closest("table")?.querySelector("thead tr");
  if (thead) {
    thead.innerHTML = "<th>Team</th><th>Demand Share</th><th>Eff. Users</th><th>Unit Margin</th><th>Realized Profit</th><th>Utility</th>";
  }

  if (teamFinancials.length) {
    el.matchingTableBody.innerHTML = teamFinancials
      .map((tf) => {
        const share = ((tf.demand_share ?? 0) * 100).toFixed(1);
        return `<tr>
          <td><strong>${tf.team}</strong></td>
          <td>${share}%</td>
          <td>${(tf.effective_users ?? 0).toFixed(1)}</td>
          <td>${(tf.unit_margin ?? 0).toFixed(1)}</td>
          <td><strong>${(tf.realized_profit ?? 0).toFixed(1)}</strong></td>
          <td>${(tf.realized_utility ?? 0).toFixed(1)}</td>
        </tr>`;
      })
      .join("");
  } else {
    // Fallback for old discrete-matching cached results (array of user IDs)
    if (thead) {
      thead.innerHTML = "<th>Team Product</th><th>Assigned Users</th>";
    }
    const loads = payload.market_loads || {};
    el.matchingTableBody.innerHTML = Object.entries(payload.market_to_users)
      .map(([teamId, users]) => {
        const load = loads[teamId] || {};
        const countLabel = `${load.assigned_count ?? (Array.isArray(users) ? users.length : users)}/${load.capacity ?? "-"}`;
        const userList = Array.isArray(users) ? (users.join(", ") || "-") : String(users);
        return `<tr><td>${teamId} (${countLabel})</td><td>${userList}</td></tr>`;
      })
      .join("");
  }

  // Per-segment breakdown
  _renderSegmentShares(payload.segment_shares || [], teamFinancials.map((tf) => tf.team));
}

function _renderSegmentShares(segmentShares, teams) {
  if (!el.segmentSharesHead || !el.segmentSharesBody) return;
  if (!segmentShares.length || !teams.length) {
    if (el.segmentSharesDetails) el.segmentSharesDetails.style.display = "none";
    return;
  }
  if (el.segmentSharesDetails) el.segmentSharesDetails.style.display = "";

  // Header: Segment | density | Team1 | Team2 | ...
  el.segmentSharesHead.innerHTML =
    `<th>#</th><th>Density</th>` + teams.map((t) => `<th>${t}</th>`).join("");

  el.segmentSharesBody.innerHTML = segmentShares
    .map((s) => {
      const cells = teams
        .map((t) => {
          const pct = s.shares?.[t] ?? 0;
          return `<td>${pct.toFixed(1)}%</td>`;
        })
        .join("");
      return `<tr><td>${s.segment_index}</td><td>${(s.density ?? 0).toFixed(3)}</td>${cells}</tr>`;
    })
    .join("");
}

// Flag: have we initialised the admin inputs from the server at least once this session?
let _betaInputsInitialized = false;

export function resetBetaInputsInitialized() {
  _betaInputsInitialized = false;
}

function _applyBetaFromData(data) {
  const a = Number(data.beta_alpha);
  const b = Number(data.beta_beta);
  const d = Number(data.delta);
  const aOk = Number.isFinite(a) && a > 0;
  const bOk = Number.isFinite(b) && b > 0;
  const dOk = Number.isFinite(d) && d > 0;
  const changed = (aOk && a !== state.betaAlpha) || (bOk && b !== state.betaBeta) || (dOk && d !== state.delta);
  if (aOk) state.betaAlpha = a;
  if (bOk) state.betaBeta = b;
  if (dOk) state.delta = d;
  if (changed) {
    // Sync admin inputs ONLY on first load so polls never overwrite what the admin typed
    if (!_betaInputsInitialized) {
      if (el.betaAlpha)  el.betaAlpha.value  = state.betaAlpha;
      if (el.betaBeta)   el.betaBeta.value   = state.betaBeta;
      if (el.deltaInput) el.deltaInput.value = state.delta;
      _betaInputsInitialized = true;
    }
    renderDistributionChart();
  }
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

  async function tick() {
    if (!state.gameCode || el.gameScreen.classList.contains("hidden")) return;
    try {
      await loadCurrentRound();
      if (state.role === "admin") await loadLatestMatch();
    } catch (_err) {
      // Ignore transient polling failures; explicit user actions still show errors.
    }
  }

  tick(); // run immediately on start, don't wait for first interval
  state.roundSyncId = setInterval(tick, ROUND_SYNC_INTERVAL_MS);
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
