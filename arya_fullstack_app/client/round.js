import { state, el, ROUND_SYNC_INTERVAL_MS } from "./state.js";
import { api } from "./api.js";
import { renderDistributionChart } from "./distribution.js";
import { renderConfigInfo } from "./suppliers.js";

let _latestMatchCompletedAt = "";
let _matchInFlight = false;

function applyRoundMeta(data) {
  const totalRounds = Number(data?.total_rounds);
  const trialRounds = Number(data?.trial_rounds);
  const scheduledRounds = Number(data?.scheduled_rounds);
  if (Number.isFinite(totalRounds)) state.totalRounds = totalRounds;
  if (Number.isFinite(trialRounds)) state.trialRounds = trialRounds;
  if (Number.isFinite(scheduledRounds)) state.scheduledRounds = scheduledRounds;
  else if (Number.isFinite(totalRounds) || Number.isFinite(trialRounds)) {
    state.scheduledRounds = (state.totalRounds || 0) + (state.trialRounds || 0);
  }
}

function roundLabel(roundNo) {
  const trialRounds = Math.max(0, Number(state.trialRounds) || 0);
  const totalRounds = Math.max(0, Number(state.totalRounds) || 0);
  if (!roundNo) {
    if (trialRounds > 0 && totalRounds > 0) {
      return `Round: not started | Trial ${trialRounds} + Game ${totalRounds}`;
    }
    const suffix = totalRounds ? ` / ${totalRounds}` : "";
    return `Round: not started${suffix}`;
  }
  if (trialRounds > 0 && roundNo <= trialRounds) {
    return `Trial Round ${roundNo}/${trialRounds}`;
  }
  const gameRoundNo = Math.max(1, roundNo - trialRounds);
  const suffix = totalRounds ? `/${totalRounds}` : "";
  return `Round ${gameRoundNo}${suffix}`;
}

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
  const label = roundLabel(state.roundNo);

  if (state.roundEndsAt) {
    const diffMs = new Date(state.roundEndsAt).getTime() - Date.now();
    const sec = Math.max(0, Math.floor(diffMs / 1000));
    const mm = String(Math.floor(sec / 60)).padStart(2, "0");
    const ss = String(sec % 60).padStart(2, "0");
    el.roundSummary.textContent = `${label} | Remaining: ${mm}:${ss}`;
    return;
  }

  if (!state.roundNo) {
    el.roundSummary.textContent = label;
    return;
  }

  el.roundSummary.textContent = `${label} | No timer`;
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
  if (el.sessionPlayersCard) {
    el.sessionPlayersCard.classList.toggle("hidden", !isAdmin);
  }
  renderAdminPlayMode();
}

export function renderAdminPlayMode() {
  const isAdmin = state.role === "admin";
  const adminIsPlaying = isAdmin && state.adminPlays;
  if (el.adminPlayToggle) {
    el.adminPlayToggle.checked = Boolean(state.adminPlays);
  }
  if (el.gameMainGrid) {
    el.gameMainGrid.classList.toggle("hidden", isAdmin && !adminIsPlaying);
  }
  if (el.adminObserverCard) {
    el.adminObserverCard.classList.toggle("hidden", !isAdmin || adminIsPlaying);
  }
  if (isAdmin && el.teamName) {
    el.teamName.readOnly = !adminIsPlaying;
    el.teamName.style.opacity = adminIsPlaying ? "" : "0.65";
  }
}

function matchCompletedAt(payload) {
  return String(payload?.meta?.completed_at || "");
}

export function renderMatchingResult(payload, options = {}) {
  if (!el.matchingResultText || !el.matchingTableBody) return;
  if (!payload || !payload.market_to_users) {
    if (_latestMatchCompletedAt && !options.force) return false;
    el.matchingResultText.textContent = "No matching result yet.";
    el.matchingTableBody.innerHTML = "";
    if (options.force) _latestMatchCompletedAt = "";
    return true;
  }

  const completedAt = matchCompletedAt(payload);
  if (_latestMatchCompletedAt && (!completedAt || completedAt < _latestMatchCompletedAt)) {
    return false;
  }
  if (completedAt) _latestMatchCompletedAt = completedAt;

  const meta = payload.meta || {};
  const excluded = payload.excluded_infeasible_teams || [];
  const financials = payload.round_financials || {};
  const teamFinancials = financials.team_financials || [];
  const audit = payload.audit || {};
  const caughtSuppliers = audit.caught_suppliers || [];
  const auditPenalizedTeams = audit.penalized_teams || [];
  const auditedCount = Object.values(audit.audited_suppliers || {}).filter(Boolean).length;
  const auditSummary = audit.audit_probability > 0
    ? [
        `Scrutiny: level=${audit.audit_probability ?? 0}, detection=${audit.catch_probability ?? 0}`,
        `Investigated suppliers: ${auditedCount}`,
        caughtSuppliers.length ? `Violations found: ${caughtSuppliers.join(", ")}` : "Violations found: none",
        auditPenalizedTeams.length ? `Audit penalty ${audit.utility_penalty ?? -10}: ${auditPenalizedTeams.join(", ")}` : null,
      ].filter(Boolean).join("  |  ")
    : "Scrutiny: off";

  el.matchingResultText.textContent = [
    `Solver: ${meta.solver || "-"}`,
    `Teams: ${teamFinancials.length || meta.eligible_team_count || 0}`,
    `Users: ${meta.user_pool_count ?? 0}`,
    `δ: ${financials.delta ?? "-"}`,
    `Sustainability sens.: ${financials.quality_sensitivity ?? "-"}`,
    auditSummary,
    excluded.length ? `Excluded: ${excluded.join(", ")}` : null,
  ].filter(Boolean).join("  |  ");

  // Update table header for MNL
  const thead = el.matchingTableBody.closest("table")?.querySelector("thead tr");
  if (thead) {
    thead.innerHTML = "<th>Team</th><th>Demand Share</th><th>Realized Profit</th><th>Market Utility</th>";
  }

  if (teamFinancials.length) {
    el.matchingTableBody.innerHTML = teamFinancials
      .map((tf) => {
        const share = ((tf.demand_share ?? 0) * 100).toFixed(1);
        return `<tr>
          <td><strong>${tf.team}</strong></td>
          <td>${share}%</td>
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
  return true;
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
  const isAdmin = state.role === "admin";
  const q = Number(data.quality_sensitivity ?? state.qualitySensitivity);
  const auditProbability = Number(data.audit_probability ?? state.auditProbability);
  const catchProbability = Number(data.catch_probability ?? state.catchProbability);
  const aOk = Number.isFinite(a) && a > 0;
  const bOk = Number.isFinite(b) && b > 0;
  const dOk = Number.isFinite(d) && d > 0;
  const qOk = Number.isFinite(q) && q >= 0;
  const auditOk = Number.isFinite(auditProbability) && auditProbability >= 0 && auditProbability <= 1;
  const catchOk = Number.isFinite(catchProbability) && catchProbability >= 0 && catchProbability <= 1;
  const chartChanged = (aOk && a !== state.betaAlpha) || (bOk && b !== state.betaBeta) || (dOk && d !== state.delta);
  if (aOk) state.betaAlpha = a;
  if (bOk) state.betaBeta = b;
  if (dOk) state.delta = d;
  if (qOk) state.qualitySensitivity = q;
  if (auditOk) state.auditProbability = auditProbability;
  if (catchOk) state.catchProbability = catchProbability;
  renderConfigInfo();
  // Sync admin inputs ONLY on first load so polls never overwrite what the admin typed
  if (!_betaInputsInitialized) {
    if (el.betaAlpha)  el.betaAlpha.value  = state.betaAlpha;
    if (el.betaBeta)   el.betaBeta.value   = state.betaBeta;
    if (el.deltaInput && dOk) el.deltaInput.value = state.delta;
    if (el.qualitySensitivityInput) el.qualitySensitivityInput.value = state.qualitySensitivity;
    if (el.auditProbabilityInput) el.auditProbabilityInput.value = state.auditProbability;
    if (el.catchProbabilityInput) el.catchProbabilityInput.value = state.catchProbability;
    if (!isAdmin || dOk) _betaInputsInitialized = true;
  }
  if (chartChanged) {
    renderDistributionChart();
  }
}

export function applyBetaFromServer(data) {
  _applyBetaFromData(data);
}

export async function loadCurrentRound() {
  if (!state.gameCode) return;
  try {
    const query = state.role === "admin" ? "?include_delta=true" : "";
    const data = await api(`/api/sessions/${state.gameCode}/rounds/current${query}`);
    const r = data.round;
    applyRoundMeta(data);
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
    applyRoundMeta(data);
    state.roundEndsAt = data.ends_at || null;
    renderRoundSummary();
    startRoundCountdown();
    if (el.adminRoundHint) {
      const gameRoundsLeft = Number.isFinite(Number(data.remaining_game_rounds)) ? Number(data.remaining_game_rounds) : null;
      const trialRoundsLeft = Number.isFinite(Number(data.remaining_trial_rounds)) ? Number(data.remaining_trial_rounds) : null;
      if (data.is_trial_round) {
        const suffix = trialRoundsLeft === null ? "" : ` Trial rounds left: ${trialRoundsLeft}.`;
        el.adminRoundHint.textContent = `Trial round ${data.trial_round_no} started.${suffix}`;
      } else if (Number(data.game_round_no) === 1 && Number(state.trialRounds) > 0) {
        const suffix = gameRoundsLeft === null ? "" : ` Remaining scored rounds: ${gameRoundsLeft}.`;
        el.adminRoundHint.textContent = `Scores reset. Game round 1 started.${suffix}`;
      } else {
        const suffix = gameRoundsLeft === null ? "" : ` Remaining scored rounds: ${gameRoundsLeft}.`;
        el.adminRoundHint.textContent = `Game round ${data.game_round_no} started.${suffix}`;
      }
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
  if (_matchInFlight) return;
  _matchInFlight = true;
  if (el.btnRunMatch) {
    el.btnRunMatch.disabled = true;
    el.btnRunMatch.textContent = "Running...";
  }
  if (el.adminRoundHint) el.adminRoundHint.textContent = "";

  try {
    const data = await api(`/api/sessions/${state.gameCode}/match`, {
      method: "POST",
      body: JSON.stringify({ round_no: state.roundNo }),
    });
    const rendered = renderMatchingResult(data.matching);
    if (rendered && el.adminRoundHint) {
      el.adminRoundHint.textContent = `Matching completed for round ${data.round_no}.`;
    }
  } catch (e) {
    if (el.adminRoundHint) {
      el.adminRoundHint.textContent = e.message || "Matching failed.";
    }
  } finally {
    _matchInFlight = false;
    if (el.btnRunMatch) {
      el.btnRunMatch.disabled = false;
      el.btnRunMatch.textContent = "Run Match";
    }
  }
}
