const state = {
  objective: "max_profit",
  role: null,
  gameCode: "",
  gameName: "",
  suppliers: [],
  selected: new Set(),
  config: null,
  sortBy: "profit",
  feasibleOnly: false,
  latestRows: [],
  plotX: "profit",
  plotY: "utility",
  roundNo: null,
  totalRounds: null,
  roundEndsAt: null,
  roundTimerId: null,
  roundSyncId: null,
};

const LOBBY_STORAGE_KEY = "arya_lobby_state_v1";
const ROUND_SYNC_INTERVAL_MS = 3000;

const el = {
  supplierList: document.getElementById("supplierList"),
  manualMetrics: document.getElementById("manualMetrics"),
  benchmarkMetrics: document.getElementById("benchmarkMetrics"),
  statusText: document.getElementById("statusText"),
  configInfo: document.getElementById("configInfo"),
  benchmarkProfitPanel: document.getElementById("benchmarkProfitPanel"),
  benchmarkUtilPanel: document.getElementById("benchmarkUtilPanel"),
  teamName: document.getElementById("teamName"),
  playerName: document.getElementById("playerName"),
  leaderboardBody: document.querySelector("#leaderboardTable tbody"),
  sortSelect: document.getElementById("sortSelect"),
  feasibleOnly: document.getElementById("feasibleOnly"),
  plotXSelect: document.getElementById("plotXSelect"),
  plotYSelect: document.getElementById("plotYSelect"),
  leaderboardScatter: document.getElementById("leaderboardScatter"),
  lobbyScreen: document.getElementById("lobbyScreen"),
  gameScreen: document.getElementById("gameScreen"),
  sessionSummary: document.getElementById("sessionSummary"),
  adminGameName: document.getElementById("adminGameName"),
  adminName: document.getElementById("adminName"),
  adminNumberOfRounds: document.getElementById("adminNumberOfRounds"),
  btnEnterAdmin: document.getElementById("btnEnterAdmin"),
  adminHint: document.getElementById("adminHint"),
  playerJoinCode: document.getElementById("playerJoinCode"),
  playerTeamName: document.getElementById("playerTeamName"),
  btnEnterPlayer: document.getElementById("btnEnterPlayer"),
  playerHint: document.getElementById("playerHint"),
  btnBackLobby: document.getElementById("btnBackLobby"),
  roundSummary: document.getElementById("roundSummary"),
  adminControls: document.getElementById("adminControls"),
  roundTimerSeconds: document.getElementById("roundTimerSeconds"),
  marketCapacity: document.getElementById("marketCapacity"),
  btnStartRound: document.getElementById("btnStartRound"),
  btnRunMatch: document.getElementById("btnRunMatch"),
  adminRoundHint: document.getElementById("adminRoundHint"),
  matchingResultCard: document.getElementById("matchingResultCard"),
  matchingResultText: document.getElementById("matchingResultText"),
  matchingTableBody: document.querySelector("#matchingTable tbody"),
};

const PLOT_COLUMNS = {
  profit: "Profit",
  utility: "Utility",
  env_avg: "Avg Env",
  social_avg: "Avg Social",
  cost_avg: "Avg Cost",
  strategic_avg: "Avg Strategic",
  improvement_avg: "Avg Improvement",
  low_quality_avg: "Avg Low Quality",
  num_suppliers: "# Suppliers",
};

function saveLobbyState() {
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

function loadLobbyState() {
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

function renderSessionSummary() {
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

function clearRoundTimer() {
  if (state.roundTimerId) {
    clearInterval(state.roundTimerId);
    state.roundTimerId = null;
  }
}

function clearRoundSync() {
  if (state.roundSyncId) {
    clearInterval(state.roundSyncId);
    state.roundSyncId = null;
  }
}

function renderRoundSummary() {
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

function startRoundCountdown() {
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

function renderAdminControls() {
  if (!el.adminControls) return;
  const isAdmin = state.role === "admin";
  el.adminControls.classList.toggle("hidden", !isAdmin);
  if (el.matchingResultCard) {
    el.matchingResultCard.classList.toggle("hidden", !isAdmin);
  }
}

function renderMatchingResult(payload) {
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

async function loadCurrentRound() {
  if (!state.gameCode) return;
  try {
    const data = await api(`/api/sessions/${state.gameCode}/rounds/current`);
    const r = data.round;
    state.totalRounds = Number.isFinite(Number(data.total_rounds)) ? Number(data.total_rounds) : state.totalRounds;
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

function startRoundSync() {
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

async function startRound() {
  if (state.role !== "admin") return;
  if (!state.gameCode) return;
  if (el.adminRoundHint) el.adminRoundHint.textContent = "";

  const durationRaw = Number(el.roundTimerSeconds?.value || 0);
  const durationSeconds = Number.isFinite(durationRaw) && durationRaw > 0 ? Math.floor(durationRaw) : null;
  const capacityRaw = Number(el.marketCapacity?.value || 8);
  const marketCapacity = Number.isFinite(capacityRaw) && capacityRaw > 0 ? Math.floor(capacityRaw) : 8;

  try {
    const data = await api(`/api/sessions/${state.gameCode}/rounds/start`, {
      method: "POST",
      body: JSON.stringify({
        duration_seconds: durationSeconds,
        market_capacity: marketCapacity,
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

async function runMatchingNow() {
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

async function loadLatestMatch() {
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

function renderBenchmarkPanel(title, result) {
  if (!result) return `<h4>${title}</h4><p class="hint">Not available.</p>`;
  if (result.available === false) {
    return `<h4 style="margin:0 0 8px;">${title}</h4><p class="hint">${result.error || "Solver not available"}</p>`;
  }
  const m = result.metrics || {};
  const feasibleBadge = result.feasible
    ? '<span style="color:green;font-weight:bold;">✓ Feasible</span>'
    : '<span style="color:red;font-weight:bold;">✗ Infeasible</span>';
  return `
    <h4 style="margin:0 0 8px;">${title}</h4>
    <table class="bm-table">
      <tbody>
        <tr><td>Status</td><td>${feasibleBadge}</td></tr>
        <tr><td>Optimal suppliers (k)</td><td>${Math.round(m.k || 0)}</td></tr>
        <tr><td>Avg Cost Score</td><td>${fmt(m.avg_cost)}</td></tr>
        <tr><td>Avg Env Risk</td><td>${fmt(m.avg_env)}</td></tr>
        <tr><td>Avg Social Risk</td><td>${fmt(m.avg_social)}</td></tr>
        <tr><td>Profit Total</td><td><strong>${fmt(m.profit_total)}</strong></td></tr>
        <tr><td>Utility Total</td><td><strong>${fmt(m.utility_total)}</strong></td></tr>
      </tbody>
    </table>`;
}

async function loadBenchmarkSummary() {
  if (!el.benchmarkProfitPanel) return;
  el.benchmarkProfitPanel.innerHTML = `<em class="hint">Computing benchmarks...</em>`;
  try {
    const data = await api("/api/benchmarks/both");
    el.benchmarkProfitPanel.innerHTML = renderBenchmarkPanel("Max Profit", data.max_profit);
    if (el.benchmarkUtilPanel) {
      el.benchmarkUtilPanel.innerHTML = renderBenchmarkPanel("Max Utility", data.max_utility);
    }
  } catch (e) {
    el.benchmarkProfitPanel.innerHTML = `<p class="hint">${e.message}</p>`;
  }
}

function showGameScreen() {
  el.lobbyScreen.classList.add("hidden");
  el.gameScreen.classList.remove("hidden");
  renderSessionSummary();
  renderAdminControls();
  renderRoundSummary();
  startRoundSync();
  loadBenchmarkSummary();
}

function showLobbyScreen() {
  clearRoundSync();
  clearRoundTimer();
  el.gameScreen.classList.add("hidden");
  el.lobbyScreen.classList.remove("hidden");
}

function clearLobbyHints() {
  el.adminHint.textContent = "";
  el.playerHint.textContent = "";
}

async function enterAsAdmin() {
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

async function enterAsPlayer() {
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

function ensureLeaderboardPlotUI() {
  const panel = document.getElementById("panel-leaderboard");
  if (!panel) return;

  const actionsRow = panel.querySelector(".actions-row");
  if (!actionsRow) return;

  if (!el.plotXSelect) {
    const selectX = document.createElement("select");
    selectX.id = "plotXSelect";
    selectX.title = "X axis";
    actionsRow.appendChild(selectX);
    el.plotXSelect = selectX;
  }

  if (!el.plotYSelect) {
    const selectY = document.createElement("select");
    selectY.id = "plotYSelect";
    selectY.title = "Y axis";
    actionsRow.appendChild(selectY);
    el.plotYSelect = selectY;
  }

  if (!el.leaderboardScatter) {
    const scatter = document.createElement("div");
    scatter.id = "leaderboardScatter";
    scatter.className = "card leaderboard-scatter";
    const tableWrap = panel.querySelector(".table-wrap");
    if (tableWrap && tableWrap.parentNode) {
      tableWrap.parentNode.insertBefore(scatter, tableWrap);
    } else {
      panel.appendChild(scatter);
    }
    el.leaderboardScatter = scatter;
  }
}

function fmt(value) {
  const n = Number(value ?? 0);
  return Number.isFinite(n) ? n.toFixed(3) : "0.000";
}

function metricCard(key, value) {
  return `<div class="metric"><div class="k">${key}</div><div class="v">${value}</div></div>`;
}

function renderMetrics(target, title, payload) {
  if (!payload || !payload.metrics) {
    target.innerHTML = "";
    return;
  }
  const m = payload.metrics;
  const feasibleBadge = payload.feasible
    ? '<span style="color: green; font-weight: bold;">✓ Feasible</span>'
    : '<span style="color: red; font-weight: bold;">✗ Infeasible</span>';
  target.innerHTML = [
    metricCard(`${title}`, feasibleBadge),
    metricCard("Profit", fmt(m.profit_total)),
    metricCard("Utility", fmt(m.utility_total)),
    metricCard("Avg Env", fmt(m.avg_env)),
    metricCard("Avg Social", fmt(m.avg_social)),
    metricCard("Avg Cost", fmt(m.avg_cost)),
    metricCard("# Suppliers", String(Math.round(Number(m.k || 0)))),
  ].join("");
}

function renderSuppliers() {
  el.supplierList.innerHTML = state.suppliers
    .map((s) => {
      const id = String(s.supplier_id);
      const checked = state.selected.has(id) ? "checked" : "";
      const childLabor = Number(s.child_labor || 0) >= 0.5 ? "Yes" : "No";
      const bannedChem = Number(s.banned_chem || 0) >= 0.5 ? "Yes" : "No";
      return `
      <label class="supplier-item">
        <input type="checkbox" data-id="${id}" ${checked} />
        <div>
          <div><strong>${id}</strong></div>
          <div class="supplier-meta">
            Env: ${fmt(s.env_risk)} (${fmt(s.env_bad_pct)}% bad) | Social: ${fmt(s.social_risk)} (${fmt(s.social_bad_pct)}% bad) | Cost: ${fmt(s.cost_score)} (${fmt(s.cost_bad_pct)}% bad)
          </div>
          <div class="supplier-meta">
            Strategic: ${fmt(s.strategic)} (${fmt(s.strategic_good_pct)}% good) | Improvement: ${fmt(s.improvement)} (${fmt(s.improvement_good_pct)}% good) | Low Quality: ${fmt(s.low_quality)} (${fmt(s.low_quality_bad_pct)}% bad)
          </div>
          <div class="supplier-meta">
            Child labor: ${childLabor} | Banned chemicals: ${bannedChem}
          </div>
        </div>
      </label>`;
    })
    .join("");

  el.supplierList.querySelectorAll("input[type=checkbox]").forEach((input) => {
    input.addEventListener("change", (ev) => {
      const id = ev.target.dataset.id;
      if (ev.target.checked) state.selected.add(id);
      else state.selected.delete(id);
    });
  });
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function loadConfigAndSuppliers() {
  const [config, suppliers] = await Promise.all([api("/api/config"), api("/api/suppliers")]);
  state.config = config;
  state.suppliers = suppliers;
  el.configInfo.textContent = `Served users: ${config.served_users} | Risk caps: avg env ≤ ${config.env_cap}, avg social ≤ ${config.social_cap} | Price per user: ${config.price_per_user} | Cost scale: ${config.cost_scale}`;
  renderSuppliers();
}

function currentPayload() {
  return {
    objective: state.objective,
    picks: [...state.selected],
  };
}

async function runManual() {
  try {
    const res = await api("/api/manual-eval", {
      method: "POST",
      body: JSON.stringify(currentPayload()),
    });
    renderMetrics(el.manualMetrics, "Manual", res);
    el.statusText.textContent = res.feasible
      ? "Selection satisfies risk constraints."
      : "Selection violates risk constraints.";
  } catch (e) {
    el.statusText.textContent = e.message;
  }
}

async function runBenchmark() {
  try {
    const res = await api("/api/benchmark", {
      method: "POST",
      body: JSON.stringify({ objective: state.objective }),
    });
    renderMetrics(el.benchmarkMetrics, "Benchmark", res);
  } catch (e) {
    el.statusText.textContent = e.message;
  }
}

async function submit() {
  try {
    const sessionMeta = [
      state.role ? `role:${state.role}` : null,
      state.gameCode ? `code:${state.gameCode}` : null,
      state.gameName ? `game:${state.gameName}` : null,
    ].filter(Boolean).join(" | ");

    const payload = {
      ...currentPayload(),
      team: (el.teamName.value || "(anonymous)").trim(),
      player_name: (el.playerName.value || "(anonymous)").trim(),
      session_code: state.gameCode || null,
      round_no: state.roundNo,
      comment: sessionMeta || null,
    };
    await api("/api/submit", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    el.statusText.textContent = "Submission saved to leaderboard!";
    await loadLeaderboard();
  } catch (e) {
    el.statusText.textContent = e.message;
  }
}

async function loadLeaderboard() {
  try {
    const params = new URLSearchParams({
      sort_by: state.sortBy,
      feasible_only: state.feasibleOnly.toString(),
    });
    const data = await api(`/api/leaderboard?${params}`);
    const rows = (state.sortBy === "profit" ? data.top_profit : data.top_utility) || [];
    state.latestRows = data.latest || [];

    renderPlotSelectors(state.latestRows);
    renderLeaderboardScatter(state.latestRows);
    
    el.leaderboardBody.innerHTML = rows
      .map((r, idx) => {
        const createdAt = r.created_at ? new Date(r.created_at).toLocaleString() : "-";
        const feasibleIcon = r.feasible ? "✓" : "✗";
        const feasibleClass = r.feasible ? "feasible" : "infeasible";
        const suppliers = r.selected_suppliers ? r.selected_suppliers.split(",").slice(0, 5).join(", ") : "-";
        const supplierCount = r.num_suppliers || 0;
        return `<tr class="${feasibleClass}">
          <td>${idx + 1}</td>
          <td>${r.team ?? "-"}</td>
          <td>${r.player_name ?? "-"}</td>
          <td>${r.objective ?? "-"}</td>
          <td>${feasibleIcon}</td>
          <td><strong>${fmt(r.profit)}</strong></td>
          <td><strong>${fmt(r.utility)}</strong></td>
          <td>${supplierCount}</td>
          <td><small>${suppliers}${supplierCount > 5 ? "..." : ""}</small></td>
          <td>${fmt(r.env_avg)}</td>
          <td>${fmt(r.social_avg)}</td>
          <td>${fmt(r.cost_avg)}</td>
          <td><small>${createdAt}</small></td>
        </tr>`;
      })
      .join("");
  } catch (e) {
    state.latestRows = [];
    renderLeaderboardScatter([]);
    el.leaderboardBody.innerHTML = `<tr><td colspan="13">${e.message}</td></tr>`;
  }
}

function renderPlotSelectors(rows) {
  ensureLeaderboardPlotUI();
  if (!el.plotXSelect || !el.plotYSelect) return;

  const available = Object.keys(PLOT_COLUMNS).filter((key) =>
    rows.some((r) => Number.isFinite(Number(r?.[key])))
  );

  const withFallback = available.length ? available : ["profit", "utility"];
  if (!withFallback.includes(state.plotX)) state.plotX = withFallback[0];
  if (!withFallback.includes(state.plotY)) state.plotY = withFallback[Math.min(1, withFallback.length - 1)];

  const optionHtml = withFallback
    .map((key) => `<option value="${key}">${PLOT_COLUMNS[key] ?? key}</option>`)
    .join("");

  el.plotXSelect.innerHTML = optionHtml;
  el.plotYSelect.innerHTML = optionHtml;
  el.plotXSelect.value = state.plotX;
  el.plotYSelect.value = state.plotY;
}

function renderLeaderboardScatter(rows) {
  if (!el.leaderboardScatter) return;

  if (!rows || rows.length === 0) {
    el.leaderboardScatter.innerHTML = '<p class="hint">No submissions to plot yet.</p>';
    return;
  }

  const xKey = state.plotX;
  const yKey = state.plotY;
  const xLabel = PLOT_COLUMNS[xKey] ?? xKey;
  const yLabel = PLOT_COLUMNS[yKey] ?? yKey;

  const filtered = rows.filter((r) => Number.isFinite(Number(r?.[xKey])) && Number.isFinite(Number(r?.[yKey])));
  if (!filtered.length) {
    el.leaderboardScatter.innerHTML = '<p class="hint">Selected columns do not have numeric data.</p>';
    return;
  }

  const teams = [...new Set(filtered.map((r) => r.team ?? "(anonymous)"))];
  const palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#65a30d"];
  const teamColors = new Map(teams.map((team, i) => [team, palette[i % palette.length]]));

  let minX = Math.min(...filtered.map((r) => Number(r[xKey])));
  let maxX = Math.max(...filtered.map((r) => Number(r[xKey])));
  let minY = Math.min(...filtered.map((r) => Number(r[yKey])));
  let maxY = Math.max(...filtered.map((r) => Number(r[yKey])));

  if (minX === maxX) {
    minX -= 1;
    maxX += 1;
  }
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }

  const px = (v) => ((v - minX) / (maxX - minX)) * 100;
  const py = (v) => (1 - (v - minY) / (maxY - minY)) * 100;

  const points = filtered.map((r) => {
    const team = r.team ?? "(anonymous)";
    const xVal = Number(r[xKey]);
    const yVal = Number(r[yKey]);
    const left = px(xVal);
    const top = py(yVal);
    const color = teamColors.get(team) ?? "#1f2937";
    const shapeClass = r.objective === "max_utility" ? "diamond" : "circle";
    const title = [
      `Team: ${team}`,
      `Mode: ${r.objective ?? "-"}`,
      `${xLabel}: ${fmt(xVal)}`,
      `${yLabel}: ${fmt(yVal)}`,
      `Created: ${r.created_at ? new Date(r.created_at).toLocaleString() : "-"}`,
      `Suppliers: ${r.selected_suppliers ?? "-"}`,
    ].join("\n");

    return `<div class="plot-point ${shapeClass}" style="left:${left}%;top:${top}%;background:${color};" title="${title.replace(/"/g, '&quot;')}"></div>`;
  }).join("");

  const legend = teams
    .map((team) => `<span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;"><span style="width:10px;height:10px;border-radius:999px;background:${teamColors.get(team)};display:inline-block;"></span>${team}</span>`)
    .join("");

  el.leaderboardScatter.innerHTML = `
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
      <strong>${xLabel} vs ${yLabel}</strong>
      <div style="font-size:12px;color:#6b7280;">Shape: circle=max_profit, diamond=max_utility</div>
    </div>
    <div style="font-size:12px;color:#374151;margin-bottom:8px;">${legend}</div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-bottom:6px;">
      <span>${yLabel}: ${fmt(minY)}</span>
      <span>${yLabel}: ${fmt(maxY)}</span>
    </div>
    <div class="plot-area" aria-label="Leaderboard scatter plot">
      ${points}
    </div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-top:6px;">
      <span>${xLabel}: ${fmt(minX)}</span>
      <span>${xLabel}: ${fmt(maxX)}</span>
    </div>
  `;
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  const panelGame = document.getElementById("panel-game");
  const panelLeaderboard = document.getElementById("panel-leaderboard");

  tabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const key = tab.dataset.tab;

      if (key === "leaderboard") {
        panelGame.classList.add("hidden");
        panelLeaderboard.classList.remove("hidden");
        await loadLeaderboard();
        return;
      }

      state.objective = key;
      panelLeaderboard.classList.add("hidden");
      panelGame.classList.remove("hidden");
      el.benchmarkMetrics.innerHTML = "";
      await runManual();
    });
  });
}

function setupEvents() {
  el.btnEnterAdmin.addEventListener("click", enterAsAdmin);
  el.btnEnterPlayer.addEventListener("click", enterAsPlayer);
  el.btnBackLobby.addEventListener("click", showLobbyScreen);
  if (el.btnStartRound) el.btnStartRound.addEventListener("click", startRound);
  if (el.btnRunMatch) el.btnRunMatch.addEventListener("click", runMatchingNow);

  el.teamName.addEventListener("change", saveLobbyState);
  el.playerName.addEventListener("change", saveLobbyState);
  el.playerJoinCode.addEventListener("change", () => {
    el.playerJoinCode.value = (el.playerJoinCode.value || "").toUpperCase();
  });

  document.getElementById("btnManual").addEventListener("click", runManual);
  document.getElementById("btnBenchmark").addEventListener("click", runBenchmark);
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
  await loadConfigAndSuppliers();
  await runManual();
  await loadLeaderboard();
}

init();
