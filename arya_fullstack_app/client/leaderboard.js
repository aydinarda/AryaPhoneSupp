import { state, el, PLOT_COLUMNS } from "./state.js";
import { api, fmt } from "./api.js";

const PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#65a30d", "#c026d3", "#ea580c"];

function asRows(value) {
  return Array.isArray(value) ? value : [];
}

export function ensureLeaderboardPlotUI() {
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

export function renderPlotSelectors(rows) {
  ensureLeaderboardPlotUI();
  if (!el.plotXSelect || !el.plotYSelect) return;

  const available = Object.keys(PLOT_COLUMNS).filter((key) =>
    rows.some((r) => Number.isFinite(Number(r?.[key])))
  );

  const withFallback = available.length ? available : ["supplier_utility", "market_share_pct"];
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

export function renderLeaderboardScatter(rows) {
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
  const palette = PALETTE.slice(0, 8);
  const teamColors = new Map(teams.map((team, i) => [team, palette[i % palette.length]]));

  let minX = Math.min(...filtered.map((r) => Number(r[xKey])));
  let maxX = Math.max(...filtered.map((r) => Number(r[xKey])));
  let minY = Math.min(...filtered.map((r) => Number(r[yKey])));
  let maxY = Math.max(...filtered.map((r) => Number(r[yKey])));

  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }

  const px = (v) => ((v - minX) / (maxX - minX)) * 100;
  const py = (v) => (1 - (v - minY) / (maxY - minY)) * 100;

  const points = filtered.map((r) => {
    const team = r.team ?? "(anonymous)";
    const xVal = Number(r[xKey]);
    const yVal = Number(r[yKey]);
    const title = [
      `Team: ${team}`,
      r.round_no != null ? `Round: ${r.round_no}` : null,
      `${xLabel}: ${fmt(xVal)}`,
      `${yLabel}: ${fmt(yVal)}`,
      r.created_at ? `Created: ${new Date(r.created_at).toLocaleString()}` : null,
      r.selected_suppliers ? `Suppliers: ${r.selected_suppliers}` : null,
    ].filter(Boolean).join("\n");

    return `<div class="plot-point circle" style="left:${px(xVal)}%;top:${py(yVal)}%;background:${teamColors.get(team) ?? "#1f2937"};" title="${title.replace(/"/g, '&quot;')}"></div>`;
  }).join("");

  const legend = teams
    .map((team) => `<span style="display:inline-flex;align-items:center;gap:6px;margin-right:12px;"><span style="width:10px;height:10px;border-radius:999px;background:${teamColors.get(team)};display:inline-block;"></span>${team}</span>`)
    .join("");

  el.leaderboardScatter.innerHTML = `
    <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
      <strong>${xLabel} vs ${yLabel}</strong>
      <div style="font-size:12px;color:#6b7280;">Points are color-coded by team</div>
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

export async function loadRoundHistory() {
  if (!state.gameCode) return;

  let container = document.getElementById("roundHistoryChart");
  if (!container) {
    const panel = document.getElementById("panel-leaderboard");
    if (!panel) return;
    container = document.createElement("div");
    container.id = "roundHistoryChart";
    container.style.cssText = "height:280px;margin-bottom:12px;";
    const tableWrap = panel.querySelector(".table-wrap");
    if (tableWrap) panel.insertBefore(container, tableWrap);
    else panel.appendChild(container);
  }

  if (!window.Plotly) return;

  try {
    const data = await api(`/api/sessions/${state.gameCode}/rounds/history`);
    const { series = [], rounds = [] } = data;

    if (!series.length || !rounds.length) {
      container.innerHTML = '<p class="hint">No round data yet for this session.</p>';
      return;
    }

    const metric = state.historyMetric || "profit";
    const traces = series.map((s, i) => ({
      x: s.data.map((d) => d.round_no),
      y: s.data.map((d) => d[metric] ?? 0),
      mode: "lines+markers",
      name: s.team,
      line: { color: PALETTE[i % PALETTE.length], width: 2 },
      marker: { size: 6 },
    }));

    window.Plotly.react(container, traces, {
      margin: { t: 36, r: 16, b: 40, l: 56 },
      title: { text: `Round History - ${metric === "profit" ? "Realized Profit" : "Utility"} per Team`, font: { size: 13 } },
      xaxis: { title: "Round", dtick: 1, tickmode: "linear" },
      yaxis: { title: metric === "profit" ? "Profit" : "Utility" },
      legend: { orientation: "h", y: -0.25 },
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
    }, { responsive: true, displayModeBar: false });
  } catch (_e) {
    container.innerHTML = '<p class="hint">Could not load round history.</p>';
  }
}

export async function loadLeaderboard() {
  if (!state.gameCode) {
    state.latestRows = [];
    renderPlotSelectors([]);
    renderLeaderboardScatter([]);
    if (el.leaderboardBody) {
      el.leaderboardBody.innerHTML = '<tr><td colspan="8">Join or create a session to see the leaderboard.</td></tr>';
    }
    if (el.turnLeaderboardBody) {
      el.turnLeaderboardBody.innerHTML = '<tr><td colspan="9">Join or create a session to see the per-round leaderboard.</td></tr>';
    }
    return;
  }

  try {
    const data = await api(`/api/sessions/${state.gameCode}/leaderboard`);
    const cumulativeRows = asRows(data.cumulative_leaderboard);
    const turnRows = asRows(data.turn_leaderboard);
    state.latestRows = turnRows;

    renderPlotSelectors(state.latestRows);
    renderLeaderboardScatter(state.latestRows);

    el.leaderboardBody.innerHTML = cumulativeRows.length
      ? cumulativeRows.map((r, idx) => `<tr>
          <td>${idx + 1}</td>
          <td>${r.team ?? "-"}</td>
          <td>${r.rounds_played ?? 0}</td>
          <td><strong>${fmt(r.total_supplier_utility)}</strong></td>
          <td>${fmt(r.total_market_share_pct)}</td>
          <td>${fmt(r.total_supplier_quality)}</td>
          <td>${fmt(r.total_realized_utility)}</td>
          <td><strong>${fmt(r.total_profit)}</strong></td>
        </tr>`).join("")
      : '<tr><td colspan="8">No cumulative leaderboard yet.</td></tr>';

    if (el.turnLeaderboardBody) {
      el.turnLeaderboardBody.innerHTML = turnRows.length
        ? turnRows.map((r, idx) => `<tr>
            <td>${r.round_no ?? "-"}</td>
            <td>${idx + 1}</td>
            <td>${r.team ?? "-"}</td>
            <td><strong>${fmt(r.supplier_utility)}</strong></td>
            <td>${fmt(r.market_share_pct)}</td>
            <td>${fmt(r.supplier_quality)}</td>
            <td>${fmt(r.profit_cost_score)}</td>
            <td>${fmt(r.realized_utility)}</td>
            <td><strong>${fmt(r.realized_profit)}</strong></td>
          </tr>`).join("")
        : '<tr><td colspan="9">No per-round leaderboard yet.</td></tr>';
    }
  } catch (e) {
    state.latestRows = [];
    renderPlotSelectors([]);
    renderLeaderboardScatter([]);
    el.leaderboardBody.innerHTML = `<tr><td colspan="8">${e.message}</td></tr>`;
    if (el.turnLeaderboardBody) {
      el.turnLeaderboardBody.innerHTML = `<tr><td colspan="9">${e.message}</td></tr>`;
    }
  }
}
