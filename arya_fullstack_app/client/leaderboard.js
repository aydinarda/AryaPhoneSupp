import { state, el, PLOT_COLUMNS } from "./state.js";
import { api, fmt } from "./api.js";

const PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#65a30d", "#c026d3", "#ea580c"];

function asRows(value) {
  return Array.isArray(value) ? value : [];
}

function cumulativeChartRows(rows) {
  return asRows(rows).map((row) => {
    const rounds = Number(row.rounds_played ?? 0);
    const divisor = rounds > 0 ? rounds : 1;
    const {
      total_profit: _totalProfit,
      total_market_share_pct: _totalMarketSharePct,
      total_realized_utility: _totalRealizedUtility,
      total_buyer_utility: _totalBuyerUtility,
      total_supplier_quality: _totalSupplierQuality,
      total_supplier_utility: _totalSupplierUtility,
      ...rest
    } = row;
    return {
      ...rest,
      avg_profit: Number(_totalProfit ?? 0) / divisor,
      avg_market_share_pct: Number(_totalMarketSharePct ?? 0) / divisor,
      avg_realized_utility: Number(_totalRealizedUtility ?? 0) / divisor,
      avg_buyer_utility: Number(_totalBuyerUtility ?? 0) / divisor,
      avg_supplier_quality: Number(_totalSupplierQuality ?? 0) / divisor,
      avg_supplier_utility: Number(_totalSupplierUtility ?? 0) / divisor,
    };
  });
}

function renderCumulativeMatchSummary(rows) {
  if (!el.cumulativeMatchBody) return;

  const sorted = cumulativeChartRows(rows)
    .slice()
    .sort((a, b) => Number(b.avg_profit ?? 0) - Number(a.avg_profit ?? 0));

  el.cumulativeMatchBody.innerHTML = sorted.length
    ? sorted.map((r) => `<tr>
        <td><strong>${r.team ?? "-"}</strong></td>
        <td>${fmt(r.avg_market_share_pct)}%</td>
        <td><strong>${fmt(r.avg_profit)}</strong></td>
        <td>${fmt(r.avg_realized_utility)}</td>
        <td><strong>${fmt(r.avg_buyer_utility)}</strong></td>
      </tr>`).join("")
    : '<tr><td colspan="5">No cumulative match summary yet.</td></tr>';
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

  const withFallback = available.length ? available : ["realized_profit", "buyer_utility", "realized_utility", "market_share_pct"];
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

export async function loadLeaderboard() {
  if (!state.gameCode) {
    state.latestRows = [];
    renderCumulativeMatchSummary([]);
    renderPlotSelectors([]);
    renderLeaderboardScatter([]);
    if (el.leaderboardBody) {
      el.leaderboardBody.innerHTML = '<tr><td colspan="9">Join or create a session to see the leaderboard.</td></tr>';
    }
    return;
  }

  try {
    const data = await api(`/api/sessions/${state.gameCode}/leaderboard`);
    const cumulativeRows = asRows(data.cumulative_leaderboard);
    const cumulativeRowsForChart = cumulativeChartRows(cumulativeRows);
    state.latestRows = cumulativeRowsForChart;

    const cumulativeKeys = ["avg_profit", "avg_buyer_utility", "avg_realized_utility", "avg_market_share_pct", "avg_supplier_quality", "avg_supplier_utility"];
    if (!cumulativeKeys.includes(state.plotX)) state.plotX = "avg_profit";
    if (!cumulativeKeys.includes(state.plotY)) state.plotY = "avg_market_share_pct";

    renderCumulativeMatchSummary(cumulativeRows);
    renderPlotSelectors(cumulativeRowsForChart);
    renderLeaderboardScatter(cumulativeRowsForChart);

    const averageRows = cumulativeRowsForChart
      .slice()
      .sort((a, b) => Number(b.avg_supplier_utility ?? 0) - Number(a.avg_supplier_utility ?? 0));

    el.leaderboardBody.innerHTML = averageRows.length
      ? averageRows.map((r, idx) => `<tr>
          <td>${idx + 1}</td>
          <td>${r.team ?? "-"}</td>
          <td>${r.rounds_played ?? 0}</td>
          <td><strong>${fmt(r.avg_supplier_utility)}</strong></td>
          <td>${fmt(r.avg_market_share_pct)}</td>
          <td>${fmt(r.avg_supplier_quality)}</td>
          <td>${fmt(r.avg_realized_utility)}</td>
          <td><strong>${fmt(r.avg_buyer_utility)}</strong></td>
          <td><strong>${fmt(r.avg_profit)}</strong></td>
        </tr>`).join("")
      : '<tr><td colspan="9">No cumulative leaderboard yet.</td></tr>';
  } catch (e) {
    state.latestRows = [];
    renderPlotSelectors([]);
    renderLeaderboardScatter([]);
    el.leaderboardBody.innerHTML = `<tr><td colspan="9">${e.message}</td></tr>`;
  }
}
