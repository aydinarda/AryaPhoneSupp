import { state, el, PLOT_COLUMNS } from "./state.js";
import { api, fmt } from "./api.js";

const PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#65a30d", "#c026d3", "#ea580c"];

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
  const palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#65a30d"];
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
    const left = px(xVal);
    const top = py(yVal);
    const color = teamColors.get(team) ?? "#1f2937";
    const title = [
      `Team: ${team}`,
      `${xLabel}: ${fmt(xVal)}`,
      `${yLabel}: ${fmt(yVal)}`,
      `Created: ${r.created_at ? new Date(r.created_at).toLocaleString() : "-"}`,
      `Suppliers: ${r.selected_suppliers ?? "-"}`,
    ].join("\n");

    return `<div class="plot-point circle" style="left:${left}%;top:${top}%;background:${color};" title="${title.replace(/"/g, '&quot;')}"></div>`;
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
      title: { text: `Round History — ${metric === "profit" ? "Realized Profit" : "Utility"} per Team`, font: { size: 13 } },
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
        const supplierIds = r.selected_suppliers ? r.selected_suppliers.split(",").filter(Boolean) : [];
        const supplierCount = supplierIds.length;
        const suppliers = supplierIds.slice(0, 5).join(", ");
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
