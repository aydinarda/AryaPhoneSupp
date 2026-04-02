import { el } from "./state.js";
import { api, fmt } from "./api.js";

export function renderBenchmarkPanel(title, result) {
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

export async function loadBenchmarkSummary() {
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
