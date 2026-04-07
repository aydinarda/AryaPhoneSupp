import { state, el } from "./state.js";
import { api, fmt, metricCard } from "./api.js";
import { loadLeaderboard } from "./leaderboard.js";

export function renderMetrics(target, title, payload) {
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

export function renderSuppliers() {
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

export async function loadConfigAndSuppliers() {
  const [config, suppliers] = await Promise.all([api("/api/config"), api("/api/suppliers")]);
  state.config = config;
  state.suppliers = suppliers;
  if (Number.isFinite(Number(config.num_segments)) && config.num_segments > 0) {
    state.numSegments = Number(config.num_segments);
  }
  if (el.pricePerUser && (el.pricePerUser.value === "" || el.pricePerUser.value == null)) {
    el.pricePerUser.value = Number.isFinite(Number(config.price_per_user)) ? String(config.price_per_user) : "100";
  }
  el.configInfo.textContent = `Risk caps: avg env ≤ ${config.env_cap}, avg social ≤ ${config.social_cap} | Price: ${config.price_per_user} | Cost scale: ${config.cost_scale}`;
  renderSuppliers();
}

export function currentPayload() {
  const rawPrice = Number(el.pricePerUser?.value);
  const defaultPrice = Number(state.config?.price_per_user ?? 100);
  const pricePerUser = Number.isFinite(rawPrice) && rawPrice >= 0 ? rawPrice : defaultPrice;

  return {
    objective: state.objective,
    picks: [...state.selected],
    price_per_user: pricePerUser,
    beta_alpha: state.betaAlpha ?? 3.0,
    beta_beta: state.betaBeta ?? 3.0,
    delta: state.delta ?? 0.1,
  };
}

export async function runManual() {
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

export async function submit() {
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
