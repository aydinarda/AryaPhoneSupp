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
    metricCard("Cost / Unit", fmt(m.cost_per_unit)),
    metricCard("Utility", fmt(m.utility_total)),
    metricCard("Avg Env", fmt(m.avg_env)),
    metricCard("Avg Social", fmt(m.avg_social)),
    metricCard("Avg Cost", fmt(m.avg_cost)),
    metricCard("# Suppliers", String(Math.round(Number(m.k || 0)))),
  ].join("");
}

const CATEGORY_LABELS = {
  camera: "Select a Camera",
  keyboard: "Select a Keyboard",
  cable: "Select a Cable",
};
const CATEGORY_ORDER = ["camera", "keyboard", "cable"];

function clampPct(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(100, numeric));
}

function supplierBar(label, score, pct, tone) {
  const width = clampPct(pct);
  return `
    <div class="supplier-bar-row">
      <div class="supplier-bar-head">
        <span>${label}</span>
        <span>${fmt(score)}</span>
      </div>
      <div class="supplier-bar-track" aria-hidden="true">
        <div class="supplier-bar-fill ${tone}" style="width:${width}%;"></div>
      </div>
    </div>`;
}

function supplierCard(s, inputType, nameAttr, checked) {
  const id = String(s.supplier_id);
  const hasChildLabor = Number(s.child_labor || 0) >= 0.5;
  const hasBannedChem = Number(s.banned_chem || 0) >= 0.5;
  const cat = s.category || "";
  return `
    <label class="supplier-item">
      <input type="${inputType}" ${nameAttr} data-id="${id}" data-cat="${cat}" ${checked} />
      <div class="supplier-card-body">
        <div class="supplier-title"><strong>${id}</strong></div>
        <div class="supplier-bars">
          ${supplierBar("Environmental risk", s.env_risk, s.env_bad_pct, "risk")}
          ${supplierBar("Social risk", s.social_risk, s.social_bad_pct, "risk")}
          ${supplierBar("Cost score", s.cost_score, s.cost_bad_pct, "cost")}
          ${supplierBar("Strategic value", s.strategic, s.strategic_good_pct, "good")}
        </div>
        <div class="supplier-flags">
          <span class="supplier-flag ${hasChildLabor ? "flag-warn" : "flag-ok"}">Child labor: ${hasChildLabor ? "Yes" : "No"}</span>
          <span class="supplier-flag ${hasBannedChem ? "flag-warn" : "flag-ok"}">Banned chemicals: ${hasBannedChem ? "Yes" : "No"}</span>
        </div>
      </div>
    </label>`;
}

export function renderSuppliers() {
  const hasCategories = state.suppliers.some((s) => s.category);

  if (!hasCategories) {
    // Fallback: flat checkbox list (no category column in Excel)
    el.supplierList.innerHTML = state.suppliers
      .map((s) => {
        const id = String(s.supplier_id);
        const checked = state.selected instanceof Set && state.selected.has(id) ? "checked" : "";
        return supplierCard(s, "checkbox", ``, checked);
      })
      .join("");
    el.supplierList.querySelectorAll("input[type=checkbox]").forEach((input) => {
      input.addEventListener("change", (ev) => {
        const id = ev.target.dataset.id;
        if (!(state.selected instanceof Set)) state.selected = new Set();
        if (ev.target.checked) state.selected.add(id);
        else state.selected.delete(id);
      });
    });
    return;
  }

  // Group by category
  const grouped = {};
  for (const s of state.suppliers) {
    const cat = (s.category || "other").toLowerCase().trim();
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(s);
  }

  const orderedCats = [
    ...CATEGORY_ORDER.filter((c) => grouped[c]),
    ...Object.keys(grouped).filter((c) => !CATEGORY_ORDER.includes(c)),
  ];

  const sectionsHtml = orderedCats
    .map((cat) => {
      const label = CATEGORY_LABELS[cat] || `Select a ${cat.charAt(0).toUpperCase() + cat.slice(1)}`;
      const selectedId = state.selected[cat] || null;
      const options = grouped[cat]
        .map((s) => {
          const checked = selectedId === String(s.supplier_id) ? "checked" : "";
          return supplierCard(s, "radio", `name="supplier-${cat}"`, checked);
        })
        .join("");
      return `
        <div class="supplier-category-section">
          <div class="supplier-category-label">${label}</div>
          ${options}
        </div>`;
    })
    .join("");

  el.supplierList.innerHTML = `<div class="supplier-list-grid">${sectionsHtml}</div>`;

  el.supplierList.querySelectorAll("input[type=radio]").forEach((input) => {
    input.addEventListener("change", (ev) => {
      if (!ev.target.checked) return;
      state.selected[ev.target.dataset.cat] = ev.target.dataset.id;
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

function getMissingCategories() {
  const allCats = [
    ...new Set(
      state.suppliers
        .map((s) => (s.category || "").toLowerCase().trim())
        .filter(Boolean)
    ),
  ];
  return allCats.filter((cat) => !state.selected[cat]);
}

export function currentPayload() {
  const rawPrice = Number(el.pricePerUser?.value);
  const defaultPrice = Number(state.config?.price_per_user ?? 100);
  const pricePerUser = Number.isFinite(rawPrice) && rawPrice >= 0 ? rawPrice : defaultPrice;

  const picks =
    state.selected instanceof Set
      ? [...state.selected]
      : Object.values(state.selected).filter(Boolean);

  return {
    objective: state.objective,
    picks,
    price_per_user: pricePerUser,
    beta_alpha: state.betaAlpha ?? 3.0,
    beta_beta: state.betaBeta ?? 3.0,
    delta: state.delta ?? 0.1,
  };
}

export async function runManual() {
  const missing = getMissingCategories();
  if (missing.length > 0) {
    el.statusText.textContent = `You haven't selected from: ${missing.join(", ")}.`;
    return;
  }
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
  const missing = getMissingCategories();
  if (missing.length > 0) {
    el.statusText.textContent = `You haven't selected from: ${missing.join(", ")}.`;
    return;
  }
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
