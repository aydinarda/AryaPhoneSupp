import { state, el, LOBBY_STORAGE_KEY } from "./state.js";
import { api, fmt, metricCard } from "./api.js";
import { loadLeaderboard } from "./leaderboard.js";

let _submitResetTimer = null;
const INFEASIBLE_SUBMIT_MESSAGE = "Check your submission. It is not feasible!";

function persistSelectionState() {
  try {
    const raw = localStorage.getItem(LOBBY_STORAGE_KEY);
    const existing = raw ? JSON.parse(raw) : {};
    const selected = state.selected instanceof Set ? [...state.selected] : state.selected;
    localStorage.setItem(LOBBY_STORAGE_KEY, JSON.stringify({
      ...existing,
      role: state.role,
      adminPlays: state.adminPlays,
      gameCode: state.gameCode,
      gameName: state.gameName,
      totalRounds: state.totalRounds,
      teamName: (el.teamName?.value || existing.teamName || "").trim(),
      playerName: (el.teamName?.value || existing.playerName || "").trim(),
      selected,
    }));
  } catch (_err) {
    // Local persistence is a convenience only; gameplay still works without it.
  }
}

function fmtConfigValue(value, fallback = "-") {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Number.isInteger(n) ? String(n) : String(Math.round(n * 1000) / 1000);
}

export function renderConfigInfo() {
  if (!el.configInfo || !state.config) return;

  const config = state.config;
  el.configInfo.textContent = [
    `Risk caps: avg env ≤ ${fmtConfigValue(config.env_cap)}, avg social ≤ ${fmtConfigValue(config.social_cap)}`,
    `Cost scale: ${fmtConfigValue(config.cost_scale)}`,
    `Sustainability sensitivity: ${fmtConfigValue(state.qualitySensitivity)}`,
    `Scrutiny level: ${fmtConfigValue(state.auditProbability)}`,
    `Detection prob.: ${fmtConfigValue(state.catchProbability)}`,
  ].join(" | ");
}

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
    metricCard("Max Possible Profit", fmt(m.profit_total)),
    metricCard("Unit Cost", fmt(m.cost_per_unit)),
    metricCard("Market Utility", fmt(m.utility_total)),
    metricCard("Avg Env Risk", fmt(m.avg_env)),
    metricCard("Avg Social Risk", fmt(m.avg_social)),
    metricCard("Avg Cost", fmt(m.avg_cost)),
    metricCard("# Suppliers", String(Math.round(Number(m.k || 0)))),
  ].join("");
}

function setStatus(message, tone = "info") {
  if (!el.statusText) return;
  el.statusText.textContent = message || "";
  el.statusText.className = `hint submission-feedback is-${tone}`;
}

function resetSubmitButton() {
  if (_submitResetTimer) {
    clearTimeout(_submitResetTimer);
    _submitResetTimer = null;
  }
  if (!el.btnSubmit) return;
  el.btnSubmit.disabled = false;
  el.btnSubmit.textContent = "Submit";
  el.btnSubmit.classList.remove("is-submitting", "is-submitted");
}

function markSubmitPending() {
  if (!el.btnSubmit) return;
  if (_submitResetTimer) {
    clearTimeout(_submitResetTimer);
    _submitResetTimer = null;
  }
  el.btnSubmit.disabled = true;
  el.btnSubmit.textContent = "Submitting...";
  el.btnSubmit.classList.add("is-submitting");
  el.btnSubmit.classList.remove("is-submitted");
}

function flashSubmitSuccess() {
  if (!el.btnSubmit) return;
  el.btnSubmit.textContent = "Submitted";
  el.btnSubmit.classList.remove("is-submitting");
  el.btnSubmit.classList.add("is-submitted");
  _submitResetTimer = setTimeout(() => {
    resetSubmitButton();
  }, 1100);
}

const CATEGORY_LABELS = {
  camera: "Select a Camera",
  keyboard: "Select a Keyboard",
  cable: "Select a Cable",
};
const CATEGORY_ORDER = ["camera", "keyboard", "cable"];

function supplierBar(label, score, tone) {
  const width = Math.max(0, Math.min(100, (Number(score) / 5) * 100));
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
  const selectedClass = checked ? " is-selected" : "";
  return `
    <label class="supplier-item${selectedClass}">
      <input type="${inputType}" ${nameAttr} data-id="${id}" data-cat="${cat}" ${checked} />
      <span class="supplier-select-indicator ${inputType}" aria-hidden="true"></span>
      <div class="supplier-card-body">
        <div class="supplier-title"><strong>${id}</strong></div>
        <div class="supplier-bars">
          ${supplierBar("Environmental Risk Score", s.env_risk, "risk")}
          ${supplierBar("Social Risk Score", s.social_risk, "risk")}
          ${supplierBar("Cost score", s.cost_score, "cost")}
        </div>
        <div class="supplier-flags">
          <span class="supplier-flag ${hasChildLabor ? "flag-warn" : "flag-ok"}">Child Labor Existence: ${hasChildLabor ? "Yes" : "No"}</span>
          <span class="supplier-flag ${hasBannedChem ? "flag-warn" : "flag-ok"}">Banned Chemicals Existence: ${hasBannedChem ? "Yes" : "No"}</span>
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
        ev.target.closest(".supplier-item")?.classList.toggle("is-selected", ev.target.checked);
        persistSelectionState();
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
      el.supplierList
        .querySelectorAll(`input[type=radio][name="${ev.target.name}"]`)
        .forEach((radio) => {
          radio.closest(".supplier-item")?.classList.toggle("is-selected", radio.checked);
        });
      persistSelectionState();
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
  if (Number.isFinite(Number(config.quality_sensitivity))) {
    state.qualitySensitivity = Number(config.quality_sensitivity);
  }
  if (el.qualitySensitivityInput && (el.qualitySensitivityInput.value === "" || el.qualitySensitivityInput.value == null)) {
    el.qualitySensitivityInput.value = String(state.qualitySensitivity);
  }
  if (el.pricePerUser && (el.pricePerUser.value === "" || el.pricePerUser.value == null)) {
    el.pricePerUser.value = Number.isFinite(Number(config.price_per_user)) ? String(config.price_per_user) : "100";
  }
  renderConfigInfo();
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
    quality_sensitivity: state.qualitySensitivity ?? 1.0,
  };
}

async function evaluateCurrentSelection() {
  const res = await api("/api/manual-eval", {
    method: "POST",
    body: JSON.stringify(currentPayload()),
  });
  renderMetrics(el.manualMetrics, "Manual", res);
  return res;
}

export async function runManual() {
  const missing = getMissingCategories();
  if (missing.length > 0) {
    setStatus(`You haven't selected from: ${missing.join(", ")}.`, "warning");
    return;
  }
  try {
    const res = await evaluateCurrentSelection();
    setStatus(
      res.feasible
        ? "Selection satisfies risk constraints."
        : "Selection violates risk constraints.",
      res.feasible ? "success" : "error",
    );
  } catch (e) {
    setStatus(e.message, "error");
  }
}

export async function submit() {
  const missing = getMissingCategories();
  if (missing.length > 0) {
    setStatus(`You haven't selected from: ${missing.join(", ")}.`, "warning");
    return;
  }
  try {
    const manual = await evaluateCurrentSelection();
    if (!manual.feasible) {
      setStatus(INFEASIBLE_SUBMIT_MESSAGE, "error");
      return;
    }

    markSubmitPending();
    const sessionMeta = [
      state.role ? `role:${state.role}` : null,
      state.gameCode ? `code:${state.gameCode}` : null,
      state.gameName ? `game:${state.gameName}` : null,
    ].filter(Boolean).join(" | ");

    const payload = {
      ...currentPayload(),
      team: (el.teamName.value || "(anonymous)").trim(),
      player_name: (el.teamName.value || "(anonymous)").trim(),
      session_code: state.gameCode || null,
      round_no: state.roundNo,
      comment: sessionMeta || null,
    };
    await api("/api/submit", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setStatus("Submission saved to leaderboard!", "success");
    flashSubmitSuccess();
    loadLeaderboard().catch(() => {});
  } catch (e) {
    resetSubmitButton();
    setStatus(e.message, "error");
  }
}
