export async function api(path, options = {}) {
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

export function fmt(value) {
  const n = Number(value ?? 0);
  return Number.isFinite(n) ? n.toFixed(3) : "0.000";
}

export function metricCard(key, value) {
  return `<div class="metric"><div class="k">${key}</div><div class="v">${value}</div></div>`;
}
