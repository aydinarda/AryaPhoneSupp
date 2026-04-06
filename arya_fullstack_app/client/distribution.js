import { state, el } from "./state.js";

/**
 * Evaluates Beta(alpha, beta) PDF at x using the log-gamma approach.
 * Pure JS — no scipy dependency.
 */
function logGamma(z) {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function betaPDF(x, alpha, beta) {
  if (x <= 0 || x >= 1) return 0;
  const logB = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  return Math.exp((alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - logB);
}

export function renderDistributionChart() {
  const container = el.distributionChart;
  if (!container || !window.Plotly) return;

  const alpha = state.betaAlpha;
  const beta  = state.betaBeta;
  const N     = state.numSegments;

  if (!alpha || !beta || alpha <= 0 || beta <= 0) {
    container.innerHTML = '<p class="hint">Invalid distribution parameters.</p>';
    return;
  }

  const traces = [];

  if (N > 0) {
    // --- Discrete bar chart: one bar per segment ---
    // Segments are ordered by w_cost: index 1 = most quality-preferring, N = most price-sensitive.
    const xs    = [];
    const ys    = [];
    const texts = [];

    for (let i = 0; i < N; i++) {
      const pos     = (i + 0.5) / N;   // midpoint of segment i in [0,1]
      const density = betaPDF(pos, alpha, beta);
      xs.push(i + 1);                  // 1-indexed segment number
      ys.push(density);
      texts.push(`Segment ${i + 1}<br>Density: ${density.toFixed(3)}`);
    }

    // Cap y-axis to avoid infinite spikes on U-shape distributions
    const sorted = [...ys].sort((a, b) => a - b);
    const cap    = (sorted[Math.floor(sorted.length * 0.98)] ?? sorted[sorted.length - 1]) * 1.2;

    traces.push({
      x:    xs,
      y:    ys.map((v) => Math.min(v, cap)),
      text: texts,
      type: "bar",
      marker: { color: "rgba(37,99,235,0.65)", line: { color: "#2563eb", width: 1 } },
      hovertemplate: "%{text}<extra></extra>",
      name: "Segment density",
    });

    const layout = {
      margin: { t: 36, r: 16, b: 50, l: 50 },
      title: {
        text: `Segment Density Distribution  ·  Beta(α=${alpha}, β=${beta})`,
        font: { size: 13 },
      },
      xaxis: {
        title: "Segment index  (1 = quality-preferring  →  N = price-sensitive)",
        tickmode: "linear",
        dtick: Math.max(1, Math.ceil(N / 10)),
        range: [0.5, N + 0.5],
      },
      yaxis: {
        title: "Density (relative group size)",
        range: [0, cap * 1.05],
      },
      bargap: 0.15,
      showlegend: false,
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
    };

    window.Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });

  } else {
    // --- Fallback: continuous curve when N is unknown ---
    const xs = [];
    const ys = [];
    const nPts = 300;
    for (let i = 1; i < nPts; i++) {
      const t = i / nPts;
      xs.push(t);
      ys.push(betaPDF(t, alpha, beta));
    }

    const sorted = [...ys].sort((a, b) => a - b);
    const cap    = (sorted[Math.floor(sorted.length * 0.98)] ?? sorted[sorted.length - 1]) * 1.1;

    traces.push({
      x:    xs,
      y:    ys.map((v) => Math.min(v, cap)),
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      line: { color: "#2563eb", width: 2 },
      fillcolor: "rgba(37,99,235,0.12)",
      name: `Beta(α=${alpha}, β=${beta})`,
    });

    window.Plotly.react(container, traces, {
      margin: { t: 36, r: 16, b: 50, l: 50 },
      title: { text: `Segment Density Distribution  ·  Beta(α=${alpha}, β=${beta})`, font: { size: 13 } },
      xaxis: { title: "Preference position  (0 = quality-preferring  →  1 = price-sensitive)", range: [0, 1] },
      yaxis: { title: "Density", range: [0, cap] },
      showlegend: false,
      plot_bgcolor: "#ffffff",
      paper_bgcolor: "#ffffff",
    }, { responsive: true, displayModeBar: false });
  }
}
