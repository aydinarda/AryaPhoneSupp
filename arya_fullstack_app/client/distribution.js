import { state, el } from "./state.js";

/**
 * Evaluates Beta(alpha, beta) PDF at x using the log-gamma approach.
 * Avoids importing scipy — pure JS implementation.
 */
function logGamma(z) {
  // Lanczos approximation
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

function buildCurve(alpha, beta, N = 0, nPoints = 300) {
  const xs = [];
  const ys = [];
  // avoid exact 0 and 1 to prevent -Infinity
  for (let i = 1; i < nPoints; i++) {
    const t = i / nPoints;  // normalised position in (0, 1)
    xs.push(N > 1 ? t * N : t);  // scale to [0, N] when segments are known
    ys.push(betaPDF(t, alpha, beta));
  }
  return { xs, ys };
}

export function renderDistributionChart() {
  const container = el.distributionChart;
  if (!container) return;

  const alpha = state.betaAlpha;
  const beta  = state.betaBeta;
  const N = state.numSegments;

  if (!alpha || !beta || alpha <= 0 || beta <= 0) {
    container.innerHTML = '<p class="hint">Invalid distribution parameters.</p>';
    return;
  }

  const { xs, ys } = buildCurve(alpha, beta, N);

  // Cap the y-axis at the 98th percentile to handle U-shape spikes
  const sorted = [...ys].sort((a, b) => a - b);
  const cap = sorted[Math.floor(sorted.length * 0.98)] * 1.1;
  const ysClipped = ys.map((y) => Math.min(y, cap));

  const trace = {
    x: xs,
    y: ysClipped,
    type: "scatter",
    mode: "lines",
    fill: "tozeroy",
    line: { color: "#2563eb", width: 2 },
    fillcolor: "rgba(37,99,235,0.12)",
    name: `Beta(α=${alpha}, β=${beta})`,
  };

  let xaxisConfig;
  if (N > 1) {
    // Build tick marks at the centre of each group segment.
    // Group i (1-indexed) occupies [i-1, i] in [0, N] space → centre at i-0.5.
    const tickStep = Math.max(1, Math.ceil(N / 10));
    const tickvals = [];
    const ticktext = [];
    for (let i = 1; i <= N; i += tickStep) {
      tickvals.push(i - 0.5);
      ticktext.push(String(i));
    }
    xaxisConfig = {
      title: `Group Index (1 = low cost-sensitivity, ${N} = high)`,
      range: [0, N],
      tickvals,
      ticktext,
    };
  } else {
    xaxisConfig = {
      title: "Preference position (low cost-sensitivity → high)",
      range: [0, 1],
    };
  }

  const layout = {
    margin: { t: 32, r: 16, b: 40, l: 44 },
    title: {
      text: `Segment Density Distribution  ·  Beta(α=${alpha}, β=${beta})`,
      font: { size: 13 },
    },
    xaxis: xaxisConfig,
    yaxis: { title: "Density", range: [0, cap] },
    showlegend: false,
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
  };

  if (window.Plotly) {
    window.Plotly.react(container, [trace], layout, { responsive: true, displayModeBar: false });
  } else {
    container.innerHTML = '<p class="hint">Plotly not loaded.</p>';
  }
}
