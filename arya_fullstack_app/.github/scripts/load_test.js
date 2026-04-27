/**
 * Arya Phone Game — Load Test (50 users)
 * Target: https://aryaphonesupp.onrender.com
 *
 * Metrics:
 *   t_join_ms          — POST /api/sessions/{code}/join
 *   t_ws_open_ms       — WebSocket handshake time
 *   t_ws_first_msg_ms  — Time until first WS message received
 *   t_submit_ms        — POST /api/submit
 *   err_join / err_ws / err_submit — error rates
 */

import http from "k6/http";
import ws from "k6/ws";
import { check, sleep, group } from "k6";
import { Trend, Rate } from "k6/metrics";
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

const TARGET = __ENV.TARGET_URL || "https://aryaphonesupp.onrender.com";
const WS_URL = TARGET.replace(/^https/, "wss").replace(/^http/, "ws");
const HEADERS = { "Content-Type": "application/json" };

const tJoin   = new Trend("t_join_ms",        true);
const tSubmit = new Trend("t_submit_ms",      true);
const tWsOpen = new Trend("t_ws_open_ms",     true);
const tWsMsg  = new Trend("t_ws_first_msg_ms",true);
const errJoin   = new Rate("err_join");
const errWs     = new Rate("err_ws");
const errSubmit = new Rate("err_submit");

export const options = {
  scenarios: {
    classroom: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "15s", target: 50 },  // öğrenciler bağlanıyor
        { duration: "60s", target: 50 },  // hepsi aktif
        { duration: "10s", target: 0  },  // bitiyor
      ],
    },
  },
  thresholds: {
    http_req_duration:    ["p(95)<5000"],
    t_join_ms:            ["p(95)<4000"],
    t_ws_open_ms:         ["p(95)<5000"],
    t_ws_first_msg_ms:    ["p(95)<7000"],
    t_submit_ms:          ["p(95)<4000"],
    err_join:             ["rate<0.10"],
    err_ws:               ["rate<0.15"],
    err_submit:           ["rate<0.10"],
  },
};

// ---------------------------------------------------------------------------
// setup: session oluştur, round başlat, geçerli supplier picks al
// ---------------------------------------------------------------------------
export function setup() {
  // Geçerli tedarikçi ID'leri: her kategoriden bir tane
  const suppRes = http.get(`${TARGET}/api/suppliers`);
  let picks = [];
  try {
    const suppliers = JSON.parse(suppRes.body);
    const byCategory = {};
    for (const s of suppliers) {
      const cat = s.category || "__";
      if (!byCategory[cat]) byCategory[cat] = s.supplier_id;
    }
    picks = Object.values(byCategory);
    if (!picks.length && suppliers.length >= 2) {
      picks = [suppliers[0].supplier_id, suppliers[1].supplier_id];
    }
  } catch (_) {}

  // Session yarat
  const sessRes = http.post(
    `${TARGET}/api/sessions`,
    JSON.stringify({
      game_name: "GH_LoadTest_50",
      admin_name: "CI",
      number_of_rounds: 1,
      trial_rounds: 0,
    }),
    { headers: HEADERS }
  );

  let code = null;
  try { code = JSON.parse(sessRes.body).code; } catch (_) {}
  console.log(`Session: ${code} | Picks: ${JSON.stringify(picks)}`);

  if (code) {
    // Round 1 başlat
    http.post(
      `${TARGET}/api/sessions/${code}/rounds/start`,
      JSON.stringify({ duration_seconds: 300, market_capacity: 50 }),
      { headers: HEADERS }
    );
    sleep(1);
  }

  return { code, picks };
}

// ---------------------------------------------------------------------------
// default: her VU için join → websocket → submit
// ---------------------------------------------------------------------------
export default function (data) {
  const { code, picks } = data;
  if (!code) return;

  const team = `LT_VU${__VU}_I${__ITER}`;

  group("01_join", () => {
    const t0 = Date.now();
    const r = http.post(
      `${TARGET}/api/sessions/${code}/join`,
      JSON.stringify({ team_name: team }),
      { headers: HEADERS }
    );
    tJoin.add(Date.now() - t0);
    errJoin.add(!check(r, { "join 200": (x) => x.status === 200 }));
  });

  sleep(Math.random() * 0.5);

  group("02_websocket", () => {
    const t0 = Date.now();
    let gotMsg = false;

    const res = ws.connect(
      `${WS_URL}/api/sessions/${code}/ws`,
      {},
      (socket) => {
        tWsOpen.add(Date.now() - t0);

        socket.on("message", () => {
          if (!gotMsg) {
            tWsMsg.add(Date.now() - t0);
            gotMsg = true;
          }
        });

        socket.on("error", () => errWs.add(1));

        socket.setTimeout(() => {
          socket.send(JSON.stringify({ type: "ping" }));
        }, 5000);

        // 35 saniye bağlı kal
        socket.setTimeout(() => socket.close(), 35000);
      }
    );

    errWs.add(!check(res, { "ws 101": (r) => r && r.status === 101 }));
  });

  if (picks && picks.length) {
    group("03_submit", () => {
      const t0 = Date.now();
      const r = http.post(
        `${TARGET}/api/submit`,
        JSON.stringify({
          picks,
          team,
          player_name: team,
          session_code: code,
          round_no: 1,
          objective: "max_profit",
          price_per_user: 100,
          beta_alpha: 3.0,
          beta_beta: 3.0,
        }),
        { headers: HEADERS }
      );
      tSubmit.add(Date.now() - t0);
      errSubmit.add(!check(r, { "submit ok": (x) => x.status === 200 }));
      if (r.status !== 200) {
        console.warn(`VU${__VU} submit ${r.status}: ${r.body.substring(0, 300)}`);
      }
    });
  }

  sleep(1);
}

// ---------------------------------------------------------------------------
// teardown: tüm submitler bittikten sonra matching çalıştır
// ---------------------------------------------------------------------------
export function teardown(data) {
  const { code } = data;
  if (!code) return;

  sleep(2);
  console.log("Running match...");
  const r = http.post(
    `${TARGET}/api/sessions/${code}/match`,
    JSON.stringify({}),
    { headers: HEADERS }
  );
  console.log(`Match result: HTTP ${r.status}`);
}

// ---------------------------------------------------------------------------
// summary: HTML + JSON artifact
// ---------------------------------------------------------------------------
export function handleSummary(data) {
  return {
    "summary.html": htmlReport(data),
    "summary.json": JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: "  ", enableColors: true }),
  };
}
