/**
 * Arya Phone Game — Extreme / realistic classroom load test
 * ---------------------------------------------------------
 * Models MULTIPLE concurrent sessions, each with many players, playing
 * several timed rounds. Reproduces the real client traffic shape:
 *
 *   - one persistent WebSocket per player (push channel, periodic ping)
 *   - per player, per round: 1-2 evaluate calls, 1 submit, 1 leaderboard view
 *   - an orchestrator that starts each round and runs matching on a fixed
 *     ROUND_SECONDS cadence (mirrors the admin)
 *
 * The old browser client also polled GET /rounds/current every 2s. That
 * polling was removed (updates now arrive over the WebSocket). To measure the
 * capacity saved, set SIMULATE_POLLING=true to re-introduce that polling and
 * compare request rate + latencies against SIMULATE_POLLING=false.
 *
 * Env (all optional):
 *   TARGET_URL           default https://aryaphonesupp.onrender.com
 *   SESSIONS             default 2
 *   PLAYERS_PER_SESSION  default 25
 *   ROUNDS               default 3
 *   ROUND_SECONDS        default 35
 *   SIMULATE_POLLING     default false   (true = emulate the old 2s polling)
 *   POLL_INTERVAL_S      default 2
 *
 * Example:
 *   k6 run --env SESSIONS=2 --env PLAYERS_PER_SESSION=25 --env ROUNDS=3 \
 *          --env ROUND_SECONDS=35 .github/scripts/load_test_classroom.js
 *   # A/B the polling cost:
 *   k6 run --env SIMULATE_POLLING=true  ... load_test_classroom.js
 */

import http from "k6/http";
import ws from "k6/ws";
import { check, sleep, group } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

const TARGET = __ENV.TARGET_URL || "https://aryaphonesupp.onrender.com";
const WS_URL = TARGET.replace(/^https/, "wss").replace(/^http/, "ws");
const HEADERS = { "Content-Type": "application/json" };

const SESSIONS            = parseInt(__ENV.SESSIONS || "2", 10);
const PLAYERS_PER_SESSION = parseInt(__ENV.PLAYERS_PER_SESSION || "25", 10);
const ROUNDS              = parseInt(__ENV.ROUNDS || "3", 10);
const ROUND_SECONDS       = parseInt(__ENV.ROUND_SECONDS || "35", 10);
const SIMULATE_POLLING    = (__ENV.SIMULATE_POLLING || "false").toLowerCase() === "true";
const POLL_INTERVAL_S     = parseInt(__ENV.POLL_INTERVAL_S || "2", 10);

const TOTAL_PLAYERS = SESSIONS * PLAYERS_PER_SESSION;
const GAME_SECONDS  = ROUNDS * ROUND_SECONDS + 20; // + matching/lobby slack

const tJoin        = new Trend("t_join_ms", true);
const tSubmit      = new Trend("t_submit_ms", true);
const tEval        = new Trend("t_eval_ms", true);
const tLeaderboard = new Trend("t_leaderboard_ms", true);
const tMatch       = new Trend("t_match_ms", true);
const tPoll        = new Trend("t_poll_ms", true);
const tWsOpen      = new Trend("t_ws_open_ms", true);
const tWsMsg       = new Trend("t_ws_first_msg_ms", true);

const errJoin   = new Rate("err_join");
const errWs     = new Rate("err_ws");
const errSubmit = new Rate("err_submit");
const errMatch  = new Rate("err_match");
const pollReqs  = new Counter("poll_requests_total");

export const options = {
  scenarios: {
    // One long-lived WebSocket per player for the whole game.
    ws_holders: {
      executor: "per-vu-iterations",
      exec: "wsHolder",
      vus: TOTAL_PLAYERS,
      iterations: 1,
      maxDuration: `${GAME_SECONDS + 30}s`,
    },
    // HTTP lifecycle per player (join + per-round eval/submit/leaderboard).
    player_actions: {
      executor: "per-vu-iterations",
      exec: "playerActions",
      vus: TOTAL_PLAYERS,
      iterations: 1,
      maxDuration: `${GAME_SECONDS + 60}s`,
    },
    // Drives every session's round lifecycle (start round / run match).
    orchestrator: {
      executor: "per-vu-iterations",
      exec: "orchestrator",
      vus: 1,
      iterations: 1,
      maxDuration: `${GAME_SECONDS + 60}s`,
    },
  },
  thresholds: {
    http_req_failed:   ["rate<0.10"],
    t_join_ms:         ["p(95)<4000"],
    t_submit_ms:       ["p(95)<5000"],
    t_ws_open_ms:      ["p(95)<5000"],
    t_ws_first_msg_ms: ["p(95)<7000"],
    t_match_ms:        ["p(95)<15000"],
    err_join:          ["rate<0.10"],
    err_ws:            ["rate<0.15"],
    err_submit:        ["rate<0.10"],
    err_match:         ["rate<0.20"],
  },
};

function sessionFor(data) {
  if (!data.sessions || !data.sessions.length) return null;
  return data.sessions[(__VU - 1) % data.sessions.length];
}

// --------------------------------------------------------------------------- //
// setup: build picks, create N sessions, start round 1 on each
// --------------------------------------------------------------------------- //
export function setup() {
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

  const sessions = [];
  for (let i = 0; i < SESSIONS; i++) {
    const res = http.post(
      `${TARGET}/api/sessions`,
      JSON.stringify({
        game_name: `GH_Classroom_${i + 1}`,
        admin_name: "CI",
        number_of_rounds: ROUNDS,
        trial_rounds: 0,
      }),
      { headers: HEADERS }
    );
    let code = null;
    try { code = JSON.parse(res.body).code; } catch (_) {}
    if (code) {
      http.post(
        `${TARGET}/api/sessions/${code}/rounds/start`,
        JSON.stringify({ duration_seconds: ROUND_SECONDS, market_capacity: PLAYERS_PER_SESSION }),
        { headers: HEADERS }
      );
      sessions.push({ code, picks });
    }
  }
  console.log(`Sessions: ${JSON.stringify(sessions.map((s) => s.code))} | picks: ${JSON.stringify(picks)} | polling=${SIMULATE_POLLING}`);
  sleep(1);
  return { sessions };
}

// --------------------------------------------------------------------------- //
// wsHolder: hold one WebSocket open for the whole game (push channel)
// --------------------------------------------------------------------------- //
export function wsHolder(data) {
  const s = sessionFor(data);
  if (!s) return;
  sleep(Math.random() * 3); // spread the connection storm slightly

  const t0 = Date.now();
  let gotMsg = false;
  const res = ws.connect(`${WS_URL}/api/sessions/${s.code}/ws`, {}, (socket) => {
    socket.on("open", () => tWsOpen.add(Date.now() - t0));
    socket.on("message", () => {
      if (!gotMsg) { tWsMsg.add(Date.now() - t0); gotMsg = true; }
    });
    socket.on("error", () => errWs.add(1));
    // keep-alive ping (also proves the WS survives the whole game)
    socket.setInterval(() => socket.send(JSON.stringify({ type: "ping" })), 25000);
    socket.setTimeout(() => socket.close(), GAME_SECONDS * 1000);
  });
  errWs.add(!check(res, { "ws 101": (r) => r && r.status === 101 }));
}

// --------------------------------------------------------------------------- //
// playerActions: join, then per round: evaluate -> submit -> (window) -> leaderboard
// --------------------------------------------------------------------------- //
export function playerActions(data) {
  const s = sessionFor(data);
  if (!s) return;
  const team = `S${(__VU - 1) % SESSIONS}_VU${__VU}`;
  sleep(Math.random() * 3);

  group("join", () => {
    const t = Date.now();
    const r = http.post(`${TARGET}/api/sessions/${s.code}/join`,
      JSON.stringify({ team_name: team }), { headers: HEADERS });
    tJoin.add(Date.now() - t);
    errJoin.add(!check(r, { "join 200": (x) => x.status === 200 }));
  });

  // Client boot reads (config + suppliers) happen once on load.
  http.get(`${TARGET}/api/config`);
  http.get(`${TARGET}/api/suppliers`);

  for (let round = 1; round <= ROUNDS; round++) {
    if (!s.picks || !s.picks.length) break;

    // 1-2 evaluate previews before locking in.
    const evalCount = 1 + Math.floor(Math.random() * 2);
    for (let e = 0; e < evalCount; e++) {
      const t = Date.now();
      http.post(`${TARGET}/api/manual-eval`, JSON.stringify({
        objective: "max_profit", picks: s.picks, price_per_user: 90 + Math.floor(Math.random() * 40),
        beta_alpha: 3.0, beta_beta: 3.0,
      }), { headers: HEADERS });
      tEval.add(Date.now() - t);
      sleep(0.5 + Math.random());
    }

    group("submit", () => {
      const t = Date.now();
      const r = http.post(`${TARGET}/api/submit`, JSON.stringify({
        picks: s.picks, team, player_name: team, session_code: s.code, round_no: round,
        objective: "max_profit", price_per_user: 100, beta_alpha: 3.0, beta_beta: 3.0,
      }), { headers: HEADERS });
      tSubmit.add(Date.now() - t);
      errSubmit.add(!check(r, { "submit 200": (x) => x.status === 200 }));
    });

    // Wait out the round. This is where the OLD client polled /rounds/current.
    roundWindow(s.code);

    // View standings once the round's match has run.
    const t = Date.now();
    http.get(`${TARGET}/api/sessions/${s.code}/leaderboard`);
    tLeaderboard.add(Date.now() - t);
  }
}

function roundWindow(code) {
  let elapsed = 0;
  while (elapsed < ROUND_SECONDS) {
    if (SIMULATE_POLLING) {
      const t = Date.now();
      http.get(`${TARGET}/api/sessions/${code}/rounds/current`);
      tPoll.add(Date.now() - t);
      pollReqs.add(1);
    }
    sleep(POLL_INTERVAL_S);
    elapsed += POLL_INTERVAL_S;
  }
}

// --------------------------------------------------------------------------- //
// orchestrator: run matching + advance rounds for every session on a cadence
// --------------------------------------------------------------------------- //
export function orchestrator(data) {
  const sessions = data.sessions || [];
  if (!sessions.length) return;

  for (let round = 1; round <= ROUNDS; round++) {
    sleep(ROUND_SECONDS); // let players submit during the round window
    for (const s of sessions) {
      const t = Date.now();
      const r = http.post(`${TARGET}/api/sessions/${s.code}/match`, "{}", { headers: HEADERS });
      tMatch.add(Date.now() - t);
      errMatch.add(!check(r, { "match 200": (x) => x.status === 200 }));
    }
    if (round < ROUNDS) {
      for (const s of sessions) {
        http.post(`${TARGET}/api/sessions/${s.code}/rounds/start`,
          JSON.stringify({ duration_seconds: ROUND_SECONDS, market_capacity: PLAYERS_PER_SESSION }),
          { headers: HEADERS });
      }
    }
  }
}

export function handleSummary(data) {
  return {
    "summary.html": htmlReport(data),
    "summary.json": JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: "  ", enableColors: true }),
  };
}
