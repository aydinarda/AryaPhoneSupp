import { state, el } from "./state.js";
import {
  applyBetaFromServer,
  renderMatchingResult,
  renderRoundSummary,
  startRoundCountdown,
  clearRoundTimer,
} from "./round.js";
import { loadLeaderboard } from "./leaderboard.js";
import { setJoinedTeams } from "./sessionPlayers.js";

let _ws = null;
let _reconnectTimer = null;
let _reconnectDelay = 1000;
const _MAX_DELAY = 30000;
let _intentionalClose = false;
let _currentCode = null;

export function connectWS(sessionCode) {
  if (!sessionCode) return;
  _intentionalClose = false;
  _currentCode = sessionCode;
  _reconnectDelay = 1000;
  _doConnect(sessionCode);
}

export function disconnectWS() {
  _intentionalClose = true;
  _currentCode = null;
  if (_reconnectTimer) { clearTimeout(_reconnectTimer); _reconnectTimer = null; }
  if (_ws) { try { _ws.close(); } catch (_) {} _ws = null; }
  _reconnectDelay = 1000;
}

function _doConnect(sessionCode) {
  if (_ws) { try { _ws.close(); } catch (_) {} _ws = null; }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/sessions/${sessionCode}/ws`);
  _ws = ws;

  ws.addEventListener("open", () => {
    _reconnectDelay = 1000;
  });

  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    _handleMessage(msg);
  });

  ws.addEventListener("close", () => {
    _ws = null;
    if (!_intentionalClose && _currentCode) {
      _reconnectTimer = setTimeout(() => {
        _reconnectDelay = Math.min(_reconnectDelay * 2, _MAX_DELAY);
        _doConnect(_currentCode);
      }, _reconnectDelay);
    }
  });

  ws.addEventListener("error", () => { /* close event handles reconnect */ });
}

function _applyRoundFromMsg(msg) {
  const r = msg.round;
  const totalRounds = Number(msg.total_rounds);
  if (Number.isFinite(totalRounds)) state.totalRounds = totalRounds;
  if (r) {
    state.roundNo = r.round_no ?? null;
    state.roundEndsAt = r.ends_at || null;
    renderRoundSummary();
    startRoundCountdown();
  } else {
    state.roundNo = null;
    state.roundEndsAt = null;
    clearRoundTimer();
    renderRoundSummary();
  }
}

function _handleMessage(msg) {
  const { type } = msg;

  if (type === "sync") {
    setJoinedTeams(msg.players || []);
    applyBetaFromServer(msg);
    _applyRoundFromMsg(msg);
    if (msg.match && state.role === "admin") {
      renderMatchingResult(msg.match);
    }
    return;
  }

  if (type === "round_started") {
    state.roundNo = msg.round_no ?? null;
    state.roundEndsAt = msg.ends_at || null;
    if (Number.isFinite(Number(msg.total_rounds))) state.totalRounds = Number(msg.total_rounds);
    renderRoundSummary();
    startRoundCountdown();
    return;
  }

  if (type === "match_result") {
    if (state.role === "admin") {
      const rendered = renderMatchingResult(msg.matching);
      if (rendered && el.adminRoundHint) {
        el.adminRoundHint.textContent = `Matching completed for round ${msg.round_no}.`;
      }
    }
    loadLeaderboard();
    return;
  }

  if (type === "player_joined") {
    setJoinedTeams(msg.players || []);
    return;
  }

  if (type === "pong") return;
}
