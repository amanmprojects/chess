/**
 * UI layer — board rendering, input handling, and coordination with the AI.
 *
 * The engine is the single source of truth: every interaction resolves to a
 * move that `Chess.moves()` produced, so the interface cannot reach a position
 * the rules disallow.
 */

import {
  Chess, WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
  START_FEN, algebraic, squareFromAlgebraic, colorOf, typeOf,
  moveTo, movePromo, moveFlags,
  FLAG_CAPTURE, FLAG_PROMO, FLAG_EP, FLAG_KCASTLE, FLAG_QCASTLE,
} from './engine.js';
import { pieceSvg, pieceName } from './pieces.js';

const FILES = 'abcdefgh';
/** Values used only for the material readout beside each player. */
const DISPLAY_VALUE = { [PAWN]: 1, [KNIGHT]: 3, [BISHOP]: 3, [ROOK]: 5, [QUEEN]: 9, [KING]: 0 };
/** A full army, for working out which pieces have been captured. */
const FULL_ARMY = { [PAWN]: 8, [KNIGHT]: 2, [BISHOP]: 2, [ROOK]: 2, [QUEEN]: 1, [KING]: 1 };
const PROMOTION_CHOICES = [QUEEN, ROOK, BISHOP, KNIGHT];
const CAPTURED_ORDER = [QUEEN, ROOK, BISHOP, KNIGHT, PAWN];

const $ = (id) => document.getElementById(id);
const squareIndex = (name) => squareFromAlgebraic(name);

const el = {
  board: $('board'),
  statusTurn: $('status-turn'),
  statusDetail: $('status-detail'),
  moves: $('moves'),
  thinking: $('thinking'),
  evalFill: $('eval-fill'),
  evalText: $('eval-text'),
  promotion: $('promotion'),
  promotionChoices: $('promotion-choices'),
  result: $('result'),
  resultTitle: $('result-title'),
  resultDetail: $('result-detail'),
  resultNewGame: $('result-newgame'),
  stripTop: $('strip-top'),
  stripBottom: $('strip-bottom'),
  fen: $('fen'),
  fenMsg: $('fen-msg'),
  opponent: $('opponent'),
  level: $('level'),
  side: $('side'),
  levelField: $('level-field'),
  sideField: $('side-field'),
  animate: $('animate'),
  animateField: $('animate-field'),
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  game: new Chess(),
  /** FEN after each played move; index 0 is the starting position. */
  timeline: [START_FEN],
  /** Move records, parallel to timeline[1..]. */
  played: [],
  orientation: WHITE,
  opponent: 'ai',
  level: 'intermediate',
  /** Which colour the human plays when the opponent is the computer. */
  humanSide: WHITE,
  /** Slide the computer's pieces to their square instead of snapping them. */
  animateOpponent: true,
  selected: -1,
  /** Legal moves from the selected square. */
  candidates: [],
  lastMove: null,
  thinking: false,
  /** -1 = live game; otherwise an index into `timeline` being reviewed. */
  reviewIndex: -1,
  /** Bumped whenever the position changes, to discard stale worker replies. */
  generation: 0,
  pendingRequest: null,
  evaluation: null,
  promotionPending: null,
};

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

let worker = null;

function ensureWorker() {
  if (worker) return worker;
  try {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = onWorkerMessage;
    worker.onerror = () => {
      // If the worker cannot start (some file:// setups), fall back to running
      // the search on the main thread rather than leaving the game unplayable.
      worker = null;
      state.workerBroken = true;
    };
  } catch {
    worker = null;
    state.workerBroken = true;
  }
  return worker;
}

function onWorkerMessage(event) {
  const data = event.data;
  const request = state.pendingRequest;
  // Ignore anything from a search that a newer position has superseded.
  if (!request || data.id !== request.id) return;

  if (data.type === 'progress') {
    state.evaluation = data.score;
    renderEval();
    return;
  }

  if (data.type === 'error') {
    state.pendingRequest = null;
    setThinking(false);
    showFenMessage(`Engine error: ${data.message}`, 'error');
    return;
  }

  if (data.type !== 'bestmove') return;

  state.pendingRequest = null;
  setThinking(false);
  state.evaluation = data.score;

  if (!data.uci) { render(); return; }

  if (request.purpose === 'hint') {
    showHint(data.uci);
    return;
  }

  playMove(data.uci);
}

/** Ask the engine for a move. `purpose` is 'play' or 'hint'. */
function requestSearch(purpose, level) {
  const id = ++state.generation;
  state.pendingRequest = { id, purpose };
  setThinking(true);

  const payload = {
    type: 'search',
    id,
    fen: state.game.fen(),
    level,
    history: state.game.positions.slice(),
  };

  const w = ensureWorker();
  if (w) {
    w.postMessage(payload);
    return;
  }

  // Main-thread fallback. Deferred so the "thinking" indicator paints first.
  setTimeout(async () => {
    try {
      const [{ chooseMove }, { moveToUci }] = await Promise.all([
        import('./ai.js'),
        import('./engine.js'),
      ]);
      const game = new Chess(payload.fen);
      game.positions = payload.history;
      const result = chooseMove(game, level);
      onWorkerMessage({
        data: {
          type: 'bestmove', id,
          uci: result.move ? moveToUci(result.move) : null,
          depth: result.depth, nodes: result.nodes,
          score: { type: 'cp', value: result.score },
        },
      });
    } catch (error) {
      onWorkerMessage({ data: { type: 'error', id, message: String(error?.message ?? error) } });
    }
  }, 20);
}

/** True when the person at the keyboard owns the current turn. */
function humanToMove() {
  return state.opponent === 'human' || state.game.turn === state.humanSide;
}

/** True when input should be accepted at all. */
function inputEnabled() {
  return !state.thinking && !state.promotionPending && state.reviewIndex === -1
    && !state.game.status().over && humanToMove();
}

// ---------------------------------------------------------------------------
// Board construction
// ---------------------------------------------------------------------------

/** Square index -> its button element. */
const squareEls = new Map();

/** Squares in the order they should appear for the current orientation. */
function displayOrder() {
  const order = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const rank = state.orientation === WHITE ? 7 - row : row;
      const file = state.orientation === WHITE ? col : 7 - col;
      order.push(rank * 16 + file);
    }
  }
  return order;
}

function buildBoard() {
  el.board.replaceChildren();
  squareEls.clear();

  for (const sq of displayOrder()) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'square';
    button.dataset.square = String(sq);
    button.tabIndex = -1;

    const slot = document.createElement('span');
    slot.className = 'piece-slot';
    button.append(slot);

    el.board.append(button);
    squareEls.set(sq, button);
  }

  // One square is tabbable, and the arrow keys move focus from there.
  const first = squareEls.get(displayOrder()[0]);
  if (first) first.tabIndex = 0;
}

/** Reorder the existing buttons after a flip, without rebuilding them. */
function reorderBoard() {
  for (const sq of displayOrder()) {
    el.board.append(squareEls.get(sq));
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  const reviewing = state.reviewIndex !== -1;
  const position = reviewing ? new Chess(state.timeline[state.reviewIndex]) : state.game;
  const highlight = reviewing
    ? (state.reviewIndex > 0 ? state.played[state.reviewIndex - 1] : null)
    : state.lastMove;

  renderBoard(position, highlight, reviewing);
  renderStatus(position, reviewing);
  renderMoveList();
  renderStrips(position);
  renderEval();
  renderResult();

  el.fen.value = position.fen();
}

function renderBoard(position, highlight, reviewing) {
  const checkedSquare = position.inCheck() ? position.kings[position.turn] : -1;
  const fromSq = highlight ? squareIndex(highlight.from) : -1;
  const toSq = highlight ? squareIndex(highlight.to) : -1;

  const targets = new Map();
  if (!reviewing) {
    for (const move of state.candidates) {
      targets.set(moveTo(move), (moveFlags(move) & FLAG_CAPTURE) !== 0);
    }
  }

  for (const [sq, button] of squareEls) {
    const rank = sq >> 4;
    const file = sq & 7;
    const isLight = ((rank + file) & 1) === 1;

    const classes = ['square', isLight ? 'light' : 'dark'];
    if (sq === fromSq || sq === toSq) classes.push('last-move');
    if (sq === state.selected && !reviewing) classes.push('from');
    if (sq === checkedSquare) classes.push('checked');
    if (targets.has(sq)) {
      classes.push('hint', targets.get(sq) ? 'capture' : 'quiet');
    } else if (state.hintMove && !reviewing
               && (sq === squareIndex(state.hintMove.from) || sq === squareIndex(state.hintMove.to))) {
      classes.push('selectable');
    }
    button.className = classes.join(' ');

    const piece = position.board[sq];
    const slot = button.firstElementChild;
    const key = piece ? `${piece}` : '';
    if (slot.dataset.piece !== key) {
      slot.dataset.piece = key;
      slot.innerHTML = piece ? pieceSvg(colorOf(piece), typeOf(piece)) : '';
    }

    const name = algebraic(sq);
    const occupant = piece ? pieceName(colorOf(piece), typeOf(piece)) : 'empty';
    button.setAttribute('aria-label', `${name}, ${occupant}`);
    button.disabled = false;
  }

  // Coordinates along the outer edges only, so the board stays uncluttered.
  const order = displayOrder();
  for (let i = 0; i < order.length; i++) {
    const button = squareEls.get(order[i]);
    button.querySelectorAll('.coord').forEach((n) => n.remove());
    const lastRow = i >= 56;
    const firstCol = i % 8 === 0;
    if (lastRow) {
      const f = document.createElement('span');
      f.className = 'coord file';
      f.textContent = FILES[order[i] & 7];
      button.append(f);
    }
    if (firstCol) {
      const r = document.createElement('span');
      r.className = 'coord rank';
      r.textContent = String((order[i] >> 4) + 1);
      button.append(r);
    }
  }
}

function renderStatus(position, reviewing) {
  if (reviewing) {
    el.statusTurn.textContent = `Reviewing move ${state.reviewIndex} of ${state.played.length}`;
    el.statusDetail.textContent = 'Make a move or press Live to resume play.';
    return;
  }

  const status = position.status();
  const mover = position.turn === WHITE ? 'White' : 'Black';

  if (status.over) {
    el.statusTurn.textContent = status.result === '1/2-1/2'
      ? 'Draw'
      : `${status.result === '1-0' ? 'White' : 'Black'} wins`;
    el.statusDetail.textContent = `by ${status.reason}`;
    return;
  }

  el.statusTurn.textContent = `${mover} to move`;
  const parts = [];
  if (position.inCheck()) parts.push('Check!');
  if (state.thinking) parts.push('Computer is thinking');
  el.statusDetail.textContent = parts.join(' · ');
}

function renderMoveList() {
  el.moves.replaceChildren();

  for (let i = 0; i < state.played.length; i += 2) {
    const number = document.createElement('li');
    number.className = 'no';
    number.textContent = `${i / 2 + 1}.`;
    el.moves.append(number);

    for (const offset of [0, 1]) {
      const record = state.played[i + offset];
      const cell = document.createElement('li');
      if (!record) {
        cell.textContent = '';
      } else {
        const ply = i + offset;
        cell.className = 'move';
        cell.textContent = record.san;
        cell.dataset.ply = String(ply);
        const isCurrent = state.reviewIndex === -1
          ? ply === state.played.length - 1
          : ply === state.reviewIndex - 1;
        if (isCurrent) cell.classList.add('current');
      }
      el.moves.append(cell);
    }
  }

  if (state.reviewIndex === -1) el.moves.scrollTop = el.moves.scrollHeight;
}

function renderEval() {
  const score = state.evaluation;
  if (!score || state.opponent === 'human') {
    el.evalText.textContent = '—';
    el.evalFill.style.width = '50%';
    return;
  }

  if (score.type === 'mate') {
    const sign = score.winning === WHITE ? '' : '-';
    el.evalText.textContent = `M${sign}${score.moves}`;
    el.evalFill.style.width = score.winning === WHITE ? '100%' : '0%';
    el.evalFill.style.background = score.winning === WHITE ? '#f0f0f0' : '#2c2c2c';
    return;
  }

  // The score arrives from the mover's point of view; show it from White's.
  const white = state.game.turn === WHITE ? score.value : -score.value;
  const pawns = white / 100;
  el.evalText.textContent = `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}`;
  // Squash to a percentage; ±5 pawns is effectively decisive.
  const pct = 100 / (1 + Math.exp(-white / 300));
  el.evalFill.style.width = `${pct.toFixed(1)}%`;
  el.evalFill.style.background = '#e8ecf3';
}

function renderStrips(position) {
  const bottomColor = state.orientation;
  const topColor = bottomColor ^ 1;
  const counts = countMissing(position);

  fillStrip(el.stripTop, topColor, counts, position);
  fillStrip(el.stripBottom, bottomColor, counts, position);
}

/**
 * How many of each piece each side has lost, derived from the board rather
 * than from the move list so that a loaded FEN reports sensibly too.
 */
function countMissing(position) {
  const present = { [WHITE]: {}, [BLACK]: {} };
  for (let sq = 0; sq < 128; sq++) {
    if (sq & 0x88) continue;
    const p = position.board[sq];
    if (!p) continue;
    const c = colorOf(p);
    const t = typeOf(p);
    present[c][t] = (present[c][t] ?? 0) + 1;
  }

  const missing = { [WHITE]: {}, [BLACK]: {} };
  let score = { [WHITE]: 0, [BLACK]: 0 };
  for (const color of [WHITE, BLACK]) {
    for (const type of CAPTURED_ORDER) {
      const lost = Math.max(0, FULL_ARMY[type] - (present[color][type] ?? 0));
      missing[color][type] = lost;
      score[color] += (present[color][type] ?? 0) * DISPLAY_VALUE[type];
    }
  }
  return { missing, score };
}

function fillStrip(strip, color, counts, position) {
  const nameEl = strip.querySelector('.player-name');
  const capturedEl = strip.querySelector('.captured');
  const materialEl = strip.querySelector('.material');

  const isHuman = state.opponent === 'human' || color === state.humanSide;
  const side = color === WHITE ? 'White' : 'Black';
  const who = state.opponent === 'human'
    ? side
    : `${side} — ${isHuman ? 'You' : levelLabel()}`;
  nameEl.textContent = who;

  strip.classList.toggle('active', position.turn === color && !position.status().over);

  // A player's strip shows the enemy pieces they have captured.
  const taken = counts.missing[color ^ 1];
  const svgs = [];
  for (const type of CAPTURED_ORDER) {
    for (let i = 0; i < taken[type]; i++) {
      svgs.push(pieceSvg(color ^ 1, type, { className: 'captured-piece' }));
    }
  }
  capturedEl.innerHTML = svgs.join('');

  const diff = counts.score[color] - counts.score[color ^ 1];
  materialEl.textContent = diff > 0 ? `+${diff}` : '';
}

function levelLabel() {
  const option = el.level.options[el.level.selectedIndex];
  return option ? option.textContent : state.level;
}

function renderResult() {
  const status = state.reviewIndex === -1 ? state.game.status() : { over: false };
  if (!status.over || state.resultDismissed) {
    el.result.hidden = true;
    return;
  }

  el.result.hidden = false;
  if (status.result === '1/2-1/2') {
    el.resultTitle.textContent = 'Draw';
  } else {
    const winner = status.result === '1-0' ? WHITE : BLACK;
    const youWon = state.opponent === 'ai' && winner === state.humanSide;
    el.resultTitle.textContent = state.opponent === 'ai'
      ? (youWon ? 'You win' : 'Computer wins')
      : `${winner === WHITE ? 'White' : 'Black'} wins`;
  }
  el.resultDetail.textContent = `${status.result} by ${status.reason}`;
}

function setThinking(on) {
  state.thinking = on;
  el.thinking.hidden = !on;
  el.board.classList.toggle('busy', on);
  renderStatus(state.game, state.reviewIndex !== -1);
}

function showFenMessage(text, kind = '') {
  el.fenMsg.textContent = text;
  el.fenMsg.className = `fen-msg ${kind}`;
  if (!text) return;
  clearTimeout(showFenMessage.timer);
  showFenMessage.timer = setTimeout(() => {
    el.fenMsg.textContent = '';
    el.fenMsg.className = 'fen-msg';
  }, 4000);
}

// ---------------------------------------------------------------------------
// Move animation
// ---------------------------------------------------------------------------

/**
 * Moves are animated after the board has already been re-rendered in its new
 * state: the piece is placed on its destination square, offset back to where
 * it came from, and then slid to zero. Nothing here can desynchronise the
 * board from the engine, because the animation only ever touches `transform`.
 */

const ANIM_MS = 190;
const ANIM_EASING = 'cubic-bezier(0.22, 0.68, 0.36, 1)';

const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');

/** Animations currently in flight, so a new move can cut them short. */
let running = [];
/** Set by the drag handler: the piece already travelled under the pointer. */
let skipNextAnimation = false;

function animationsEnabled() {
  return !reduceMotion?.matches && typeof Element.prototype.animate === 'function';
}

/** Stop anything in flight and drop it at its final position. */
function finishAnimations() {
  for (const animation of running) animation.finish();
  running = [];
  // Done synchronously: the callbacks that would otherwise tidy these run on a
  // microtask, which is too late for a move that is about to start animating.
  for (const square of el.board.querySelectorAll('[data-slide]')) {
    square.style.zIndex = '';
    delete square.dataset.slide;
  }
  el.board.querySelectorAll('.capture-ghost').forEach((n) => n.remove());
}

/** Distinguishes successive lifts of the same square. See `slidePiece`. */
let slideToken = 0;

/** Slide the piece sitting on `toSq` in from `fromSq`. */
function slidePiece(fromSq, toSq) {
  const fromEl = squareEls.get(fromSq);
  const toEl = squareEls.get(toSq);
  if (!fromEl || !toEl) return;

  const a = fromEl.getBoundingClientRect();
  const b = toEl.getBoundingClientRect();
  const dx = a.left - b.left;
  const dy = a.top - b.top;
  if (dx === 0 && dy === 0) return;

  // Squares are grid siblings, so the travelling piece has to be lifted at the
  // square level to pass over the ones it crosses.
  const token = String(++slideToken);
  toEl.style.zIndex = '20';
  toEl.dataset.slide = token;

  const animation = toEl.firstElementChild.animate(
    [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'translate(0px, 0px)' }],
    { duration: ANIM_MS, easing: ANIM_EASING },
  );
  running.push(animation);

  // Interrupting resolves `finished` on a microtask, which can land *after* a
  // following move has already re-lifted this same square. The token means a
  // stale release cannot drop a piece that is currently mid-slide.
  const release = () => {
    if (toEl.dataset.slide !== token) return;
    toEl.style.zIndex = '';
    delete toEl.dataset.slide;
  };
  animation.finished.then(release, release);
}

/** Fade out the piece that was taken, so captures do not just blink away. */
function fadeCapture(sq, capturedPiece) {
  const square = squareEls.get(sq);
  if (!square || !capturedPiece) return;

  const ghost = document.createElement('span');
  ghost.className = 'capture-ghost';
  ghost.innerHTML = pieceSvg(colorOf(capturedPiece), typeOf(capturedPiece));
  square.append(ghost);

  const animation = ghost.animate(
    [{ opacity: 0.85, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(0.72)' }],
    { duration: ANIM_MS + 60, easing: 'ease-out' },
  );
  running.push(animation);
  // A mid-flight interruption (`finish`) resolves `finished` too, so removal is
  // handled from one place rather than racing with the finish loop above.
  animation.finished.then(
    () => ghost.remove(),
    () => ghost.remove(),
  );
}

/** Briefly ring the destination square, so a move is easy to spot. */
function flashDestination(sq) {
  const square = squareEls.get(sq);
  if (!square) return;
  const animation = square.animate(
    [
      { boxShadow: 'inset 0 0 0 3px rgba(79, 140, 255, 0.9)' },
      { boxShadow: 'inset 0 0 0 3px rgba(79, 140, 255, 0)' },
    ],
    { duration: 620, easing: 'ease-out' },
  );
  running.push(animation);
}

/**
 * Animate a move that has just been applied.
 *
 * @param {object} record the record returned by `Chess.move`
 * @param {Array}  boardBefore a copy of the board from before the move
 * @param {object} [opts]
 * @param {boolean} [opts.flash] ring the destination when the move arrives
 */
function animateMove(record, boardBefore, { flash = false } = {}) {
  finishAnimations();
  if (!animationsEnabled()) return;

  const from = squareIndex(record.from);
  const to = squareIndex(record.to);
  const flags = moveFlags(record.move);

  // En passant takes a pawn that is not on the destination square.
  const capturedSq = (flags & FLAG_EP)
    ? (record.color === WHITE ? to - 16 : to + 16)
    : to;
  if (flags & FLAG_CAPTURE) fadeCapture(capturedSq, boardBefore[capturedSq]);

  slidePiece(from, to);

  // Castling moves two pieces; the rook has to travel as well or the king
  // appears to jump over a rook that teleported.
  if (flags & (FLAG_KCASTLE | FLAG_QCASTLE)) {
    const rookFrom = (flags & FLAG_KCASTLE) ? to + 1 : to - 2;
    const rookTo = (flags & FLAG_KCASTLE) ? to - 1 : to + 1;
    slidePiece(rookFrom, rookTo);
  }

  if (flash) flashDestination(to);
}

// ---------------------------------------------------------------------------
// Playing moves
// ---------------------------------------------------------------------------

/**
 * Apply a move and advance the game. Accepts anything `Chess.move` accepts —
 * in practice a UCI string from the engine or a packed move from the board.
 */
function playMove(input) {
  // Snapshot before the move so the animation can still draw what was taken.
  const boardBefore = state.game.board.slice();
  // A move that lands mid-animation cuts the previous one short rather than
  // overlapping with it.
  finishAnimations();

  const record = state.game.move(input);
  if (!record) {
    showFenMessage('That move is not legal in this position.', 'error');
    return null;
  }

  const byOpponent = state.opponent === 'ai' && record.color !== state.humanSide;
  // The toggle covers the computer's moves. A human's click-to-move still
  // slides: that is direct feedback for an action they just took, and it is
  // never a surprise the way a piece moving on its own is.
  const animate = !skipNextAnimation && (!byOpponent || state.animateOpponent);
  skipNextAnimation = false;

  state.played.push(record);
  state.timeline.push(state.game.fen());
  state.lastMove = record;
  state.selected = -1;
  state.candidates = [];
  state.hintMove = null;
  state.reviewIndex = -1;
  state.resultDismissed = false;

  render();
  if (animate) animateMove(record, boardBefore, { flash: byOpponent });
  maybeStartEngineTurn();
  return record;
}

/** Hand the turn to the computer if it is now its move. */
function maybeStartEngineTurn() {
  if (state.opponent !== 'ai') return;
  if (state.game.status().over) return;
  if (state.game.turn === state.humanSide) return;
  requestSearch('play', state.level);
}

function showHint(uci) {
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  state.hintMove = { from, to };
  render();
  showFenMessage(`Try ${from}–${to}.`, 'ok');
}

// ---------------------------------------------------------------------------
// Selection and input
// ---------------------------------------------------------------------------

function selectSquare(sq) {
  const piece = state.game.board[sq];
  if (!piece || colorOf(piece) !== state.game.turn) {
    clearSelection();
    return false;
  }
  state.selected = sq;
  state.candidates = state.game.generateMoves({ square: sq });
  state.hintMove = null;
  render();
  return state.candidates.length > 0;
}

function clearSelection() {
  if (state.selected === -1 && state.candidates.length === 0) return;
  state.selected = -1;
  state.candidates = [];
  render();
}

/** Try to move from the selected square to `sq`. Returns true if handled. */
function tryMoveTo(sq) {
  const matches = state.candidates.filter((m) => moveTo(m) === sq);
  if (matches.length === 0) return false;

  // Several matches means a promotion, which differ only in the new piece.
  if (matches.length > 1 || (moveFlags(matches[0]) & FLAG_PROMO)) {
    openPromotion(matches);
    return true;
  }

  playMove(matches[0]);
  return true;
}

/** The shared click / tap / drop handler. */
function handleSquare(sq) {
  if (!inputEnabled()) return;

  if (state.selected !== -1) {
    if (sq === state.selected) { clearSelection(); return; }
    if (tryMoveTo(sq)) return;
  }
  selectSquare(sq);
}

// ---------------------------------------------------------------------------
// Promotion
// ---------------------------------------------------------------------------

function openPromotion(moves) {
  state.promotionPending = moves;
  el.promotionChoices.replaceChildren();

  const color = state.game.turn;
  for (const type of PROMOTION_CHOICES) {
    const move = moves.find((m) => movePromo(m) === type);
    if (!move) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'promo-btn';
    button.innerHTML = pieceSvg(color, type);
    button.setAttribute('aria-label', `Promote to ${pieceName(color, type)}`);
    button.addEventListener('click', () => {
      closePromotion();
      playMove(move);
    });
    el.promotionChoices.append(button);
  }

  el.promotion.hidden = false;
  el.promotionChoices.firstElementChild?.focus();
}

function closePromotion() {
  state.promotionPending = null;
  el.promotion.hidden = true;
}

function cancelPromotion() {
  closePromotion();
  clearSelection();
}

// ---------------------------------------------------------------------------
// Drag and drop
// ---------------------------------------------------------------------------

/** The in-flight drag, or null. */
let drag = null;
/**
 * A drag both selects a square and ends in a `click`, so the click that
 * follows a drag must be ignored or it would undo what the drag just did.
 */
let suppressClick = false;

function squareAtPoint(clientX, clientY) {
  const rect = el.board.getBoundingClientRect();
  const x = (clientX - rect.left) / rect.width;
  const y = (clientY - rect.top) / rect.height;
  if (x < 0 || x >= 1 || y < 0 || y >= 1) return -1;
  return displayOrder()[Math.floor(y * 8) * 8 + Math.floor(x * 8)];
}

/** True when `sq` holds a piece belonging to the side to move. */
function ownPieceAt(sq) {
  const piece = state.game.board[sq];
  return piece !== 0 && colorOf(piece) === state.game.turn;
}

function beginDrag(sq, clientX, clientY) {
  const piece = state.game.board[sq];
  const floating = document.createElement('div');
  floating.id = 'dragged';
  floating.innerHTML = pieceSvg(colorOf(piece), typeOf(piece));
  document.body.append(floating);

  drag = { from: sq, el: floating };
  selectSquare(sq);
  // The piece is under the pointer now, so fade the one left on the square.
  squareEls.get(sq)?.firstElementChild.classList.add('ghost');
  positionDrag(clientX, clientY);

  document.addEventListener('pointermove', onDragMove);
  document.addEventListener('pointerup', onDragEnd);
  document.addEventListener('pointercancel', onDragEnd);
}

function positionDrag(clientX, clientY) {
  const cell = el.board.getBoundingClientRect().width / 8;
  drag.el.style.setProperty('--drag-size', `${cell}px`);
  drag.el.style.left = `${clientX}px`;
  drag.el.style.top = `${clientY}px`;
}

function onDragMove(event) {
  if (!drag) return;
  event.preventDefault();
  positionDrag(event.clientX, event.clientY);
}

function onDragEnd(event) {
  if (!drag) return;
  document.removeEventListener('pointermove', onDragMove);
  document.removeEventListener('pointerup', onDragEnd);
  document.removeEventListener('pointercancel', onDragEnd);

  const from = drag.from;
  drag.el.remove();
  drag = null;
  squareEls.get(from)?.firstElementChild.classList.remove('ghost');

  // Either way the square is already selected, so the trailing click would
  // only toggle that selection back off.
  suppressClick = true;

  const target = squareAtPoint(event.clientX, event.clientY);
  // Dropped back where it started (or off the board): keep it selected so the
  // move can be finished with a second click.
  if (target === -1 || target === from) { render(); return; }

  // The piece has already been carried to the square under the pointer, so
  // sliding it there again would be a second, redundant journey.
  skipNextAnimation = true;
  if (!tryMoveTo(target)) render();
  skipNextAnimation = false;
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

function squareAt(row, col) {
  const rank = state.orientation === WHITE ? 7 - row : row;
  const file = state.orientation === WHITE ? col : 7 - col;
  return rank * 16 + file;
}

/** Where the tabbable square currently is, or the selected one. */
function focusedSquare() {
  const active = document.activeElement;
  if (active?.dataset?.square !== undefined) return Number(active.dataset.square);
  return state.selected;
}

function moveFocus(sq) {
  if (sq < 0 || sq & 0x88) return;
  squareEls.get(sq)?.focus();
}

function onBoardKeyDown(event) {
  if (event.key === 'Tab' || event.key.startsWith('F')) return;

  const cur = focusedSquare();
  if (cur === undefined || cur === -1) return;
  const row = Math.floor(cur / 16);
  const col = cur & 7;
  const deltas = {
    ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1],
  };
  const delta = deltas[event.key];
  if (delta) {
    event.preventDefault();
    moveFocus(squareAt(row + delta[0], col + delta[1]));
    return;
  }

  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    handleSquare(cur);
    return;
  }
  if (event.key === 'Escape') {
    clearSelection();
    return;
  }
}

// ---------------------------------------------------------------------------
// Game controls
// ---------------------------------------------------------------------------

function newGame() {
  finishAnimations();
  state.game = new Chess();
  state.timeline = [START_FEN];
  state.played = [];
  state.selected = -1;
  state.candidates = [];
  state.lastMove = null;
  state.hintMove = null;
  state.reviewIndex = -1;
  state.resultDismissed = false;
  state.evaluation = null;
  state.generation += 1;
  state.pendingRequest = null;
  setThinking(false);
  render();

  if (state.opponent === 'ai' && state.humanSide === BLACK) {
    requestSearch('play', state.level);
  }
}

function undo() {
  if (state.thinking) return;
  finishAnimations();
  state.resultDismissed = false;

  // Take back a full round — the computer's move and the human's.
  const takeBack = () => {
    if (state.played.length === 0) return false;
    state.game.undo();
    state.played.pop();
    state.timeline.pop();
    return true;
  };

  let took = false;
  if (state.opponent === 'ai') {
    took = takeBack() && takeBack();
  } else {
    took = takeBack();
  }
  if (!took) return;

  state.selected = -1;
  state.candidates = [];
  state.lastMove = state.played.at(-1) ?? null;
  state.hintMove = null;
  state.evaluation = null;
  state.generation += 1; // invalidate any in-flight search
  state.pendingRequest = null;
  setThinking(false);
  render();
}

function flip() {
  // Squares are about to change places, which would leave any in-flight slide
  // travelling towards the wrong one.
  finishAnimations();
  state.orientation ^= 1;
  reorderBoard();
  render();
}

function hint() {
  if (state.opponent === 'human' || !humanToMove() || state.game.status().over) return;
  requestSearch('hint', state.level);
}

// ---------------------------------------------------------------------------
// Move-list review
// ---------------------------------------------------------------------------

function enterReview(ply) {
  if (ply < 0 || ply >= state.played.length) return;
  finishAnimations();
  state.reviewIndex = ply + 1;
  render();
}

function exitReview() {
  if (state.reviewIndex === -1) return;
  finishAnimations();
  state.reviewIndex = -1;
  render();
}

// ---------------------------------------------------------------------------
// FEN box
// ---------------------------------------------------------------------------

function loadFen() {
  const text = el.fen.value.trim();
  try {
    finishAnimations();
    const game = new Chess(text);
    state.game = game;
    state.played = [];
    state.timeline = [game.fen()];
    state.selected = -1;
    state.candidates = [];
    state.lastMove = null;
    state.hintMove = null;
    state.reviewIndex = -1;
    state.resultDismissed = false;
    state.evaluation = null;
    state.generation += 1;
    state.pendingRequest = null;
    setThinking(false);
    render();
    showFenMessage('Position loaded.', 'ok');
    if (state.opponent === 'ai' && game.turn !== state.humanSide && !game.status().over) {
      requestSearch('play', state.level);
    }
  } catch (error) {
    showFenMessage(String(error?.message ?? error), 'error');
  }
}

function copyFen() {
  const text = el.fen.value.trim();
  if (!navigator.clipboard) {
    showFenMessage('Clipboard is not available in this browser.', 'error');
    return;
  }
  navigator.clipboard.writeText(text).then(
    () => showFenMessage('Copied to clipboard.', 'ok'),
    () => showFenMessage('Could not copy.', 'error'),
  );
}

// ---------------------------------------------------------------------------
// Setup and events
// ---------------------------------------------------------------------------

function setup() {
  buildBoard();

  // A reload can restore a previously unticked box, so the flag is taken from
  // the control rather than assumed to still match its default.
  state.animateOpponent = el.animate.checked;

  el.board.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    if (state.reviewIndex !== -1) exitReview();
    if (!inputEnabled()) return;

    const sq = squareAtPoint(event.clientX, event.clientY);
    if (sq === -1) return;

    // Only a piece of the side to move can be dragged; everything else is
    // left to `click`, which handles selecting and completing a move.
    if (!ownPieceAt(sq)) return;
    event.preventDefault();
    beginDrag(sq, event.clientX, event.clientY);
  });

  el.board.addEventListener('click', (event) => {
    if (suppressClick) { suppressClick = false; return; }
    const sq = squareAtPoint(event.clientX, event.clientY);
    if (sq !== -1) handleSquare(sq);
  });

  el.board.addEventListener('keydown', onBoardKeyDown);

  el.promotion.addEventListener('click', (event) => {
    if (event.target === el.promotion) cancelPromotion();
  });
  el.promotion.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') cancelPromotion();
  });

  $('btn-new').addEventListener('click', newGame);
  $('btn-undo').addEventListener('click', undo);
  $('btn-flip').addEventListener('click', flip);
  $('btn-hint').addEventListener('click', hint);
  el.resultNewGame.addEventListener('click', newGame);
  $('btn-load').addEventListener('click', loadFen);
  $('btn-copy').addEventListener('click', copyFen);

  el.animate.addEventListener('change', () => {
    state.animateOpponent = el.animate.checked;
    // Turning it off mid-slide should take effect at once.
    if (!state.animateOpponent) finishAnimations();
  });

  el.opponent.addEventListener('change', () => {
    state.opponent = el.opponent.value;
    const isHuman = state.opponent === 'human';
    el.levelField.hidden = isHuman;
    el.sideField.hidden = isHuman;
    el.animateField.hidden = isHuman;
    if (!isHuman) {
      state.humanSide = el.side.value === 'white' ? WHITE : BLACK;
      if (state.game.turn !== state.humanSide && !state.game.status().over) {
        requestSearch('play', state.level);
      }
    }
  });

  el.side.addEventListener('change', () => {
    state.humanSide = el.side.value === 'white' ? WHITE : BLACK;
    state.evaluation = null;
    state.generation += 1;
    state.pendingRequest = null;
    setThinking(false);
    render();
    if (state.opponent === 'ai' && state.game.turn !== state.humanSide && !state.game.status().over) {
      requestSearch('play', state.level);
    }
  });

  el.level.addEventListener('change', () => {
    state.level = el.level.value;
  });

  el.moves.addEventListener('click', (event) => {
    const cell = event.target.closest('li.move');
    if (!cell) return;
    const ply = Number(cell.dataset.ply);
    if (state.reviewIndex === -1) {
      enterReview(ply);
    } else {
      state.reviewIndex = ply + 1;
      render();
    }
  });

  el.moves.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') exitReview();
  });

  render();
  maybeStartEngineTurn();
}

setup();
