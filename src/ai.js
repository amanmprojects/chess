/**
 * Search — negamax with alpha-beta pruning.
 *
 * Features: iterative deepening, transposition table, quiescence search,
 * MVV-LVA capture ordering, killer and history heuristics, null-move pruning,
 * and tapered piece-square evaluation.
 */

import {
  Chess, WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, EMPTY,
  typeOf, colorOf, moveFrom, moveTo, movePromo, moveFlags,
  FLAG_CAPTURE, FLAG_PROMO, onBoard, rankOf, fileOf, to64,
} from './engine.js';

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

const MATE = 30000;
const MATE_THRESHOLD = MATE - 1000;
const INFINITY = 1 << 20;

/** Middlegame and endgame material values, in centipawns. */
const MG_VALUE = [0, 82, 337, 365, 477, 1025, 0];
const EG_VALUE = [0, 94, 281, 297, 512, 936, 0];

/** Phase weights: the game is "endgame" once the heavy pieces come off. */
const PHASE_WEIGHT = [0, 0, 1, 1, 2, 4, 0];
const TOTAL_PHASE = 24;

// Piece-square tables, written from White's point of view with rank 8 first so
// they read like a board. Values are centipawns added to the piece's worth.
const table = (rows) => {
  const t = new Int16Array(64);
  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      // rows[0] is rank 8, and index 0 of the result is a1.
      t[(7 - rank) * 8 + file] = rows[rank][file];
    }
  }
  return t;
};

const MG_PST = {
  [PAWN]: table([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [ 98, 134,  61,  95,  68, 126,  34, -11],
    [ -6,   7,  26,  31,  65,  56,  25, -20],
    [-14,  13,   6,  21,  23,  12,  17, -23],
    [-27,  -2,  -5,  12,  17,   6,  10, -25],
    [-26,  -4,  -4, -10,   3,   3,  33, -12],
    [-35,  -1, -20, -23, -15,  24,  38, -22],
    [  0,   0,   0,   0,   0,   0,   0,   0],
  ]),
  [KNIGHT]: table([
    [-167, -89, -34, -49,  61, -97, -15, -107],
    [ -73, -41,  72,  36,  23,  62,   7,  -17],
    [ -47,  60,  37,  65,  84, 129,  73,   44],
    [  -9,  17,  19,  53,  37,  69,  18,   22],
    [ -13,   4,  16,  13,  28,  19,  21,   -8],
    [ -23,  -9,  12,  10,  19,  17,  25,  -16],
    [ -29, -53, -12,  -3,  -1,  18, -14,  -19],
    [-105, -21, -58, -33, -17, -28, -19,  -23],
  ]),
  [BISHOP]: table([
    [-29,   4, -82, -37, -25, -42,   7,  -8],
    [-26,  16, -18, -13,  30,  59,  18, -47],
    [-16,  37,  43,  40,  35,  50,  37,  -2],
    [ -4,   5,  19,  50,  37,  37,   7,  -2],
    [ -6,  13,  13,  26,  34,  12,  10,   4],
    [  0,  15,  15,  15,  14,  27,  18,  10],
    [  4,  15,  16,   0,   7,  21,  33,   1],
    [-33,  -3, -14, -21, -13, -12, -39, -21],
  ]),
  [ROOK]: table([
    [ 32,  42,  32,  51,  63,   9,  31,  43],
    [ 27,  32,  58,  62,  80,  67,  26,  44],
    [ -5,  19,  26,  36,  17,  45,  61,  16],
    [-24, -11,   7,  26,  24,  35,  -8, -20],
    [-36, -26, -12,  -1,   9,  -7,   6, -23],
    [-45, -25, -16, -17,   3,   0,  -5, -33],
    [-44, -16, -20,  -9,  -1,  11,  -6, -71],
    [-19, -13,   1,  17,  16,   7, -37, -26],
  ]),
  [QUEEN]: table([
    [-28,   0,  29,  12,  59,  44,  43,  45],
    [-24, -39,  -5,   1, -16,  57,  28,  54],
    [-13, -17,   7,   8,  29,  56,  47,  57],
    [-27, -27, -16, -16,  -1,  17,  -2,   1],
    [ -9, -26,  -9, -10,  -2,  -4,   3,  -3],
    [-14,   2, -11,  -2,  -5,   2,  14,   5],
    [-35,  -8,  11,   2,   8,  15,  -3,   1],
    [ -1, -18,  -9,  10, -15, -25, -31, -50],
  ]),
  [KING]: table([
    [-65,  23,  16, -15, -56, -34,   2,  13],
    [ 29,  -1, -20,  -7,  -8,  -4, -38, -29],
    [ -9,  24,   2, -16, -20,   6,  22, -22],
    [-17, -20, -12, -27, -30, -25, -14, -36],
    [-49,  -1, -27, -39, -46, -44, -33, -51],
    [-14, -14, -22, -46, -44, -30, -15, -27],
    [  1,   7,  -8, -64, -43, -16,   9,   8],
    [-15,  36,  12, -54,   8, -28,  24,  14],
  ]),
};

const EG_PST = {
  [PAWN]: table([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [178, 173, 158, 134, 147, 132, 165, 187],
    [ 94, 100,  85,  67,  56,  53,  82,  84],
    [ 32,  24,  13,   5,  -2,   4,  17,  17],
    [ 13,   9,  -3,  -7,  -7,  -8,   3,  -1],
    [  4,   7,  -6,   1,   0,  -5,  -1,  -8],
    [ 13,   8,   8,  10,  13,   0,   2,  -7],
    [  0,   0,   0,   0,   0,   0,   0,   0],
  ]),
  [KNIGHT]: table([
    [-58, -38, -13, -28, -31, -27, -63, -99],
    [-25,  -8, -25,  -2,  -9, -25, -24, -52],
    [-24, -20,  10,   9,  -1,  -9, -19, -41],
    [-17,   3,  22,  22,  22,  11,   8, -18],
    [-18,  -6,  16,  25,  16,  17,   4, -18],
    [-23,  -3,  -1,  15,  10,  -3, -20, -22],
    [-42, -20, -10,  -5,  -2, -20, -23, -44],
    [-29, -51, -23, -15, -22, -18, -50, -64],
  ]),
  [BISHOP]: table([
    [-14, -21, -11,  -8, -7,  -9, -17, -24],
    [ -8,  -4,   7, -12, -3, -13,  -4, -14],
    [  2,  -8,   0,  -1, -2,   6,   0,   4],
    [ -3,   9,  12,   9, 14,  10,   3,   2],
    [ -6,   3,  13,  19,  7,  10,  -3,  -9],
    [-12,  -3,   8,  10, 13,   3,  -7, -15],
    [-14, -18,  -7,  -1,  4,  -9, -15, -27],
    [-23,  -9, -23,  -5, -9, -16,  -5, -17],
  ]),
  [ROOK]: table([
    [13, 10, 18, 15, 12,  12,   8,   5],
    [11, 13, 13, 11, -3,   3,   8,   3],
    [ 7,  7,  7,  5,  4,  -3,  -5,  -3],
    [ 4,  3, 13,  1,  2,   1,  -1,   2],
    [ 3,  5,  8,  4, -5,  -6,  -8, -11],
    [-4,  0, -5, -1, -7, -12,  -8, -16],
    [-6, -6,  0,  2, -9,  -9, -11,  -3],
    [-9,  2,  3, -1, -5, -13,   4, -20],
  ]),
  [QUEEN]: table([
    [ -9,  22,  22,  27,  27,  19,  10,  20],
    [-17,  20,  32,  41,  58,  25,  30,   0],
    [-20,   6,   9,  49,  47,  35,  19,   9],
    [  3,  22,  24,  45,  57,  40,  57,  36],
    [-18,  28,  19,  47,  31,  34,  39,  23],
    [-16, -27,  15,   6,   9,  17,  10,   5],
    [-22, -23, -30, -16, -16, -23, -36, -32],
    [-33, -28, -22, -43,  -5, -32, -20, -41],
  ]),
  [KING]: table([
    [-74, -35, -18, -18, -11,  15,   4, -17],
    [-12,  17,  14,  17,  17,  38,  23,  11],
    [ 10,  17,  23,  15,  20,  45,  44,  13],
    [ -8,  22,  24,  27,  26,  33,  26,   3],
    [-18,  -4,  21,  24,  27,  23,   9, -11],
    [-19,  -3,  11,  21,  23,  16,   7,  -9],
    [-27, -11,   4,  13,  14,   4,  -5, -17],
    [-53, -34, -21, -11, -28, -14, -24, -43],
  ]),
};

/** Mirror a square index vertically, to read White's tables for Black. */
const flip = (i) => i ^ 56;

const PASSED_PAWN_BONUS = [0, 10, 17, 15, 62, 168, 276, 0];
const ISOLATED_PAWN_PENALTY = -15;
const DOUBLED_PAWN_PENALTY = -10;
const BISHOP_PAIR_BONUS = 30;
const ROOK_OPEN_FILE = 25;
const ROOK_SEMI_OPEN_FILE = 12;
const TEMPO = 10;

/**
 * Static evaluation, in centipawns, from the side-to-move's point of view.
 */
export function evaluate(game) {
  const board = game.board;
  let mg = [0, 0];
  let eg = [0, 0];
  let phase = 0;

  // Pawn counts per file, used for structure terms.
  const pawnFiles = [new Int8Array(8), new Int8Array(8)];
  const pawnRanks = [[], []];
  for (let i = 0; i < 8; i++) { pawnRanks[0][i] = []; pawnRanks[1][i] = []; }

  let bishops = [0, 0];

  for (let sq = 0; sq < 128; sq++) {
    if (sq & 0x88) { sq += 7; continue; }
    const p = board[sq];
    if (p === EMPTY) continue;
    const color = colorOf(p);
    const type = typeOf(p);
    const idx = color === WHITE ? to64(sq) : flip(to64(sq));

    mg[color] += MG_VALUE[type] + MG_PST[type][idx];
    eg[color] += EG_VALUE[type] + EG_PST[type][idx];
    phase += PHASE_WEIGHT[type];

    if (type === PAWN) {
      pawnFiles[color][fileOf(sq)]++;
      pawnRanks[color][fileOf(sq)].push(rankOf(sq));
    } else if (type === BISHOP) {
      bishops[color]++;
    }
  }

  for (const color of [WHITE, BLACK]) {
    const them = color ^ 1;

    if (bishops[color] >= 2) {
      mg[color] += BISHOP_PAIR_BONUS;
      eg[color] += BISHOP_PAIR_BONUS;
    }

    for (let file = 0; file < 8; file++) {
      const count = pawnFiles[color][file];
      if (count === 0) continue;

      if (count > 1) {
        const penalty = DOUBLED_PAWN_PENALTY * (count - 1);
        mg[color] += penalty;
        eg[color] += penalty;
      }

      const leftEmpty = file === 0 || pawnFiles[color][file - 1] === 0;
      const rightEmpty = file === 7 || pawnFiles[color][file + 1] === 0;
      if (leftEmpty && rightEmpty) {
        mg[color] += ISOLATED_PAWN_PENALTY;
        eg[color] += ISOLATED_PAWN_PENALTY;
      }

      // Passed pawns: no enemy pawn ahead on this or an adjacent file.
      for (const rank of pawnRanks[color][file]) {
        const relRank = color === WHITE ? rank : 7 - rank;
        let passed = true;
        for (let f = Math.max(0, file - 1); f <= Math.min(7, file + 1) && passed; f++) {
          for (const enemyRank of pawnRanks[them][f]) {
            const ahead = color === WHITE ? enemyRank > rank : enemyRank < rank;
            if (ahead) { passed = false; break; }
          }
        }
        if (passed) {
          mg[color] += PASSED_PAWN_BONUS[relRank] >> 1;
          eg[color] += PASSED_PAWN_BONUS[relRank];
        }
      }
    }
  }

  // Rooks on open and half-open files.
  for (let sq = 0; sq < 128; sq++) {
    if (sq & 0x88) { sq += 7; continue; }
    const p = board[sq];
    if (p === EMPTY || typeOf(p) !== ROOK) continue;
    const color = colorOf(p);
    const file = fileOf(sq);
    if (pawnFiles[color][file] === 0) {
      const bonus = pawnFiles[color ^ 1][file] === 0 ? ROOK_OPEN_FILE : ROOK_SEMI_OPEN_FILE;
      mg[color] += bonus;
      eg[color] += bonus;
    }
  }

  // Taper between the middlegame and endgame scores.
  const clamped = Math.min(phase, TOTAL_PHASE);
  const mgScore = mg[WHITE] - mg[BLACK];
  const egScore = eg[WHITE] - eg[BLACK];
  const score = (mgScore * clamped + egScore * (TOTAL_PHASE - clamped)) / TOTAL_PHASE;

  const sideScore = game.turn === WHITE ? score : -score;
  return Math.round(sideScore) + TEMPO;
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

const TT_EXACT = 0;
const TT_LOWER = 1;
const TT_UPPER = 2;

class TranspositionTable {
  constructor(sizeMb = 32) {
    // Each entry is 4 Int32s: keyHi, keyLo, move, packed(score/depth/flag).
    const entries = Math.max(1024, (sizeMb * 1024 * 1024) / 16) | 0;
    this.mask = (1 << (31 - Math.clz32(entries))) - 1; // round down to a power of two
    this.keysHi = new Int32Array(this.mask + 1);
    this.keysLo = new Int32Array(this.mask + 1);
    this.moves = new Int32Array(this.mask + 1);
    this.scores = new Int32Array(this.mask + 1);
    this.depths = new Int8Array(this.mask + 1);
    this.flags = new Int8Array(this.mask + 1);
    this.used = new Uint8Array(this.mask + 1);
  }

  clear() {
    this.used.fill(0);
  }

  probe(hi, lo) {
    const i = (lo ^ hi) & this.mask;
    if (!this.used[i] || this.keysHi[i] !== hi || this.keysLo[i] !== lo) return null;
    return { move: this.moves[i], score: this.scores[i], depth: this.depths[i], flag: this.flags[i] };
  }

  store(hi, lo, move, score, depth, flag) {
    const i = (lo ^ hi) & this.mask;
    // Depth-preferred replacement, but always overwrite a different position.
    if (this.used[i] && this.keysHi[i] === hi && this.keysLo[i] === lo
        && this.depths[i] > depth && flag !== TT_EXACT) {
      return;
    }
    this.keysHi[i] = hi;
    this.keysLo[i] = lo;
    this.moves[i] = move;
    this.scores[i] = score;
    this.depths[i] = depth;
    this.flags[i] = flag;
    this.used[i] = 1;
  }
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------

/** Most Valuable Victim / Least Valuable Attacker. */
const MVV_LVA_VICTIM = [0, 100, 300, 320, 500, 900, 0];

const MAX_PLY = 64;

export class Search {
  constructor({ ttSizeMb = 32 } = {}) {
    this.tt = new TranspositionTable(ttSizeMb);
    this.killers = Array.from({ length: MAX_PLY }, () => [0, 0]);
    this.history = new Int32Array(128 * 128);
    this.nodes = 0;
    this.stopped = false;
    // Zobrist halves of the positions played before the search root, so the
    // tree can see repetitions that started in the actual game.
    this.rootHi = [];
    this.rootLo = [];
  }

  scoreMove(game, move, ply, ttMove) {
    if (move === ttMove) return 1_000_000;

    const flags = moveFlags(move);
    const to = moveTo(move);
    const from = moveFrom(move);

    if (flags & FLAG_PROMO) {
      return 900_000 + MVV_LVA_VICTIM[movePromo(move)];
    }

    if (flags & FLAG_CAPTURE) {
      const victim = game.board[to];
      const victimType = victim === EMPTY ? PAWN : typeOf(victim); // ep captures a pawn
      const attackerType = typeOf(game.board[from]);
      return 800_000 + MVV_LVA_VICTIM[victimType] * 10 - MVV_LVA_VICTIM[attackerType] / 100;
    }

    if (this.killers[ply][0] === move) return 700_000;
    if (this.killers[ply][1] === move) return 600_000;

    return this.history[from * 128 + to];
  }

  orderMoves(game, moves, ply, ttMove) {
    const scored = moves.map((m) => ({ m, s: this.scoreMove(game, m, ply, ttMove) }));
    scored.sort((a, b) => b.s - a.s);
    return scored.map((x) => x.m);
  }

  // -------------------------------------------------------------------------
  // Quiescence — only captures and promotions, so the evaluation is not taken
  // in the middle of an exchange.
  // -------------------------------------------------------------------------

  quiescence(game, alpha, beta, ply) {
    if ((this.nodes & 2047) === 0 && this.outOfTime()) { this.stopped = true; return 0; }
    this.nodes++;

    const standPat = evaluate(game);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;
    if (ply >= MAX_PLY - 1) return standPat;

    // Delta pruning: if even capturing a queen cannot reach alpha, give up.
    if (standPat + MG_VALUE[QUEEN] < alpha) return alpha;

    const moves = this.orderMoves(game, game.generateMoves({ capturesOnly: true }), ply, 0);
    const us = game.turn;

    for (const move of moves) {
      game.makeMove(move);
      if (game.isSquareAttacked(game.kings[us], us ^ 1)) { game.undoMove(); continue; }
      const score = -this.quiescence(game, -beta, -alpha, ply + 1);
      game.undoMove();

      if (this.stopped) return 0;
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }

    return alpha;
  }

  // -------------------------------------------------------------------------
  // Main negamax search
  // -------------------------------------------------------------------------

  negamax(game, depth, alpha, beta, ply, canNull = true) {
    if ((this.nodes & 2047) === 0 && this.outOfTime()) { this.stopped = true; return 0; }

    // Draw detection inside the tree — a repetition or the fifty-move rule is
    // worth exactly zero, and missing it makes the engine play into draws.
    if (ply > 0) {
      if (game.halfmoves >= 100 || this.isRepetition(game)) return 0;
      if (game.isInsufficientMaterial()) return 0;
    }

    const inCheck = game.inCheck();
    if (inCheck) depth++; // check extension

    if (depth <= 0) return this.quiescence(game, alpha, beta, ply);

    this.nodes++;

    const alphaOrig = alpha;
    let ttMove = 0;
    const entry = this.tt.probe(game.hashHi, game.hashLo);
    if (entry) {
      ttMove = entry.move;
      if (ply > 0 && entry.depth >= depth) {
        let score = entry.score;
        // Mate scores are stored relative to the root; re-anchor them to this ply.
        if (score > MATE_THRESHOLD) score -= ply;
        else if (score < -MATE_THRESHOLD) score += ply;

        if (entry.flag === TT_EXACT) return score;
        if (entry.flag === TT_LOWER && score > alpha) alpha = score;
        else if (entry.flag === TT_UPPER && score < beta) beta = score;
        if (alpha >= beta) return score;
      }
    }

    // Null-move pruning: give the opponent a free move; if we are still winning
    // comfortably, this node is unlikely to matter. Skipped in check, in the
    // endgame (zugzwang), and when the position is already losing.
    if (canNull && !inCheck && depth >= 3 && ply > 0 && this.hasNonPawnMaterial(game)) {
      const R = 2 + (depth > 6 ? 1 : 0);
      game.makeNullMove();
      const score = -this.negamax(game, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
      game.undoNullMove();
      if (this.stopped) return 0;
      if (score >= beta && Math.abs(score) < MATE_THRESHOLD) return beta;
    }

    const moves = this.orderMoves(game, game.generateMoves({ legal: false }), ply, ttMove);
    const us = game.turn;
    let bestMove = 0;
    let bestScore = -INFINITY;
    let legalCount = 0;

    for (let i = 0; i < moves.length; i++) {
      const move = moves[i];
      game.makeMove(move);
      if (game.isSquareAttacked(game.kings[us], us ^ 1)) { game.undoMove(); continue; }
      legalCount++;

      let score;
      if (legalCount === 1) {
        score = -this.negamax(game, depth - 1, -beta, -alpha, ply + 1);
      } else {
        // Late move reductions: quiet moves late in a well-ordered list are
        // searched shallower first, and re-searched only if they beat alpha.
        const isQuiet = (moveFlags(move) & (FLAG_CAPTURE | FLAG_PROMO)) === 0;
        let reduction = 0;
        if (depth >= 3 && legalCount > 3 && isQuiet && !inCheck) {
          reduction = legalCount > 6 ? 2 : 1;
        }
        score = -this.negamax(game, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
        if (score > alpha && reduction > 0) {
          score = -this.negamax(game, depth - 1, -alpha - 1, -alpha, ply + 1);
        }
        if (score > alpha && score < beta) {
          score = -this.negamax(game, depth - 1, -beta, -alpha, ply + 1);
        }
      }

      game.undoMove();
      if (this.stopped) return 0;

      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }

      if (score > alpha) {
        alpha = score;
        if (score >= beta) {
          // A quiet move that causes a cutoff is worth remembering.
          if ((moveFlags(move) & (FLAG_CAPTURE | FLAG_PROMO)) === 0) {
            if (this.killers[ply][0] !== move) {
              this.killers[ply][1] = this.killers[ply][0];
              this.killers[ply][0] = move;
            }
            this.history[moveFrom(move) * 128 + moveTo(move)] += depth * depth;
          }
          break;
        }
      }
    }

    // No legal moves: checkmate (scored by distance, so shorter mates win) or
    // stalemate.
    if (legalCount === 0) {
      return inCheck ? -MATE + ply : 0;
    }

    let stored = bestScore;
    if (stored > MATE_THRESHOLD) stored += ply;
    else if (stored < -MATE_THRESHOLD) stored -= ply;

    const flag = bestScore <= alphaOrig ? TT_UPPER : bestScore >= beta ? TT_LOWER : TT_EXACT;
    this.tt.store(game.hashHi, game.hashLo, bestMove, stored, depth, flag);

    return bestScore;
  }

  hasNonPawnMaterial(game) {
    const us = game.turn;
    for (let sq = 0; sq < 128; sq++) {
      if (sq & 0x88) { sq += 7; continue; }
      const p = game.board[sq];
      if (p === EMPTY || colorOf(p) !== us) continue;
      const t = typeOf(p);
      if (t !== PAWN && t !== KING) return true;
    }
    return false;
  }

  /**
   * Load the positions played before the search root.
   *
   * `game.positions` holds `"hi,lo"` key strings for every position of the real
   * game, including the root itself as its last element. The search compares
   * numbers, so parse once here rather than per node.
   */
  setupRepetitionHistory(game) {
    this.rootHi = [];
    this.rootLo = [];

    const keys = game.positions;
    if (!Array.isArray(keys) || keys.length === 0) return;

    // Drop the trailing entry when it is the root — the in-tree walk covers it.
    let end = keys.length;
    if (keys[end - 1] === game.hashKey()) end--;

    for (let i = 0; i < end; i++) {
      const key = keys[i];
      if (typeof key !== 'string') continue;
      const comma = key.indexOf(',');
      if (comma < 0) continue;
      // Both halves are Int32s, so a leading '-' is the only other character.
      this.rootHi.push(Number(key.slice(0, comma)) | 0);
      this.rootLo.push(Number(key.slice(comma + 1)) | 0);
    }
  }

  isRepetition(game) {
    // Only positions since the last irreversible move can repeat, and the
    // halfmove clock counts exactly those plies. It spans the in-tree undo
    // stack first, then continues into the pre-root game history.
    let remaining = game.halfmoves;
    if (remaining < 4) return false; // a repetition needs at least four plies

    const hist = game.history;
    for (let i = hist.length - 1; i >= 0 && remaining > 0; i--, remaining--) {
      const state = hist[i];
      // A null move fabricates a position that real play cannot reach, so
      // anything beyond it is not a genuine ancestor of this node.
      if (state.null) return false;
      if (state.hashHi === game.hashHi && state.hashLo === game.hashLo) return true;
    }

    for (let i = this.rootHi.length - 1; i >= 0 && remaining > 0; i--, remaining--) {
      if (this.rootHi[i] === game.hashHi && this.rootLo[i] === game.hashLo) return true;
    }

    return false;
  }

  outOfTime() {
    return this.deadline > 0 && Date.now() >= this.deadline;
  }

  /**
   * Find the best move by iterative deepening.
   *
   * @param {Chess}   game
   * @param {object}  opts
   * @param {number}  [opts.depth=64]        maximum depth
   * @param {number}  [opts.movetime=1000]   milliseconds to think (0 = no limit)
   * @param {Function}[opts.onIteration]     called after each completed depth
   * @returns {{move:number, score:number, depth:number, pv:number[], nodes:number}}
   */
  findBestMove(game, { depth = 64, movetime = 1000, onIteration } = {}) {
    this.nodes = 0;
    this.stopped = false;
    this.deadline = movetime > 0 ? Date.now() + movetime : 0;
    this.killers = Array.from({ length: MAX_PLY }, () => [0, 0]);
    this.history.fill(0);
    this.setupRepetitionHistory(game);

    const rootMoves = game.generateMoves();
    if (rootMoves.length === 0) return { move: 0, score: 0, depth: 0, pv: [], nodes: 0 };
    if (rootMoves.length === 1) {
      return { move: rootMoves[0], score: 0, depth: 1, pv: [rootMoves[0]], nodes: 1 };
    }

    let best = { move: rootMoves[0], score: -INFINITY, depth: 0, pv: [], nodes: 0 };
    let alpha = -INFINITY;
    let beta = INFINITY;

    for (let d = 1; d <= depth; d++) {
      const score = this.negamax(game, d, alpha, beta, 0);

      if (this.stopped) break;

      // Aspiration windows: retry with a wider window if we fell outside it.
      if (score <= alpha || score >= beta) {
        alpha = -INFINITY;
        beta = INFINITY;
        d--;
        continue;
      }

      alpha = score - 50;
      beta = score + 50;

      const pv = this.extractPv(game, d);
      best = {
        move: pv[0] ?? best.move,
        score,
        depth: d,
        pv,
        nodes: this.nodes,
      };

      onIteration?.(best);

      // A forced mate has been found — no need to search deeper.
      if (Math.abs(score) > MATE_THRESHOLD) break;
    }

    return best;
  }

  /** Walk the transposition table to recover the principal variation. */
  extractPv(game, maxLength) {
    const pv = [];
    const seen = new Set();
    for (let i = 0; i < maxLength; i++) {
      const entry = this.tt.probe(game.hashHi, game.hashLo);
      if (!entry || !entry.move) break;
      const key = `${game.hashHi},${game.hashLo}`;
      if (seen.has(key)) break; // a repetition loop in the table
      seen.add(key);
      // Confirm the stored move is legal here before trusting it.
      if (!game.generateMoves().includes(entry.move)) break;
      pv.push(entry.move);
      game.makeMove(entry.move);
    }
    for (let i = 0; i < pv.length; i++) game.undoMove();
    return pv;
  }
}

// ---------------------------------------------------------------------------
// Difficulty levels
// ---------------------------------------------------------------------------

export const LEVELS = {
  beginner:     { depth: 1, movetime: 100,  randomness: 180, label: 'Beginner' },
  casual:       { depth: 2, movetime: 200,  randomness: 90,  label: 'Casual' },
  intermediate: { depth: 4, movetime: 600,  randomness: 30,  label: 'Intermediate' },
  strong:       { depth: 8, movetime: 1500, randomness: 0,   label: 'Strong' },
  expert:       { depth: 20, movetime: 4000, randomness: 0,  label: 'Expert' },
};

/**
 * Pick a move at the given difficulty. Weaker levels deliberately blunder by
 * sampling among moves whose score is within `randomness` centipawns of the
 * best — that yields human-like inaccuracy rather than nonsense.
 */
export function chooseMove(game, levelName = 'intermediate', { search, onIteration } = {}) {
  const level = LEVELS[levelName] ?? LEVELS.intermediate;
  const engine = search ?? new Search();

  const result = engine.findBestMove(game, {
    depth: level.depth,
    movetime: level.movetime,
    onIteration,
  });

  if (!level.randomness || !result.move) return result;

  // The root search may have just stopped on its time budget. The re-scoring
  // below is shallow (at most level.depth - 1), so give it the time it needs:
  // an aborted search returns 0, which would flatten every score and turn the
  // near-best sampling into a completely random move.
  engine.stopped = false;
  engine.deadline = 0;

  // Re-score the root moves shallowly and sample from the near-best set.
  const rootMoves = game.generateMoves();
  const scored = [];
  for (const move of rootMoves) {
    game.makeMove(move);
    const score = -engine.negamax(game, Math.max(0, level.depth - 1), -INFINITY, INFINITY, 1);
    game.undoMove();
    scored.push({ move, score });
  }
  scored.sort((a, b) => b.score - a.score);

  const cutoff = scored[0].score - level.randomness;
  const pool = scored.filter((s) => s.score >= cutoff);
  const pick = pool[Math.floor(Math.random() * pool.length)];

  return { ...result, move: pick.move, score: pick.score };
}

export { MATE, MATE_THRESHOLD, MG_VALUE };
