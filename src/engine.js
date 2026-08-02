/**
 * chess engine — 0x88 board representation.
 *
 * Implements the complete rules of chess: sliding/leaping move generation,
 * castling (incl. all its legality conditions), en passant, promotion,
 * check/checkmate/stalemate, the fifty-move rule, threefold repetition and
 * insufficient material. Move generation is verified against standard perft
 * results in test/perft.test.mjs.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const WHITE = 0;
export const BLACK = 1;

export const PAWN = 1;
export const KNIGHT = 2;
export const BISHOP = 3;
export const ROOK = 4;
export const QUEEN = 5;
export const KING = 6;

export const EMPTY = 0;

/** A piece is `color << 3 | type`, so white pawn = 1, black pawn = 9. */
export const piece = (color, type) => (color << 3) | type;
export const colorOf = (p) => p >> 3;
export const typeOf = (p) => p & 7;

export const START_FEN =
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// Castling rights bitmask.
export const CR_WK = 1;
export const CR_WQ = 2;
export const CR_BK = 4;
export const CR_BQ = 8;

// Move flags.
export const FLAG_CAPTURE = 1;
export const FLAG_EP = 2;
export const FLAG_DOUBLE = 4;
export const FLAG_KCASTLE = 8;
export const FLAG_QCASTLE = 16;
export const FLAG_PROMO = 32;

// Named squares (0x88: index = rank * 16 + file, a1 = 0).
const A1 = 0, C1 = 2, D1 = 3, E1 = 4, F1 = 5, G1 = 6, H1 = 7;
const A8 = 112, C8 = 114, D8 = 115, E8 = 116, F8 = 117, G8 = 118, H8 = 119;

const KNIGHT_OFFSETS = [33, 31, 18, 14, -14, -18, -31, -33];
const BISHOP_OFFSETS = [17, 15, -15, -17];
const ROOK_OFFSETS = [16, 1, -1, -16];
const KING_OFFSETS = [17, 16, 15, 1, -1, -15, -16, -17];

const SLIDER_OFFSETS = {
  [BISHOP]: BISHOP_OFFSETS,
  [ROOK]: ROOK_OFFSETS,
  [QUEEN]: KING_OFFSETS,
};

const PIECE_LETTERS = { 1: 'p', 2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'k' };
const LETTER_PIECES = { p: PAWN, n: KNIGHT, b: BISHOP, r: ROOK, q: QUEEN, k: KING };

/**
 * Moving from or to one of these squares clears the listed castling rights.
 * (Every other square leaves rights untouched, hence the default of 15.)
 */
const CASTLING_MASK = new Int8Array(128).fill(15);
CASTLING_MASK[E1] = 15 & ~(CR_WK | CR_WQ);
CASTLING_MASK[H1] = 15 & ~CR_WK;
CASTLING_MASK[A1] = 15 & ~CR_WQ;
CASTLING_MASK[E8] = 15 & ~(CR_BK | CR_BQ);
CASTLING_MASK[H8] = 15 & ~CR_BK;
CASTLING_MASK[A8] = 15 & ~CR_BQ;

// ---------------------------------------------------------------------------
// Square helpers
// ---------------------------------------------------------------------------

export const onBoard = (sq) => (sq & 0x88) === 0;
export const rankOf = (sq) => sq >> 4;
export const fileOf = (sq) => sq & 15;
/** Square index used by 8x8 arrays (0 = a1 … 63 = h8). */
export const to64 = (sq) => (sq >> 4) * 8 + (sq & 7);
export const from64 = (i) => ((i / 8) | 0) * 16 + (i % 8);

export function algebraic(sq) {
  return 'abcdefgh'[fileOf(sq)] + (rankOf(sq) + 1);
}

export function squareFromAlgebraic(s) {
  const file = 'abcdefgh'.indexOf(s[0]);
  const rank = s.charCodeAt(1) - 49; // '1' -> 0
  if (file < 0 || rank < 0 || rank > 7) return -1;
  return rank * 16 + file;
}

/** True if the square is light-coloured (a1 is dark). */
export const isLightSquare = (sq) => ((rankOf(sq) + fileOf(sq)) & 1) === 1;

// ---------------------------------------------------------------------------
// Move encoding — packed into a single 32-bit integer
//   bits  0.. 6  from square
//   bits  7..13  to square
//   bits 14..16  promotion piece type (0 = none)
//   bits 17..22  flags
// ---------------------------------------------------------------------------

export const encodeMove = (from, to, promo = 0, flags = 0) =>
  from | (to << 7) | (promo << 14) | (flags << 17);

export const moveFrom = (m) => m & 0x7f;
export const moveTo = (m) => (m >> 7) & 0x7f;
export const movePromo = (m) => (m >> 14) & 7;
export const moveFlags = (m) => (m >> 17) & 0x3f;

/** Long algebraic / UCI form, e.g. "e2e4" or "e7e8q". */
export function moveToUci(m) {
  const promo = movePromo(m);
  return algebraic(moveFrom(m)) + algebraic(moveTo(m)) + (promo ? PIECE_LETTERS[promo] : '');
}

// ---------------------------------------------------------------------------
// Zobrist hashing (two 32-bit halves — no BigInt, so it stays fast)
// ---------------------------------------------------------------------------

function makeRandom(seed) {
  // xorshift32, seeded deterministically so hashes are reproducible.
  let x = seed | 0;
  return () => {
    x ^= x << 13; x |= 0;
    x ^= x >>> 17;
    x ^= x << 5; x |= 0;
    return x;
  };
}

const rand32 = makeRandom(0x9e3779b9);
const Z_PIECE_HI = [], Z_PIECE_LO = [];
for (let p = 0; p < 16; p++) {
  Z_PIECE_HI[p] = new Int32Array(128);
  Z_PIECE_LO[p] = new Int32Array(128);
  for (let sq = 0; sq < 128; sq++) {
    Z_PIECE_HI[p][sq] = rand32();
    Z_PIECE_LO[p][sq] = rand32();
  }
}
const Z_CASTLE_HI = new Int32Array(16), Z_CASTLE_LO = new Int32Array(16);
for (let i = 0; i < 16; i++) { Z_CASTLE_HI[i] = rand32(); Z_CASTLE_LO[i] = rand32(); }
const Z_EP_HI = new Int32Array(8), Z_EP_LO = new Int32Array(8);
for (let i = 0; i < 8; i++) { Z_EP_HI[i] = rand32(); Z_EP_LO[i] = rand32(); }
const Z_SIDE_HI = rand32(), Z_SIDE_LO = rand32();

// ---------------------------------------------------------------------------
// The game
// ---------------------------------------------------------------------------

export class Chess {
  constructor(fen = START_FEN) {
    this.load(fen);
  }

  load(fen = START_FEN) {
    const parts = fen.trim().split(/\s+/);
    if (parts.length < 4) throw new Error(`Invalid FEN: ${fen}`);
    const [placement, side, castling, ep] = parts;

    this.board = new Int8Array(128);
    this.kings = [-1, -1];

    let sq = A8;
    for (const ch of placement) {
      if (ch === '/') {
        sq -= 24; // down a rank, back to the a-file
      } else if (ch >= '1' && ch <= '8') {
        sq += ch.charCodeAt(0) - 48;
      } else {
        const type = LETTER_PIECES[ch.toLowerCase()];
        if (!type) throw new Error(`Invalid FEN piece '${ch}' in: ${fen}`);
        const color = ch === ch.toUpperCase() ? WHITE : BLACK;
        this.board[sq] = piece(color, type);
        if (type === KING) this.kings[color] = sq;
        sq++;
      }
    }

    // A position without both kings is not a chess position; every attack and
    // check query below assumes they exist, so reject it here rather than
    // return nonsense later.
    if (this.kings[WHITE] < 0 || this.kings[BLACK] < 0) {
      throw new Error(`Invalid FEN — both kings are required: ${fen}`);
    }

    this.turn = side === 'b' ? BLACK : WHITE;

    this.castling = 0;
    if (castling.includes('K')) this.castling |= CR_WK;
    if (castling.includes('Q')) this.castling |= CR_WQ;
    if (castling.includes('k')) this.castling |= CR_BK;
    if (castling.includes('q')) this.castling |= CR_BQ;

    this.ep = ep === '-' ? -1 : squareFromAlgebraic(ep);
    this.halfmoves = parts.length > 4 ? parseInt(parts[4], 10) || 0 : 0;
    this.fullmoves = parts.length > 5 ? parseInt(parts[5], 10) || 1 : 1;

    this.history = [];
    this.computeHash();
    /** Zobrist keys of every position seen, for repetition detection. */
    this.positions = [this.hashKey()];
    return this;
  }

  clone() {
    const c = new Chess(this.fen());
    c.positions = this.positions.slice();
    return c;
  }

  fen() {
    let placement = '';
    for (let rank = 7; rank >= 0; rank--) {
      let empty = 0;
      for (let file = 0; file < 8; file++) {
        const p = this.board[rank * 16 + file];
        if (p === EMPTY) { empty++; continue; }
        if (empty) { placement += empty; empty = 0; }
        const letter = PIECE_LETTERS[typeOf(p)];
        placement += colorOf(p) === WHITE ? letter.toUpperCase() : letter;
      }
      if (empty) placement += empty;
      if (rank > 0) placement += '/';
    }

    let rights = '';
    if (this.castling & CR_WK) rights += 'K';
    if (this.castling & CR_WQ) rights += 'Q';
    if (this.castling & CR_BK) rights += 'k';
    if (this.castling & CR_BQ) rights += 'q';

    return [
      placement,
      this.turn === WHITE ? 'w' : 'b',
      rights || '-',
      this.ep >= 0 ? algebraic(this.ep) : '-',
      this.halfmoves,
      this.fullmoves,
    ].join(' ');
  }

  get(square) {
    const sq = typeof square === 'string' ? squareFromAlgebraic(square) : square;
    const p = this.board[sq];
    return p === EMPTY ? null : { color: colorOf(p), type: typeOf(p) };
  }

  // -------------------------------------------------------------------------
  // Hashing
  // -------------------------------------------------------------------------

  computeHash() {
    let hi = 0, lo = 0;
    for (let sq = 0; sq < 128; sq++) {
      if (sq & 0x88) { sq += 7; continue; }
      const p = this.board[sq];
      if (p !== EMPTY) { hi ^= Z_PIECE_HI[p][sq]; lo ^= Z_PIECE_LO[p][sq]; }
    }
    hi ^= Z_CASTLE_HI[this.castling]; lo ^= Z_CASTLE_LO[this.castling];
    if (this.ep >= 0) { hi ^= Z_EP_HI[fileOf(this.ep)]; lo ^= Z_EP_LO[fileOf(this.ep)]; }
    if (this.turn === BLACK) { hi ^= Z_SIDE_HI; lo ^= Z_SIDE_LO; }
    this.hashHi = hi; this.hashLo = lo;
    return this;
  }

  hashKey() {
    return `${this.hashHi},${this.hashLo}`;
  }

  // -------------------------------------------------------------------------
  // Attacks
  // -------------------------------------------------------------------------

  /** True if `color` attacks `sq` (ignores pins — that is what "attacks" means). */
  isSquareAttacked(sq, color) {
    // Pawns. A white pawn on sq-17/sq-15 attacks sq.
    const pawn = piece(color, PAWN);
    const dir = color === WHITE ? -16 : 16;
    for (const side of [-1, 1]) {
      const from = sq + dir + side;
      if (onBoard(from) && this.board[from] === pawn) return true;
    }

    const knight = piece(color, KNIGHT);
    for (const off of KNIGHT_OFFSETS) {
      const from = sq + off;
      if (onBoard(from) && this.board[from] === knight) return true;
    }

    const king = piece(color, KING);
    for (const off of KING_OFFSETS) {
      const from = sq + off;
      if (onBoard(from) && this.board[from] === king) return true;
    }

    const queen = piece(color, QUEEN);
    const bishop = piece(color, BISHOP);
    for (const off of BISHOP_OFFSETS) {
      for (let from = sq + off; onBoard(from); from += off) {
        const p = this.board[from];
        if (p === EMPTY) continue;
        if (p === bishop || p === queen) return true;
        break;
      }
    }

    const rook = piece(color, ROOK);
    for (const off of ROOK_OFFSETS) {
      for (let from = sq + off; onBoard(from); from += off) {
        const p = this.board[from];
        if (p === EMPTY) continue;
        if (p === rook || p === queen) return true;
        break;
      }
    }

    return false;
  }

  /** Is `color` (default: side to move) currently in check? */
  inCheck(color = this.turn) {
    return this.isSquareAttacked(this.kings[color], color ^ 1);
  }

  // -------------------------------------------------------------------------
  // Move generation
  // -------------------------------------------------------------------------

  /**
   * @param {object}  [opts]
   * @param {boolean} [opts.legal=true]        filter out moves leaving the king in check
   * @param {number}  [opts.square]            only moves originating from this square
   * @param {boolean} [opts.capturesOnly]      captures and promotions only (quiescence)
   */
  generateMoves({ legal = true, square = -1, capturesOnly = false } = {}) {
    const moves = [];
    const us = this.turn;
    const them = us ^ 1;

    const first = square >= 0 ? square : A1;
    const last = square >= 0 ? square : H8;

    for (let from = first; from <= last; from++) {
      if (from & 0x88) { from += 7; continue; }
      const p = this.board[from];
      if (p === EMPTY || colorOf(p) !== us) continue;
      const type = typeOf(p);

      if (type === PAWN) {
        const dir = us === WHITE ? 16 : -16;
        const startRank = us === WHITE ? 1 : 6;
        const promoRank = us === WHITE ? 7 : 0;

        // Pushes.
        const one = from + dir;
        if (!capturesOnly && onBoard(one) && this.board[one] === EMPTY) {
          if (rankOf(one) === promoRank) {
            for (const promo of [QUEEN, ROOK, BISHOP, KNIGHT]) {
              moves.push(encodeMove(from, one, promo, FLAG_PROMO));
            }
          } else {
            moves.push(encodeMove(from, one, 0, 0));
            const two = from + dir * 2;
            if (rankOf(from) === startRank && this.board[two] === EMPTY) {
              moves.push(encodeMove(from, two, 0, FLAG_DOUBLE));
            }
          }
        } else if (capturesOnly && onBoard(one) && this.board[one] === EMPTY
                   && rankOf(one) === promoRank) {
          // Quiescence still wants promotions, which change material.
          moves.push(encodeMove(from, one, QUEEN, FLAG_PROMO));
        }

        // Captures.
        for (const side of [-1, 1]) {
          const to = from + dir + side;
          if (!onBoard(to)) continue;
          const target = this.board[to];
          if (target !== EMPTY && colorOf(target) === them) {
            if (rankOf(to) === promoRank) {
              for (const promo of [QUEEN, ROOK, BISHOP, KNIGHT]) {
                moves.push(encodeMove(from, to, promo, FLAG_PROMO | FLAG_CAPTURE));
              }
            } else {
              moves.push(encodeMove(from, to, 0, FLAG_CAPTURE));
            }
          } else if (to === this.ep) {
            moves.push(encodeMove(from, to, 0, FLAG_CAPTURE | FLAG_EP));
          }
        }
        continue;
      }

      if (type === KNIGHT || type === KING) {
        const offsets = type === KNIGHT ? KNIGHT_OFFSETS : KING_OFFSETS;
        for (const off of offsets) {
          const to = from + off;
          if (!onBoard(to)) continue;
          const target = this.board[to];
          if (target === EMPTY) {
            if (!capturesOnly) moves.push(encodeMove(from, to, 0, 0));
          } else if (colorOf(target) === them) {
            moves.push(encodeMove(from, to, 0, FLAG_CAPTURE));
          }
        }
        continue;
      }

      // Sliding pieces.
      for (const off of SLIDER_OFFSETS[type]) {
        for (let to = from + off; onBoard(to); to += off) {
          const target = this.board[to];
          if (target === EMPTY) {
            if (!capturesOnly) moves.push(encodeMove(from, to, 0, 0));
            continue;
          }
          if (colorOf(target) === them) moves.push(encodeMove(from, to, 0, FLAG_CAPTURE));
          break;
        }
      }
    }

    // Castling. The king may not start in check, pass through an attacked
    // square, or land on one; the squares between king and rook must be empty.
    if (!capturesOnly) {
      const kingSq = this.kings[us];
      const generateCastles = square < 0 || square === kingSq;
      if (generateCastles && kingSq >= 0) {
        const kingSide = us === WHITE ? CR_WK : CR_BK;
        const queenSide = us === WHITE ? CR_WQ : CR_BQ;
        const [e, f, g, d, c, b] = us === WHITE
          ? [E1, F1, G1, D1, C1, 1]
          : [E8, F8, G8, D8, C8, 113];

        if ((this.castling & kingSide)
            && this.board[f] === EMPTY && this.board[g] === EMPTY
            && !this.isSquareAttacked(e, them)
            && !this.isSquareAttacked(f, them)
            && !this.isSquareAttacked(g, them)) {
          moves.push(encodeMove(e, g, 0, FLAG_KCASTLE));
        }

        if ((this.castling & queenSide)
            && this.board[d] === EMPTY && this.board[c] === EMPTY && this.board[b] === EMPTY
            && !this.isSquareAttacked(e, them)
            && !this.isSquareAttacked(d, them)
            && !this.isSquareAttacked(c, them)) {
          moves.push(encodeMove(e, c, 0, FLAG_QCASTLE));
        }
      }
    }

    if (!legal) return moves;

    const legalMoves = [];
    for (const move of moves) {
      this.makeMove(move);
      if (!this.isSquareAttacked(this.kings[us], them)) legalMoves.push(move);
      this.undoMove();
    }
    return legalMoves;
  }

  /** Legal moves as objects — convenient for the UI. */
  moves({ square } = {}) {
    const sq = typeof square === 'string' ? squareFromAlgebraic(square) : square ?? -1;
    return this.generateMoves({ square: sq }).map((m) => ({
      move: m,
      from: algebraic(moveFrom(m)),
      to: algebraic(moveTo(m)),
      promotion: movePromo(m) ? PIECE_LETTERS[movePromo(m)] : undefined,
      san: this.toSan(m),
      uci: moveToUci(m),
    }));
  }

  // -------------------------------------------------------------------------
  // Make / unmake
  // -------------------------------------------------------------------------

  makeMove(move) {
    const from = moveFrom(move);
    const to = moveTo(move);
    const flags = moveFlags(move);
    const promo = movePromo(move);
    const us = this.turn;
    const them = us ^ 1;
    const moving = this.board[from];
    const movingType = typeOf(moving);

    let captured = EMPTY;
    let capturedSq = to;
    if (flags & FLAG_EP) {
      capturedSq = us === WHITE ? to - 16 : to + 16;
      captured = this.board[capturedSq];
    } else if (this.board[to] !== EMPTY) {
      captured = this.board[to];
    }

    this.history.push({
      move,
      captured,
      capturedSq,
      castling: this.castling,
      ep: this.ep,
      halfmoves: this.halfmoves,
      hashHi: this.hashHi,
      hashLo: this.hashLo,
      kingSq: this.kings[us],
    });

    // --- Remove old castling/ep from the hash before mutating them.
    this.hashHi ^= Z_CASTLE_HI[this.castling]; this.hashLo ^= Z_CASTLE_LO[this.castling];
    if (this.ep >= 0) {
      this.hashHi ^= Z_EP_HI[fileOf(this.ep)]; this.hashLo ^= Z_EP_LO[fileOf(this.ep)];
    }

    // --- Captured piece leaves the board.
    if (captured !== EMPTY) {
      this.board[capturedSq] = EMPTY;
      this.hashHi ^= Z_PIECE_HI[captured][capturedSq];
      this.hashLo ^= Z_PIECE_LO[captured][capturedSq];
    }

    // --- Move the piece (promoting if required).
    const placed = promo ? piece(us, promo) : moving;
    this.board[from] = EMPTY;
    this.hashHi ^= Z_PIECE_HI[moving][from]; this.hashLo ^= Z_PIECE_LO[moving][from];
    this.board[to] = placed;
    this.hashHi ^= Z_PIECE_HI[placed][to]; this.hashLo ^= Z_PIECE_LO[placed][to];

    // --- Rook hop when castling.
    if (flags & (FLAG_KCASTLE | FLAG_QCASTLE)) {
      const rookFrom = flags & FLAG_KCASTLE ? to + 1 : to - 2;
      const rookTo = flags & FLAG_KCASTLE ? to - 1 : to + 1;
      const rook = this.board[rookFrom];
      this.board[rookFrom] = EMPTY;
      this.board[rookTo] = rook;
      this.hashHi ^= Z_PIECE_HI[rook][rookFrom] ^ Z_PIECE_HI[rook][rookTo];
      this.hashLo ^= Z_PIECE_LO[rook][rookFrom] ^ Z_PIECE_LO[rook][rookTo];
    }

    if (movingType === KING) this.kings[us] = to;

    // --- New castling rights and en passant square.
    this.castling &= CASTLING_MASK[from] & CASTLING_MASK[to];
    this.ep = flags & FLAG_DOUBLE ? (us === WHITE ? from + 16 : from - 16) : -1;

    this.hashHi ^= Z_CASTLE_HI[this.castling]; this.hashLo ^= Z_CASTLE_LO[this.castling];
    if (this.ep >= 0) {
      this.hashHi ^= Z_EP_HI[fileOf(this.ep)]; this.hashLo ^= Z_EP_LO[fileOf(this.ep)];
    }

    this.halfmoves = (movingType === PAWN || captured !== EMPTY) ? 0 : this.halfmoves + 1;
    if (us === BLACK) this.fullmoves++;

    this.turn = them;
    this.hashHi ^= Z_SIDE_HI; this.hashLo ^= Z_SIDE_LO;

    return this;
  }

  undoMove() {
    const state = this.history.pop();
    if (!state) return null;

    const { move, captured, capturedSq } = state;
    const from = moveFrom(move);
    const to = moveTo(move);
    const flags = moveFlags(move);
    const promo = movePromo(move);

    this.turn ^= 1;
    const us = this.turn;
    if (us === BLACK) this.fullmoves--;

    const placed = this.board[to];
    this.board[to] = EMPTY;
    this.board[from] = promo ? piece(us, PAWN) : placed;

    if (flags & (FLAG_KCASTLE | FLAG_QCASTLE)) {
      const rookFrom = flags & FLAG_KCASTLE ? to + 1 : to - 2;
      const rookTo = flags & FLAG_KCASTLE ? to - 1 : to + 1;
      this.board[rookFrom] = this.board[rookTo];
      this.board[rookTo] = EMPTY;
    }

    if (captured !== EMPTY) this.board[capturedSq] = captured;
    if (typeOf(placed) === KING) this.kings[us] = state.kingSq;

    this.castling = state.castling;
    this.ep = state.ep;
    this.halfmoves = state.halfmoves;
    this.hashHi = state.hashHi;
    this.hashLo = state.hashLo;

    return move;
  }

  /** Pass the turn without moving — used for null-move pruning in the search. */
  makeNullMove() {
    this.history.push({
      move: 0, captured: EMPTY, capturedSq: -1,
      castling: this.castling, ep: this.ep, halfmoves: this.halfmoves,
      hashHi: this.hashHi, hashLo: this.hashLo, kingSq: this.kings[this.turn],
      null: true,
    });
    if (this.ep >= 0) {
      this.hashHi ^= Z_EP_HI[fileOf(this.ep)]; this.hashLo ^= Z_EP_LO[fileOf(this.ep)];
      this.ep = -1;
    }
    this.turn ^= 1;
    this.hashHi ^= Z_SIDE_HI; this.hashLo ^= Z_SIDE_LO;
    this.halfmoves++;
  }

  undoNullMove() {
    const state = this.history.pop();
    this.turn ^= 1;
    this.ep = state.ep;
    this.halfmoves = state.halfmoves;
    this.hashHi = state.hashHi;
    this.hashLo = state.hashLo;
  }

  // -------------------------------------------------------------------------
  // Playing moves (with repetition bookkeeping)
  // -------------------------------------------------------------------------

  /**
   * Play a move given as SAN ("Nf3"), UCI ("g1f3"), an encoded int, or an
   * object `{from, to, promotion}`. Returns a record of the move, or null if
   * it is not legal in this position.
   */
  move(input) {
    const encoded = this.resolveMove(input);
    if (encoded == null) return null;

    const record = {
      san: this.toSan(encoded),
      uci: moveToUci(encoded),
      from: algebraic(moveFrom(encoded)),
      to: algebraic(moveTo(encoded)),
      color: this.turn,
      move: encoded,
      before: this.fen(),
    };

    this.makeMove(encoded);
    this.positions.push(this.hashKey());
    record.after = this.fen();
    return record;
  }

  /** Undo the last played move. */
  undo() {
    if (this.history.length === 0) return null;
    this.positions.pop();
    return this.undoMove();
  }

  /** Turn any accepted move notation into an encoded legal move, or null. */
  resolveMove(input) {
    const legal = this.generateMoves();

    if (typeof input === 'number') return legal.includes(input) ? input : null;

    if (typeof input === 'object' && input !== null) {
      const from = typeof input.from === 'string' ? squareFromAlgebraic(input.from) : input.from;
      const to = typeof input.to === 'string' ? squareFromAlgebraic(input.to) : input.to;
      const promo = input.promotion ? LETTER_PIECES[String(input.promotion).toLowerCase()] : 0;
      for (const m of legal) {
        if (moveFrom(m) !== from || moveTo(m) !== to) continue;
        if (movePromo(m) && promo && movePromo(m) !== promo) continue;
        if (movePromo(m) && !promo && movePromo(m) !== QUEEN) continue;
        return m;
      }
      return null;
    }

    if (typeof input === 'string') {
      const text = input.trim();
      // UCI first — it is unambiguous.
      if (/^[a-h][1-8][a-h][1-8][qrbn]?$/i.test(text)) {
        const from = squareFromAlgebraic(text.slice(0, 2).toLowerCase());
        const to = squareFromAlgebraic(text.slice(2, 4).toLowerCase());
        const promo = text.length > 4 ? LETTER_PIECES[text[4].toLowerCase()] : 0;
        for (const m of legal) {
          if (moveFrom(m) === from && moveTo(m) === to && movePromo(m) === promo) return m;
        }
        return null;
      }
      const normalise = (s) => s.replace(/[+#!?]+$/, '').replace(/0/g, 'O');
      const want = normalise(text);

      // Exact SAN match first.
      for (const m of legal) {
        if (normalise(this.toSan(m)) === want) return m;
      }

      // Then a structural parse, which also accepts over-specified SAN such as
      // "Rh1g1" or "e7e8=Q" that a strict generator would never emit.
      if (/^O-O(-O)?$/.test(want)) {
        const flag = want === 'O-O' ? FLAG_KCASTLE : FLAG_QCASTLE;
        return legal.find((m) => moveFlags(m) & flag) ?? null;
      }

      const m = /^([KQRBN])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([QRBN]))?$/.exec(want);
      if (!m) return null;
      const [, pieceLetter, fromFile, fromRank, toSquare, promoLetter] = m;
      const type = pieceLetter ? LETTER_PIECES[pieceLetter.toLowerCase()] : PAWN;
      const to = squareFromAlgebraic(toSquare);
      const promo = promoLetter ? LETTER_PIECES[promoLetter.toLowerCase()] : 0;

      const candidates = legal.filter((move) => {
        if (moveTo(move) !== to) return false;
        const from = moveFrom(move);
        if (typeOf(this.board[from]) !== type) return false;
        if (fromFile && fileOf(from) !== 'abcdefgh'.indexOf(fromFile)) return false;
        if (fromRank && rankOf(from) !== Number(fromRank) - 1) return false;
        if (promo) return movePromo(move) === promo;
        // An unspecified promotion piece means a queen, matching move({from,to}).
        return movePromo(move) === 0 || movePromo(move) === QUEEN;
      });

      return candidates.length === 1 ? candidates[0] : null;
    }

    return null;
  }

  // -------------------------------------------------------------------------
  // Standard Algebraic Notation
  // -------------------------------------------------------------------------

  toSan(move) {
    const flags = moveFlags(move);
    if (flags & FLAG_KCASTLE) return this.withCheckSuffix(move, 'O-O');
    if (flags & FLAG_QCASTLE) return this.withCheckSuffix(move, 'O-O-O');

    const from = moveFrom(move);
    const to = moveTo(move);
    const type = typeOf(this.board[from]);
    const isCapture = (flags & FLAG_CAPTURE) !== 0;
    let san = '';

    if (type === PAWN) {
      if (isCapture) san += 'abcdefgh'[fileOf(from)] + 'x';
      san += algebraic(to);
    } else {
      san += PIECE_LETTERS[type].toUpperCase();
      san += this.disambiguate(move, type, from, to);
      if (isCapture) san += 'x';
      san += algebraic(to);
    }

    const promo = movePromo(move);
    if (promo) san += '=' + PIECE_LETTERS[promo].toUpperCase();

    return this.withCheckSuffix(move, san);
  }

  disambiguate(move, type, from, to) {
    const rivals = [];
    for (const m of this.generateMoves()) {
      if (m === move) continue;
      if (moveTo(m) !== to) continue;
      const f = moveFrom(m);
      if (f === from) continue;
      if (typeOf(this.board[f]) !== type) continue;
      rivals.push(f);
    }
    if (rivals.length === 0) return '';
    const sameFile = rivals.some((f) => fileOf(f) === fileOf(from));
    const sameRank = rivals.some((f) => rankOf(f) === rankOf(from));
    if (!sameFile) return 'abcdefgh'[fileOf(from)];
    if (!sameRank) return String(rankOf(from) + 1);
    return algebraic(from);
  }

  withCheckSuffix(move, san) {
    this.makeMove(move);
    let suffix = '';
    if (this.inCheck()) {
      suffix = this.generateMoves().length === 0 ? '#' : '+';
    }
    this.undoMove();
    return san + suffix;
  }

  // -------------------------------------------------------------------------
  // Game state
  // -------------------------------------------------------------------------

  isCheckmate() {
    return this.inCheck() && this.generateMoves().length === 0;
  }

  isStalemate() {
    return !this.inCheck() && this.generateMoves().length === 0;
  }

  isFiftyMoveDraw() {
    return this.halfmoves >= 100;
  }

  isThreefoldRepetition() {
    const key = this.hashKey();
    let count = 0;
    for (const k of this.positions) if (k === key) count++;
    return count >= 3;
  }

  /** K vs K, K+minor vs K, and K+B vs K+B with same-coloured bishops. */
  isInsufficientMaterial() {
    const bishops = [];
    let knights = 0;
    let others = 0;

    for (let sq = 0; sq < 128; sq++) {
      if (sq & 0x88) { sq += 7; continue; }
      const p = this.board[sq];
      if (p === EMPTY) continue;
      const type = typeOf(p);
      if (type === KING) continue;
      if (type === BISHOP) bishops.push(isLightSquare(sq));
      else if (type === KNIGHT) knights++;
      else others++;
    }

    if (others > 0) return false;
    if (knights === 0 && bishops.length === 0) return true;              // K vs K
    if (knights === 1 && bishops.length === 0) return true;              // K+N vs K
    if (knights === 0 && bishops.length === 1) return true;              // K+B vs K
    if (knights === 0 && bishops.length > 1) {
      return bishops.every((light) => light === bishops[0]);             // same-colour bishops
    }
    return false;
  }

  /**
   * @returns {{over: boolean, result: string|null, reason: string|null}}
   *   result is '1-0', '0-1', '1/2-1/2' or null while the game continues.
   */
  status() {
    if (this.generateMoves().length === 0) {
      if (this.inCheck()) {
        return {
          over: true,
          result: this.turn === WHITE ? '0-1' : '1-0',
          reason: 'checkmate',
        };
      }
      return { over: true, result: '1/2-1/2', reason: 'stalemate' };
    }
    if (this.isInsufficientMaterial()) {
      return { over: true, result: '1/2-1/2', reason: 'insufficient material' };
    }
    if (this.isFiftyMoveDraw()) {
      return { over: true, result: '1/2-1/2', reason: 'fifty-move rule' };
    }
    if (this.isThreefoldRepetition()) {
      return { over: true, result: '1/2-1/2', reason: 'threefold repetition' };
    }
    return { over: false, result: null, reason: null };
  }

  // -------------------------------------------------------------------------
  // Perft — the correctness yardstick for move generation
  // -------------------------------------------------------------------------

  perft(depth) {
    if (depth === 0) return 1;
    const moves = this.generateMoves({ legal: false });
    const us = this.turn;
    const them = us ^ 1;
    let nodes = 0;
    for (const move of moves) {
      this.makeMove(move);
      if (!this.isSquareAttacked(this.kings[us], them)) {
        nodes += depth === 1 ? 1 : this.perft(depth - 1);
      }
      this.undoMove();
    }
    return nodes;
  }
}

export { PIECE_LETTERS, LETTER_PIECES };
