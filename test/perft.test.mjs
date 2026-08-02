/**
 * Perft (performance test) counts every leaf node of the legal move tree to a
 * given depth. The expected counts below are the standard published values, so
 * a match means move generation handles castling, en passant, promotion, pins
 * and discovered checks exactly as the rules require.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Chess, START_FEN } from '../src/engine.js';

const POSITIONS = [
  {
    name: 'initial position',
    fen: START_FEN,
    counts: [20, 400, 8902, 197281, 4865609],
  },
  {
    name: 'kiwipete (castling, pins, checks)',
    fen: 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    counts: [48, 2039, 97862, 4085603],
  },
  {
    name: 'en passant discovered check',
    fen: '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    counts: [14, 191, 2812, 43238, 674624],
  },
  {
    name: 'promotion and underpromotion',
    fen: 'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
    counts: [6, 264, 9467, 422333],
  },
  {
    name: 'mirrored promotion position',
    fen: 'r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1',
    counts: [6, 264, 9467, 422333],
  },
  {
    name: 'tactical middlegame',
    fen: 'rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8',
    counts: [44, 1486, 62379, 2103487],
  },
  {
    name: 'symmetric position',
    fen: 'r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10',
    counts: [46, 2079, 89890, 3894594],
  },
];

// Depth 5 of the start position and depth 4 elsewhere run in a few seconds;
// keeping them in the default suite means regressions surface immediately.
for (const { name, fen, counts } of POSITIONS) {
  test(`perft: ${name}`, () => {
    const game = new Chess(fen);
    counts.forEach((expected, i) => {
      const depth = i + 1;
      assert.equal(game.perft(depth), expected, `depth ${depth} of ${fen}`);
      // The board must be pristine after a perft run — make/unmake symmetry.
      assert.equal(game.fen(), fen, `board mutated after perft(${depth})`);
    });
  });
}

test('make/unmake restores the zobrist hash exactly', () => {
  const game = new Chess('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1');

  const walk = (depth) => {
    if (depth === 0) return;
    for (const move of game.generateMoves()) {
      const hi = game.hashHi, lo = game.hashLo, fen = game.fen();
      game.makeMove(move);
      // Recomputing from scratch must agree with the incremental update.
      const incHi = game.hashHi, incLo = game.hashLo;
      game.computeHash();
      assert.equal(game.hashHi, incHi, 'hash hi drifted');
      assert.equal(game.hashLo, incLo, 'hash lo drifted');
      walk(depth - 1);
      game.undoMove();
      assert.equal(game.hashHi, hi);
      assert.equal(game.hashLo, lo);
      assert.equal(game.fen(), fen);
    }
  };

  walk(3);
});
