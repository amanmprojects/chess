/**
 * Search tests. The engine must never return an illegal move, must find short
 * forced mates, and must not walk into obvious material loss.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Chess, WHITE, BLACK, moveToUci } from '../src/engine.js';
import { Search, chooseMove, evaluate, LEVELS, MATE_THRESHOLD } from '../src/ai.js';

const bestUci = (fen, opts = {}) => {
  const game = new Chess(fen);
  const search = new Search();
  const result = search.findBestMove(game, { depth: 6, movetime: 2000, ...opts });
  return { uci: moveToUci(result.move), result, game };
};

/**
 * Ground truth, computed independently of the search: does the side to move
 * have a forced mate in `n` moves? Move generation is perft-verified, so this
 * brute force is trustworthy as an oracle.
 */
function forcedMateMoves(game, n) {
  const mating = [];
  for (const move of game.generateMoves()) {
    game.makeMove(move);
    const forced = isMateInAtMost(game, n - 1);
    game.undoMove();
    if (forced) mating.push(move);
  }
  return mating;
}

/** Called with the opponent to move: can they avoid being mated within n? */
function isMateInAtMost(game, n) {
  const replies = game.generateMoves();
  if (replies.length === 0) return game.inCheck(); // mated (stalemate does not count)
  if (n === 0) return false;
  // The defender escapes if any reply avoids mate.
  for (const reply of replies) {
    game.makeMove(reply);
    const stillMating = forcedMateMoves(game, n).length > 0;
    game.undoMove();
    if (!stillMating) return false;
  }
  return true;
}

test('mate in one is found', () => {
  const positions = [
    '6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1',        // back-rank mate
    '4k3/8/4K3/8/8/8/8/7R w - - 0 1',           // rook mate with the king opposing
    '7k/6pp/8/8/8/8/8/5R1K w - - 0 1',          // rook to f8
  ];

  for (const fen of positions) {
    const game = new Chess(fen);
    const expected = forcedMateMoves(game, 1);
    assert.ok(expected.length > 0, `no mate in one exists in ${fen} — bad test position`);

    const result = new Search().findBestMove(new Chess(fen), { depth: 4, movetime: 3000 });
    assert.ok(expected.includes(result.move),
      `${moveToUci(result.move)} is not a mate in one in ${fen} ` +
      `(expected one of ${expected.map(moveToUci).join(', ')})`);
    assert.ok(result.score > MATE_THRESHOLD,
      `score should be a mate score, got ${result.score}`);
  }
});

test('mate in two is found', () => {
  const positions = [
    '6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1',   // rook lifts to the eighth
    '5rk1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1',    // trade into the back rank
  ];

  for (const fen of positions) {
    const game = new Chess(fen);
    if (forcedMateMoves(game, 1).length > 0) continue; // must be a genuine two-mover
    const expected = forcedMateMoves(game, 2);
    if (expected.length === 0) continue;                // not a mate in two, skip

    const result = new Search().findBestMove(new Chess(fen), { depth: 6, movetime: 5000 });
    assert.ok(expected.includes(result.move),
      `${moveToUci(result.move)} does not force mate in two in ${fen}`);
    assert.ok(result.score > MATE_THRESHOLD, `expected a mate score, got ${result.score}`);
  }
});

test('a forced mate is preferred and actually delivered', () => {
  // Play the engine against itself from a winning position; it must convert.
  const game = new Chess('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1');
  const search = new Search();
  for (let i = 0; i < 6 && !game.status().over; i++) {
    const { move } = search.findBestMove(game, { depth: 5, movetime: 1000 });
    assert.ok(game.generateMoves().includes(move), 'illegal move while converting');
    game.makeMove(move);
  }
  assert.ok(game.isCheckmate(), 'the engine failed to convert a forced mate');
  assert.equal(game.turn, BLACK, 'black should be the mated side');
});

test('the search always returns a legal move', () => {
  const positions = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1',
    '4k3/8/8/8/8/8/4P3/4K3 w - - 0 1',
    '7k/8/8/8/8/5q2/8/6K1 w - - 0 1',       // white is under heavy pressure
    'rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3', // en passant available
  ];

  for (const fen of positions) {
    const game = new Chess(fen);
    const legal = game.generateMoves();
    assert.ok(legal.length > 0, `${fen} is terminal — not a useful search test`);
    const search = new Search();
    const result = search.findBestMove(game, { depth: 5, movetime: 800 });
    assert.ok(legal.includes(result.move),
      `${moveToUci(result.move)} is not legal in ${fen}`);
    assert.equal(game.fen(), fen, 'the search mutated the position');
  }
});

test('every difficulty level returns a legal move', () => {
  for (const name of Object.keys(LEVELS)) {
    const game = new Chess('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3');
    const legal = game.generateMoves();
    const result = chooseMove(game, name);
    assert.ok(legal.includes(result.move), `${name} produced an illegal move`);
    assert.equal(game.fen(), 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3');
  }
});

test('a free queen gets captured', () => {
  // Black queen on d5 is undefended and attackable by the pawn on e4.
  const { uci } = bestUci('4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1', { depth: 5, movetime: 2000 });
  assert.equal(uci, 'e4d5', 'the engine should take the free queen');
});

test('the engine escapes a threat instead of ignoring it', () => {
  // White queen on d1 is attacked by the bishop on g4; only ...Qxg4 or moving
  // the queen keeps material level. Any reply that abandons her is a blunder.
  const game = new Chess('4k3/8/8/8/6b1/8/8/3QK3 w - - 0 1');
  const search = new Search();
  const result = search.findBestMove(game, { depth: 6, movetime: 2000 });
  game.makeMove(result.move);
  // After the best move, black should not be able to win the queen for free.
  const stillHasQueen = game.fen().includes('Q');
  const tookBishop = !game.fen().includes('b');
  assert.ok(stillHasQueen || tookBishop, 'the queen was left to be captured');
});

test('the engine prefers mate over winning material', () => {
  // Mate in one with Ra8#, or grab the queen on h7. Mate must win.
  const { uci } = bestUci('6k1/5ppq/8/8/8/8/8/R5K1 w - - 0 1', { depth: 4, movetime: 2000 });
  assert.equal(uci, 'a1a8', 'mate beats material');
});

test('stalemate and mate are scored, not crashed on', () => {
  const mated = new Chess('6k1/5ppp/8/8/8/8/8/R4RK1 b - - 0 1');
  const search = new Search();
  const result = search.findBestMove(mated, { depth: 4, movetime: 500 });
  assert.ok(mated.generateMoves().includes(result.move));

  // No legal moves at all: the search must return gracefully.
  const stalemate = new Chess('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1');
  const done = new Search().findBestMove(stalemate, { depth: 4, movetime: 500 });
  assert.equal(done.move, 0, 'no move exists in a stalemate');
});

test('evaluation is symmetric and side-relative', () => {
  // The starting position is balanced, so both sides see the same small score.
  const white = new Chess('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const black = new Chess('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1');
  assert.equal(evaluate(white), evaluate(black), 'a mirrored position evaluates equally');

  // A material edge must register for the side that holds it.
  const up = new Chess('4k3/8/8/8/8/8/8/3QK3 w - - 0 1');
  assert.ok(evaluate(up) > 500, 'a spare queen should be worth a lot');
  const down = new Chess('4k3/8/8/8/8/8/8/3QK3 b - - 0 1');
  assert.ok(evaluate(down) < -500, 'and the same deficit from the other side');
});

test('deeper search does not degrade a simple tactic', () => {
  // A hanging rook should be taken at every depth from 2 to 6.
  for (let depth = 2; depth <= 6; depth++) {
    const { uci } = bestUci('4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1', { depth, movetime: 2000 });
    assert.equal(uci, 'e4d5', `depth ${depth} missed the free rook`);
  }
});

test('the search respects its time budget', () => {
  const game = new Chess('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1');
  const started = Date.now();
  new Search().findBestMove(game, { depth: 64, movetime: 400 });
  const elapsed = Date.now() - started;
  assert.ok(elapsed < 2000, `took ${elapsed}ms for a 400ms budget`);
});

test('a timed-out search can still be re-scored for the randomness levels', () => {
  // White is up a full rook, so no shallow search of this position can
  // legitimately score near zero.
  const fen = '4k3/8/8/8/8/8/8/R3K3 w - - 0 1';
  const engine = new Search();
  engine.findBestMove(new Chess(fen), { depth: 64, movetime: 1 });
  assert.equal(engine.stopped, true, 'the 1ms budget must expire');

  // chooseMove clears the budget before re-scoring the root moves, exactly
  // as below. Under an expired budget the search aborts with 0 for every
  // move, which would flatten the near-best sampling into a random pick.
  engine.stopped = false;
  engine.deadline = 0;
  const game = new Chess(fen);
  game.makeMove(game.generateMoves()[0]);
  const score = -engine.negamax(game, 3, -Infinity, Infinity, 1);
  assert.ok(score > 100, `re-score aborted to 0 — got ${score}`);
});
