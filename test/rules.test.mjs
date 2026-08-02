/**
 * Rules tests: the specific situations that trip up chess implementations.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Chess, START_FEN, WHITE, BLACK } from '../src/engine.js';

const sans = (game) => game.moves().map((m) => m.san).sort();
const ucis = (game, square) => game.moves({ square }).map((m) => m.uci).sort();
/** Castling moves available, with any check/mate suffix stripped. */
const castles = (game) =>
  sans(game).map((s) => s.replace(/[+#]$/, '')).filter((s) => s === 'O-O' || s === 'O-O-O');

// ---------------------------------------------------------------------------
// FEN round-tripping
// ---------------------------------------------------------------------------

test('FEN parses and serialises losslessly', () => {
  const fens = [
    START_FEN,
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 12 47',
    '4k3/8/8/8/8/8/4P3/4K3 w - - 0 1',
    'rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3',
  ];
  for (const fen of fens) {
    assert.equal(new Chess(fen).fen(), fen);
  }
});

// ---------------------------------------------------------------------------
// Castling
// ---------------------------------------------------------------------------

test('castling both sides is available and moves the rook', () => {
  const game = new Chess('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  assert.deepEqual(castles(game).sort(), ['O-O', 'O-O-O']);

  game.move('O-O');
  assert.equal(game.get('g1').type, 6, 'king on g1');
  assert.equal(game.get('f1').type, 4, 'rook hopped to f1');
  assert.equal(game.get('h1'), null);
  assert.equal(game.get('e1'), null);

  game.move('O-O-O');
  assert.equal(game.get('c8').type, 6, 'black king on c8');
  assert.equal(game.get('d8').type, 4, 'black rook on d8');
  assert.equal(game.get('a8'), null);
});

test('cannot castle out of, through, or into check', () => {
  // Rook on e8 checks the king: cannot castle at all.
  let game = new Chess('4r2k/8/8/8/8/8/8/R3K2R w KQ - 0 1');
  assert.deepEqual(castles(game), [], 'may not castle out of check');

  // Rook on f8 attacks f1, the square the king crosses kingside.
  game = new Chess('5r1k/8/8/8/8/8/8/R3K2R w KQ - 0 1');
  assert.ok(!castles(game).includes('O-O'), 'may not pass through attacked f1');
  assert.ok(castles(game).includes('O-O-O'), 'queenside is unaffected');

  // Rook on g8 attacks g1, the king's destination.
  game = new Chess('k5r1/8/8/8/8/8/8/R3K2R w KQ - 0 1');
  assert.ok(!castles(game).includes('O-O'), 'may not land on attacked g1');

  // Rook on d8 attacks d1, crossed when castling queenside.
  game = new Chess('3r3k/8/8/8/8/8/8/R3K2R w KQ - 0 1');
  assert.ok(!castles(game).includes('O-O-O'), 'may not pass through attacked d1');
  assert.ok(castles(game).includes('O-O'), 'kingside is unaffected');
});

test('queenside castling is legal while b1 is merely attacked', () => {
  // A black pawn on a2 attacks b1. Castling queenside sends the king e1->c1 and
  // the rook a1->d1, so neither ever occupies b1: b1 only has to be empty.
  const game = new Chess('2k5/8/8/8/8/8/p7/R3K2R w KQ - 0 1');
  assert.ok(castles(game).includes('O-O-O'), 'b1 attacked does not forbid O-O-O');
});

test('castling is blocked by occupied squares', () => {
  const game = new Chess('r3k2r/8/8/8/8/8/8/R2QK1NR w KQkq - 0 1');
  assert.deepEqual(castles(game), [], 'knight on g1 and queen on d1 both block');
});

test('moving the king or rook forfeits the matching rights', () => {
  let game = new Chess('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  game.move('Rh1g1');
  assert.ok(!game.fen().split(' ')[2].includes('K'), 'h-rook move clears white kingside');
  assert.ok(game.fen().split(' ')[2].includes('Q'), 'queenside survives');

  game = new Chess('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  game.move('Ke1d1');
  assert.equal(game.fen().split(' ')[2], 'kq', 'king move clears both white rights');
});

test('capturing a rook on its home square clears those rights', () => {
  const game = new Chess('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  game.move('Ra1a8');
  assert.equal(game.fen().split(' ')[2], 'Kk', 'both a-file rights gone');
});

// ---------------------------------------------------------------------------
// En passant
// ---------------------------------------------------------------------------

test('en passant capture removes the passed pawn', () => {
  const game = new Chess('rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3');
  const ep = game.moves({ square: 'd4' }).find((m) => m.to === 'e3');
  assert.ok(ep, 'dxe3 en passant is generated');
  game.move(ep.uci);
  assert.equal(game.get('e3').type, 1, 'pawn landed on e3');
  assert.equal(game.get('e4'), null, 'the captured pawn is gone');
});

test('en passant is only legal immediately', () => {
  const game = new Chess('rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3');
  game.move('Nc6');   // black declines the capture
  game.move('Nf3');
  assert.ok(!ucis(game, 'd4').includes('d4e3'), 'the right has expired');
});

test('a double push sets the ep square only when generated', () => {
  const game = new Chess(START_FEN);
  game.move('e4');
  assert.equal(game.fen().split(' ')[3], 'e3');
  game.move('e5');
  assert.equal(game.fen().split(' ')[3], 'e6');
  game.move('Nf3');
  assert.equal(game.fen().split(' ')[3], '-');
});

test('en passant that would expose the king is rejected', () => {
  // White king a5, white pawn b5, black rook h5. Black plays c7-c5; capturing
  // bxc6 would clear both b5 and c5 from the rank and expose the king.
  const game = new Chess('8/2p5/8/KP5r/8/8/8/7k b - - 0 1');
  game.move('c5');
  assert.equal(game.fen().split(' ')[3], 'c6');
  assert.ok(!ucis(game, 'b5').includes('b5c6'), 'the ep capture is illegal here');
});

// ---------------------------------------------------------------------------
// Promotion
// ---------------------------------------------------------------------------

test('promotion offers all four pieces', () => {
  const game = new Chess('8/4P3/8/8/8/8/8/K6k w - - 0 1');
  const promos = game.moves({ square: 'e7' }).map((m) => m.san).sort();
  assert.deepEqual(promos, ['e8=B', 'e8=N', 'e8=Q', 'e8=R']);
});

test('promotion places the chosen piece', () => {
  let game = new Chess('8/4P3/8/8/8/8/8/K6k w - - 0 1');
  game.move('e8=N');
  assert.equal(game.get('e8').type, 2, 'knight, not an auto-queen');

  game = new Chess('8/4P3/8/8/8/8/8/K6k w - - 0 1');
  game.move({ from: 'e7', to: 'e8' });
  assert.equal(game.get('e8').type, 5, 'object form defaults to a queen');
});

test('capture-promotion works and undo restores a pawn', () => {
  const game = new Chess('3r4/4P3/8/8/8/8/8/K6k w - - 0 1');
  const move = game.moves({ square: 'e7' }).find((m) => m.san.startsWith('exd8=Q'));
  assert.ok(move);
  game.move(move.uci);
  assert.equal(game.get('d8').type, 5);
  game.undo();
  assert.equal(game.get('e7').type, 1, 'pawn is back');
  assert.equal(game.get('d8').type, 4, 'captured rook is back');
});

// ---------------------------------------------------------------------------
// Check, pins, mate
// ---------------------------------------------------------------------------

test('a pinned piece may not abandon the king', () => {
  // Black rook e8, white king e1, white knight e2 pinned along the file.
  const game = new Chess('4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1');
  assert.deepEqual(ucis(game, 'e2'), [], 'the pinned knight is frozen');
});

test('a pinned piece may still capture along the pin line', () => {
  const game = new Chess('4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1');
  assert.ok(ucis(game, 'e2').includes('e2e8'), 'the rook can take the pinner');
});

test('in check, only resolving moves are legal', () => {
  // White king e1 checked by a rook on e8; the bishop can interpose on e5.
  const game = new Chess('4r2k/8/8/8/1B6/8/8/4K3 w - - 0 1');
  assert.ok(game.inCheck());
  for (const m of game.moves()) {
    const probe = new Chess(game.fen());
    probe.move(m.uci);
    assert.ok(!probe.inCheck(WHITE), `${m.san} left the king in check`);
  }
  assert.ok(sans(game).includes('Be7'), 'interposing on the e-file is allowed');
});

test('back-rank mate is detected', () => {
  const game = new Chess('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1');
  game.move('Ra8');
  assert.ok(game.isCheckmate());
  assert.deepEqual(game.status(), { over: true, result: '1-0', reason: 'checkmate' });
});

test("fool's mate", () => {
  const game = new Chess();
  for (const m of ['f3', 'e5', 'g4', 'Qh4#']) assert.ok(game.move(m), `${m} should be legal`);
  assert.ok(game.isCheckmate());
  assert.equal(game.status().result, '0-1');
});

test('smothered stalemate is a draw, not a mate', () => {
  const game = new Chess('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1');
  assert.ok(game.isStalemate());
  assert.ok(!game.isCheckmate());
  assert.deepEqual(game.status(), { over: true, result: '1/2-1/2', reason: 'stalemate' });
});

test('the king cannot step into an attacked square', () => {
  const game = new Chess('7k/8/8/8/8/8/5q2/4K3 w - - 0 1');
  const dests = game.moves({ square: 'e1' }).map((m) => m.to).sort();
  // The queen on f2 covers d1/e1/f1/f2/d2/e2 — only nothing remains except capture-free flight.
  assert.ok(!dests.includes('d2'));
  assert.ok(!dests.includes('e2'));
  assert.ok(!dests.includes('f1'));
});

test('the king may not stay on a slider ray when fleeing', () => {
  // Black rook e8 checks along the file; e2 is still on the ray.
  const game = new Chess('4r2k/8/8/8/8/8/8/4K3 w - - 0 1');
  const dests = game.moves({ square: 'e1' }).map((m) => m.to).sort();
  assert.deepEqual(dests, ['d1', 'd2', 'f1', 'f2']);
});

// ---------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------

test('insufficient material combinations', () => {
  const draws = [
    '8/8/4k3/8/8/4K3/8/8 w - - 0 1',        // K vs K
    '8/8/4k3/8/8/4K3/8/5N2 w - - 0 1',      // K+N vs K
    '8/8/4k3/8/8/4K3/8/5B2 w - - 0 1',      // K+B vs K
    '2b5/8/4k3/8/8/4K3/8/5B2 w - - 0 1',    // bishops, both on light squares
  ];
  for (const fen of draws) {
    assert.ok(new Chess(fen).isInsufficientMaterial(), fen);
  }

  const notDraws = [
    '8/8/4k3/8/8/4K3/8/4BB2 w - - 0 1',     // two bishops, opposite colours
    '8/8/4k3/8/8/4K3/8/4NN2 w - - 0 1',     // two knights
    '8/8/4k3/8/8/4K3/8/5R2 w - - 0 1',      // rook
    '8/8/4k3/8/8/4K3/4P3/8 w - - 0 1',      // pawn
  ];
  for (const fen of notDraws) {
    assert.ok(!new Chess(fen).isInsufficientMaterial(), fen);
  }
});

test('fifty-move rule', () => {
  const game = new Chess('4k3/8/8/8/8/8/8/R3K3 w - - 99 60');
  assert.ok(!game.isFiftyMoveDraw());
  game.move('Ra2');
  assert.ok(game.isFiftyMoveDraw());
  assert.equal(game.status().reason, 'fifty-move rule');
});

test('a capture or pawn move resets the halfmove clock', () => {
  const game = new Chess('4k3/8/8/8/8/8/P7/R3K3 w - - 40 60');
  game.move('a4');
  assert.equal(game.halfmoves, 0);
});

test('threefold repetition', () => {
  const game = new Chess('4k3/8/8/8/8/8/8/R3K2R w - - 0 1');
  // Shuffle the rooks back and forth to repeat the position three times.
  for (const m of ['Ra2', 'Ke7', 'Ra1', 'Ke8', 'Ra2', 'Ke7', 'Ra1', 'Ke8']) {
    assert.ok(game.move(m), m);
  }
  assert.ok(game.isThreefoldRepetition());
  assert.equal(game.status().reason, 'threefold repetition');
});

// ---------------------------------------------------------------------------
// SAN
// ---------------------------------------------------------------------------

test('SAN disambiguation by file, rank, and full square', () => {
  // Knights on b1 and f1 both reach d2 — file letter suffices.
  let game = new Chess('4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1');
  assert.ok(sans(game).includes('Nbd2'));
  assert.ok(sans(game).includes('Nfd2'));

  // Rooks on a1 and a3 both reach a2 — same file, so use the rank.
  game = new Chess('4k3/8/8/8/8/R7/8/R3K3 w - - 0 1');
  assert.ok(sans(game).includes('R1a2'));
  assert.ok(sans(game).includes('R3a2'));

  // Queens on a8, d8 and d1 all reach d5 — one case of each disambiguation
  // rule: file letter, rank digit, and the full square.
  game = new Chess('Q2Q4/8/7k/8/8/8/8/3QK3 w - - 0 1');
  const found = sans(game);
  assert.ok(found.includes('Qad5'), 'file letter is enough for the a8 queen');
  assert.ok(found.includes('Qd8d5'),
    'the d8 queen shares a file with d1 and a rank with a8, so it needs both');
  assert.ok(found.includes('Q1d5'), 'rank digit is enough for the d1 queen');
});

test('SAN includes check and mate markers', () => {
  const game = new Chess('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1');
  assert.ok(sans(game).includes('Ra8#'));

  const game2 = new Chess('4k3/8/8/8/8/8/8/R3K3 w - - 0 1');
  assert.ok(sans(game2).includes('Re1+') === false, 'sanity: no such move');
  assert.ok(sans(game2).includes('Ra8+'));
});

test('SAN and UCI parse back to the same move', () => {
  const game = new Chess();
  const line = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Ba4', 'Nf6', 'O-O', 'Be7'];
  for (const san of line) {
    const record = game.move(san);
    assert.ok(record, `${san} should be legal`);
    assert.equal(record.san, san);
  }
  assert.equal(
    game.fen(),
    'r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6',
  );
});

test('illegal and malformed input is rejected, not thrown', () => {
  const game = new Chess();
  assert.equal(game.move('e5'), null, 'not a legal first move');
  assert.equal(game.move('Ke2'), null, 'blocked by a pawn');
  assert.equal(game.move('garbage'), null);
  assert.equal(game.move('a1a1'), null);
  assert.equal(game.move({ from: 'e2', to: 'e9' }), null);
  assert.equal(game.fen(), START_FEN, 'the position is untouched');
});

// ---------------------------------------------------------------------------
// Undo
// ---------------------------------------------------------------------------

test('undo restores position, rights, and clocks exactly', () => {
  const game = new Chess();
  const seen = [game.fen()];
  for (const m of ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6', 'Nc3', 'a6']) {
    game.move(m);
    seen.push(game.fen());
  }
  for (let i = seen.length - 1; i > 0; i--) {
    assert.equal(game.fen(), seen[i]);
    game.undo();
  }
  assert.equal(game.fen(), START_FEN);
  assert.equal(game.undo(), null, 'undo past the start is a no-op');
});

test('undoing castling and en passant restores both pieces', () => {
  let game = new Chess('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1');
  const before = game.fen();
  game.move('O-O-O');
  game.undo();
  assert.equal(game.fen(), before);

  game = new Chess('rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3');
  const epBefore = game.fen();
  game.move('dxe3');
  game.undo();
  assert.equal(game.fen(), epBefore);
  assert.equal(game.get('e4').type, 1, 'the en-passant victim returned');
});

test('turn and move counters advance correctly', () => {
  const game = new Chess();
  assert.equal(game.turn, WHITE);
  game.move('e4');
  assert.equal(game.turn, BLACK);
  assert.equal(game.fullmoves, 1);
  game.move('e5');
  assert.equal(game.turn, WHITE);
  assert.equal(game.fullmoves, 2);
});
