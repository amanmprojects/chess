/**
 * Perft from the command line — the quickest way to check that a change to
 * move generation did not break anything.
 *
 *   node test/perft-cli.mjs [depth] [fen]
 *   node test/perft-cli.mjs 5
 *   node test/perft-cli.mjs 4 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
 *
 * With `--divide`, prints the node count under each root move, which is how
 * you bisect a mismatch against another engine.
 */

import { Chess, START_FEN, moveToUci } from '../src/engine.js';

const args = process.argv.slice(2);
const divide = args.includes('--divide');
const rest = args.filter((a) => a !== '--divide');

const depth = Number(rest[0] ?? 5);
const fen = rest[1] ?? START_FEN;

if (!Number.isInteger(depth) || depth < 0) {
  console.error(`Not a valid depth: ${rest[0]}`);
  process.exit(1);
}

let game;
try {
  game = new Chess(fen);
} catch (error) {
  console.error(String(error?.message ?? error));
  process.exit(1);
}

console.log(fen);
console.log(`depth ${depth}\n`);

const started = process.hrtime.bigint();

let total;
if (divide) {
  total = 0;
  for (const move of game.generateMoves()) {
    game.makeMove(move);
    const nodes = depth <= 1 ? 1 : game.perft(depth - 1);
    game.undoMove();
    console.log(`${moveToUci(move)}: ${nodes}`);
    total += nodes;
  }
  console.log('');
} else {
  total = game.perft(depth);
}

const ms = Number(process.hrtime.bigint() - started) / 1e6;
const nps = ms > 0 ? Math.round(total / (ms / 1000)) : 0;

console.log(`nodes ${total.toLocaleString('en-US')}`);
console.log(`time  ${ms.toFixed(0)} ms`);
console.log(`speed ${nps.toLocaleString('en-US')} nodes/s`);
