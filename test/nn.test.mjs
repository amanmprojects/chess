/**
 * Validates the browser inference port (src/nn.js) against the real torch
 * model. Ground truth lives in golden.json, produced by make_golden.py (see
 * the repo root). Skips itself when the model artifacts are not committed.
 *
 *   node --test test/nn.test.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { Chess, moveToUci } from '../src/engine.js';
import { createInference, moveToSlot, pickNeuralMove } from '../src/nn.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const available = existsSync(join(root, 'model.json'))
  && existsSync(join(root, 'model.bin'))
  && existsSync(join(here, 'golden.json'));

let model = null;
let golden = null;
if (available) {
  const manifest = JSON.parse(readFileSync(join(root, 'model.json'), 'utf8'));
  const bin = readFileSync(join(root, 'model.bin'));
  model = createInference(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  golden = JSON.parse(readFileSync(join(here, 'golden.json'), 'utf8'));
}

test('move-to-slot parity with the training encoding', (t) => {
  if (!available) return t.skip('model artifacts missing — run make_golden.py');
  assert.ok(golden.slots.length > 0);

  for (const { fen, moves } of golden.slots) {
    const game = new Chess(fen);
    const jsMoves = game.generateMoves();
    const js = new Map(jsMoves.map((m) => [moveToUci(m), moveToSlot(m)]));

    // Same legal move set (by UCI) as python-chess.
    const uciSet = moves.map(([uci]) => uci).sort();
    assert.deepEqual([...js.keys()].sort(), uciSet, `move set differs for ${fen}`);

    // And every move maps to the same 64x73 slot.
    for (const [uci, slot] of moves) {
      assert.equal(js.get(uci), slot, `slot mismatch for ${uci} in ${fen}`);
    }
  }
});

test('forward pass matches the torch checkpoint', (t) => {
  if (!available) return t.skip('model artifacts missing — run make_golden.py');
  assert.ok(golden.inference.length > 0);

  for (const g of golden.inference) {
    const game = new Chess(g.fen);
    const result = pickNeuralMove(game, model);

    // The chosen move must agree exactly; fp16 weights and fp64 accumulation
    // are only allowed to nudge a near-tie, which the golden set excludes.
    assert.equal(result.uci, g.uci, `move mismatch in ${g.fen}`);
    assert.ok(Math.abs(result.value - g.value) < 0.02,
      `value mismatch in ${g.fen}: ${result.value} vs ${g.value}`);
    // Centipawns are truncated, so a rounding difference of 1 is tolerable.
    assert.ok(Math.abs(result.cp - g.cp) <= 1,
      `cp mismatch in ${g.fen}: ${result.cp} vs ${g.cp}`);
  }
});
