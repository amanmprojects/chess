/**
 * ChessNet inference in pure JS — a faithful port of model.py.
 *
 * The trained net cannot run on a server here: the Vercel deployment is a
 * static site, so the weights (chess/model.bin, float16) are downloaded by the
 * browser and the forward pass runs locally, in the game's Web Worker. No
 * dependencies, no WebGL — just typed arrays and loops, which is plenty for
 * a 5.58M-param transformer at 67 tokens.
 *
 *   - createInference(manifest, bin)  build a runnable model from the exported
 *                                     files (model.json + model.bin)
 *   - loadModel()                     fetch + cache the exported files
 *   - pickNeuralMove(game, model)     features -> masked policy -> best move,
 *                                     shaped like serve_model.py's reply
 *
 * The numerics mirror serve_model.py exactly: GELU uses the erf form, LayerNorm
 * eps 1e-5, the value head is mean-pool -> LayerNorm -> Linear -> tanh, and the
 * value is converted to centipawns from White's point of view with the same
 * logistic mapping (including the sign flip when Black is to move).
 */

import {
  Chess, WHITE, BLACK, KNIGHT, BISHOP, ROOK, QUEEN,
  fileOf, rankOf, moveFrom, moveTo, movePromo, moveToUci, from64, to64,
} from './engine.js';

export const POLICY_SIZE = 64 * 73;
const EP_NONE = 15;

// 73 labels per square: 56 queen slides (8 dirs x 7 dists), 8 knight jumps,
// 9 underpromotions. Order matches features.py / train.py exactly.
const DIRS = [
  [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], // N NE E SE S SW W NW
];
const KNIGHT_DELTAS = [
  [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2],
];
const PROMO_ID = { [KNIGHT]: 0, [BISHOP]: 1, [ROOK]: 2 };

/** Port of features.py move_to_slot: map an engine move to its 0..4671 slot. */
export function moveToSlot(move) {
  const from = moveFrom(move);
  const df = fileOf(moveTo(move)) - fileOf(from);
  const dr = rankOf(moveTo(move)) - rankOf(from);
  const promo = movePromo(move);
  const fromIdx = to64(from);

  if (promo && promo !== QUEEN) {
    // Underpromotions: 3 destination files x 3 pieces (N=0, B=1, R=2).
    return fromIdx * 73 + 64 + (df + 1) * 3 + PROMO_ID[promo];
  }
  if ((Math.abs(df) === 1 && Math.abs(dr) === 2) || (Math.abs(df) === 2 && Math.abs(dr) === 1)) {
    const idx = KNIGHT_DELTAS.findIndex(([x, y]) => x === df && y === dr);
    if (idx < 0) throw new Error(`unmapped knight move ${moveToUci(move)}`);
    return fromIdx * 73 + 56 + idx;
  }
  // Queen-style slides (covers pawn pushes, king moves, castling, en passant,
  // and queen promotions).
  const dir = DIRS.findIndex(([x, y]) => x === Math.sign(df) && y === Math.sign(dr));
  if (dir < 0 || (df === 0 && dr === 0)) throw new Error(`unmapped move ${moveToUci(move)}`);
  const dist = Math.max(Math.abs(df), Math.abs(dr));
  return fromIdx * 73 + dir * 7 + dist - 1;
}

/** Port of features.py board_to_features, read straight off the engine board. */
export function boardToFeatures(game) {
  const pieces = new Int32Array(64);
  for (let i = 0; i < 64; i++) {
    const p = game.board[from64(i)];
    if (p !== 0) pieces[i] = (p >> 3 ? 6 : 0) + (p & 7); // white 1-6, black 7-12
  }
  const aux = new Int32Array(2);
  // bit0 = side (1 white), bits 1..4 = castling KQkq — engine's castling mask
  // is already those four bits, just shifted down one.
  aux[0] = (game.turn === WHITE ? 1 : 0) | ((game.castling & 0xF) << 1);
  aux[1] = game.ep >= 0 ? fileOf(game.ep) : EP_NONE;
  return { pieces, aux };
}

/** Port of features.py policy_mask: boolean mask over all legal moves' slots. */
function policyMask(game) {
  const mask = new Uint8Array(POLICY_SIZE);
  const moves = game.generateMoves();
  for (const m of moves) mask[moveToSlot(m)] = 1;
  return { mask, moves };
}

/** serve_model.py's value -> centipawns conversion (White's point of view). */
export function valueToCp(value, turn) {
  const v = Math.max(Math.min(value, 0.999), -0.999);
  let cp = Math.trunc(-400 * Math.log10(2 / (v + 1) - 1));
  if (turn === BLACK) cp = -cp;
  return cp;
}

/**
 * Pick the model's move for an engine Chess position.
 *
 * @returns {{uci: string, value: number, cp: number}}
 */
export function pickNeuralMove(game, model, { temperature = 0 } = {}) {
  const { pieces, aux } = boardToFeatures(game);
  const { logits, value } = model.forward(pieces, aux);
  const { mask, moves } = policyMask(game);

  // No legal moves: game over. The old serve_model.py answered with
  // uci: null and the UI treats that as "nothing to play".
  if (moves.length === 0) return { uci: null, value, cp: valueToCp(value, game.turn) };

  let slot;
  if (temperature <= 0) {
    // Argmax over the masked logits — identical to pick_move at temp 0.
    let best = -Infinity;
    slot = -1;
    for (let s = 0; s < POLICY_SIZE; s++) {
      if (!mask[s]) continue;
      const l = logits[s];
      if (l > best) { best = l; slot = s; }
    }
  } else {
    // Softmax over masked logits, then sample (as in play.py).
    let max = -Infinity;
    for (let s = 0; s < POLICY_SIZE; s++) if (mask[s] && logits[s] > max) max = logits[s];
    const probs = new Float64Array(POLICY_SIZE);
    let sum = 0;
    for (let s = 0; s < POLICY_SIZE; s++) {
      if (!mask[s]) continue;
      probs[s] = Math.exp((logits[s] - max) / temperature);
      sum += probs[s];
    }
    let r = Math.random() * sum;
    slot = -1;
    for (let s = 0; s < POLICY_SIZE; s++) {
      if (!mask[s]) continue;
      r -= probs[s];
      if (r <= 0) { slot = s; break; }
    }
  }

  const move = moves.find((m) => moveToSlot(m) === slot);
  if (!move) throw new Error(`slot ${slot} not found among legal moves`);
  return { uci: moveToUci(move), value, cp: valueToCp(value, game.turn) };
}

// ---------------------------------------------------------------------------
// Model loading and the forward pass
// ---------------------------------------------------------------------------

function decodeF16(dataView, byteOffset, count) {
  // Little-endian IEEE 754 half -> float32. The exporter wrote weights in
  // this order, so a simple linear decode is all it takes.
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const bits = dataView.getUint16(byteOffset + i * 2, true);
    const sign = (bits & 0x8000) ? -1 : 1;
    const exp = (bits >> 10) & 0x1f;
    const frac = bits & 0x3ff;
    if (exp === 0) out[i] = sign * frac * 2 ** -24;              // subnormal or zero
    else if (exp === 31) out[i] = sign * (frac ? NaN : Infinity); // inf/nan
    else out[i] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return out;
}

/**
 * Build a runnable model from the exported files.
 *
 * @param {object} manifest  parsed model.json
 * @param {ArrayBuffer} bin  contents of model.bin (concatenated fp16 tensors)
 */
export function createInference(manifest, bin) {
  const d = manifest.d;
  const heads = manifest.n_heads;
  const headDim = d / heads;
  const view = new DataView(bin);

  const tensors = new Map();
  for (const t of manifest.tensors) {
    tensors.set(t.name, decodeF16(view, t.offset * 2, t.len));
  }
  const T = (name) => tensors.get(name);

  /** LayerNorm with eps 1e-5, matching torch.nn.LayerNorm defaults. */
  function layerNorm(x, w, b, rows = 1) {
    for (let r = 0; r < rows; r++) {
      const off = r * d;
      let mean = 0;
      for (let i = 0; i < d; i++) mean += x[off + i];
      mean /= d;
      let varSum = 0;
      for (let i = 0; i < d; i++) { const q = x[off + i] - mean; varSum += q * q; }
      const inv = 1 / Math.sqrt(varSum / d + 1e-5);
      for (let i = 0; i < d; i++) {
        x[off + i] = (x[off + i] - mean) * inv * w[i] + b[i];
      }
    }
    return x;
  }

  /** torch's erf-based GELU: 0.5x(1+erf(x/sqrt2)). (Abramowitz-Stegun 7.1.26, err < 1.5e-7.) */
  function gelu(x) {
    const a = x / Math.SQRT2;
    const ax = Math.abs(a);
    const t = 1 / (1 + 0.3275911 * ax);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
      * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
    const erf = a < 0 ? -y : y;
    return 0.5 * x * (1 + erf);
  }

  /** out[r,c] = bias[c] + sum_i X[r,i] * W[c,i]  (W is (cols, inner), row-major). */
  function matmul(X, W, B, rows, cols, inner) {
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const xr = r * inner;
      const orow = r * cols;
      for (let c = 0; c < cols; c++) {
        const wc = c * inner;
        let s = B[c];
        for (let i = 0; i < inner; i++) s += X[xr + i] * W[wc + i];
        out[orow + c] = s;
      }
    }
    return out;
  }

  /** One prenorm block: LN -> MHA -> +, LN -> MLP -> +. */
  function block(x, b) {
    const p = `blocks.${b}.`;
    const y = layerNorm(Float32Array.from(x), T(p + 'ln1.weight'), T(p + 'ln1.bias'), 67);

    // QKV projections (nn.MultiheadAttention: in_proj = [Wq; Wk; Wv] rows,
    // so qkv is (67, 3d) and the q/k/v sections are column slices).
    const qkv = matmul(y, T(p + 'attn.in_proj_weight'), T(p + 'attn.in_proj_bias'),
                       67, 3 * d, d);

    // Scaled dot-product attention over 67 tokens, 8 heads of 32 dims.
    const scores = new Float32Array(67 * 67);
    const attn = new Float32Array(67 * d);
    for (let h = 0; h < heads; h++) {
      const ho = h * headDim;
      for (let i = 0; i < 67; i++) {
        const qi = i * 3 * d + ho;
        const si = i * 67;
        for (let j = 0; j < 67; j++) {
          const kj = j * 3 * d + d + ho;
          let s = 0;
          for (let t = 0; t < headDim; t++) s += qkv[qi + t] * qkv[kj + t];
          scores[si + j] = s / Math.sqrt(headDim);
        }
      }
      for (let i = 0; i < 67; i++) {
        // Softmax over the key dimension, then weighted sum of values.
        const si = i * 67;
        let max = -Infinity;
        for (let j = 0; j < 67; j++) if (scores[si + j] > max) max = scores[si + j];
        let sum = 0;
        for (let j = 0; j < 67; j++) { scores[si + j] = Math.exp(scores[si + j] - max); sum += scores[si + j]; }
        const oi = i * d + ho;
        for (let t = 0; t < headDim; t++) {
          let s = 0;
          for (let j = 0; j < 67; j++) s += scores[si + j] * qkv[j * 3 * d + 2 * d + ho + t];
          attn[oi + t] = s / sum;
        }
      }
    }

    // Output projection + residual.
    const z = matmul(attn, T(p + 'attn.out_proj.weight'), T(p + 'attn.out_proj.bias'),
                     67, d, d);
    for (let i = 0; i < 67 * d; i++) x[i] += z[i];

    // MLP: Linear(d,4d) -> GELU -> Linear(4d,d), prenorm residual.
    const n = layerNorm(Float32Array.from(x), T(p + 'ln2.weight'), T(p + 'ln2.bias'), 67);
    const h = matmul(n, T(p + 'mlp.0.weight'), T(p + 'mlp.0.bias'), 67, 4 * d, d);
    for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
    const m = matmul(h, T(p + 'mlp.2.weight'), T(p + 'mlp.2.bias'), 67, d, 4 * d);
    for (let i = 0; i < 67 * d; i++) x[i] += m[i];
  }

  /**
   * Forward pass. Mirrors model.py forward():
   *   x = piece_emb(pieces) + pos[:64], concat 3 aux tokens + pos[64:],
   *   7 prenorm blocks, policy head Linear(256 -> 73) per square,
   *   value = tanh(LayerNorm -> Linear over mean-pooled squares).
   */
  function forward(pieces, aux) {
    const x = new Float32Array(67 * d);
    const pieceEmb = T('piece_emb.weight');
    for (let i = 0; i < 64; i++) {
      const row = pieces[i] * d;
      const dst = i * d;
      for (let t = 0; t < d; t++) x[dst + t] = pieceEmb[row + t];
    }
    // Aux tokens: [side, castle, ep] — same order as model.py.
    const side = aux[0] & 1;
    const castle = (aux[0] >> 1) & 0xF;
    const ep = aux[1];
    const auxRows = [T('side_emb.weight').subarray(side * d, (side + 1) * d),
                     T('castle_emb.weight').subarray(castle * d, (castle + 1) * d),
                     T('ep_emb.weight').subarray(ep * d, (ep + 1) * d)];
    for (let a = 0; a < 3; a++) {
      const dst = (64 + a) * d;
      for (let t = 0; t < d; t++) x[dst + t] = auxRows[a][t];
    }
    const pos = T('pos');
    for (let i = 0; i < 67 * d; i++) x[i] += pos[i];

    if (model.trace) model.trace.push({ stage: 'embed', x: Float32Array.from(x) });

    for (let b = 0; b < manifest.n_layers; b++) {
      block(x, b);
      if (model.trace) model.trace.push({ stage: 'block' + b, x: Float32Array.from(x) });
    }

    if (model.trace) model.trace.push({ stage: 'final', x: Float32Array.from(x) });

    // Policy head: shared Linear(d -> 73) per square, flattened to 4672.
    const pw = T('policy_head.weight');
    const pb = T('policy_head.bias');
    const logits = new Float32Array(POLICY_SIZE);
    for (let sq = 0; sq < 64; sq++) {
      const xr = sq * d;
      const orow = sq * 73;
      for (let c = 0; c < 73; c++) {
        const wc = c * d;
        let s = pb[c];
        for (let t = 0; t < d; t++) s += x[xr + t] * pw[wc + t];
        logits[orow + c] = s;
      }
    }

    // Value head: mean-pool squares -> LayerNorm -> Linear -> tanh.
    const vw = T('value_head.1.weight');
    const vb = T('value_head.1.bias');
    const mean = new Float32Array(d);
    for (let t = 0; t < d; t++) {
      let s = 0;
      for (let sq = 0; sq < 64; sq++) s += x[sq * d + t];
      mean[t] = s / 64;
    }
    layerNorm(mean, T('value_head.0.weight'), T('value_head.0.bias'));
    let v = vb[0];
    for (let t = 0; t < d; t++) v += vw[t] * mean[t];
    v = Math.tanh(v);

    return { logits, value: v };
  }

  const model = { forward, manifest, trace: null };
  return model;
}

let modelPromise = null;

/**
 * Fetch (once) and build the model. Resolves to the createInference result.
 * The caller decides where the files live; `base` defaults to the directory
 * of this module (so a Worker loading ./nn.js finds ../model.bin next to the
 * static site root).
 *
 * `onProgress` is called with {status:'downloading', loaded, total} while the
 * weights stream in and {status:'decoding'} before the fp16 decode.
 */
export function loadModel(base = new URL('..', import.meta.url), onProgress) {
  if (!modelPromise) {
    modelPromise = (async () => {
      const jsonRes = await fetch(new URL('model.json', base));
      if (!jsonRes.ok) throw new Error(`model.json: HTTP ${jsonRes.status}`);
      const manifest = await jsonRes.json();

      const binRes = await fetch(new URL('model.bin', base));
      if (!binRes.ok) throw new Error(`model.bin: HTTP ${binRes.status}`);
      const total = Number(binRes.headers.get('content-length')) || 0;

      // Stream the weights so the UI can show real download progress.
      const reader = binRes.body.getReader();
      const chunks = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onProgress?.({ status: 'downloading', loaded, total });
      }
      const bin = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) { bin.set(c, off); off += c.byteLength; }

      onProgress?.({ status: 'decoding' });
      return createInference(manifest, bin.buffer);
    })().catch((err) => {
      modelPromise = null; // a failed load can be retried
      throw err;
    });
  }
  return modelPromise;
}

/** Convience: model move for a FEN string (used by worker + main-thread fallback). */
export async function runNeuralMove(fen) {
  const model = await loadModel();
  const game = new Chess(fen);
  return pickNeuralMove(game, model);
}

