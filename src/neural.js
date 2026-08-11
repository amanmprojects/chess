/**
 * Neural level support — runs the trained ChessNet in the browser.
 *
 * The Vercel deployment is a static site, so there is no Python process to
 * host the model. Instead the exported weights (chess/model.bin, 11MB fp16)
 * are downloaded once by the game's Web Worker and the forward pass runs
 * locally in pure JS (see nn.js). Replies are shaped exactly like the
 * worker's `bestmove` / `error` messages so app.js can feed them through the
 * same handler, which already discards stale replies by id and knows how to
 * render hints and evaluation.
 */

import { loadModel, runNeuralMove } from './nn.js';

/** True when this level should be answered by the model instead of ai.js. */
export function isNeuralLevel(level) {
  return level === 'neural';
}

/**
 * Kick off the model download + decode in the background. Safe to call
 * repeatedly; the underlying fetch/parse promise is cached.
 *
 * `onStatus` receives {status:'downloading', loaded, total},
 * {status:'decoding'}, {status:'ready'}, or {status:'error', message}.
 */
export function preloadNeural(onStatus) {
  loadModel(undefined, onStatus).then(
    () => onStatus?.({ status: 'ready' }),
    (err) => onStatus?.({ status: 'error', message: String(err?.message ?? err) })
  );
}

/**
 * Ask the model for a move on a FEN position.
 *
 * Resolves to a `bestmove`-shaped result ({uci, value, cp, ms}) or throws.
 * The caller (worker.js or app.js's main-thread fallback) wraps it in the
 * message envelope; a thrown error leaves the "thinking" indicator handled
 * by the caller's error path.
 */
export async function requestNeuralMove(fen) {
  const started = Date.now();
  const result = await runNeuralMove(fen);
  return { ...result, ms: Date.now() - started };
}
