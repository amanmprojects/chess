/**
 * Client for the neural policy net (see serve_model.py in the parent project).
 *
 * The net is a 5.58M-parameter PyTorch model needing CUDA and a 67MB
 * checkpoint, so unlike ai.js it cannot run in a Web Worker. It runs in a
 * local Python process and we call it over HTTP.
 *
 * Replies are shaped exactly like the worker's `bestmove` / `error` messages so
 * app.js can feed them through the same handler, which already discards stale
 * replies by id and knows how to render hints and evaluation.
 */

const DEFAULT_ENDPOINT = 'http://127.0.0.1:8001';

/** Overridable so a non-default port does not require a code edit. */
export function endpoint() {
  const stored = globalThis.localStorage?.getItem('neuralEndpoint');
  return (stored && stored.trim()) || DEFAULT_ENDPOINT;
}

/**
 * Ask the model for a move.
 *
 * Resolves to a `bestmove`-shaped object, or an `error`-shaped one. It never
 * rejects: app.js routes both through onWorkerMessage, and a thrown error
 * there would leave the "thinking" indicator stuck on forever.
 */
export async function requestNeuralMove(id, fen, { temperature = 0, signal } = {}) {
  const started = Date.now();
  try {
    const response = await fetch(`${endpoint()}/move`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ fen, temperature }),
      signal,
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body?.error) detail = body.error;
      } catch {
        // Non-JSON error body; the status code is all we have.
      }
      return { type: 'error', id, message: `Neural server: ${detail}` };
    }

    const data = await response.json();
    if (!data.uci) {
      // Server says the game is already over — nothing to play.
      return { type: 'bestmove', id, uci: null, depth: 0, nodes: 1,
               score: { type: 'cp', value: 0 } };
    }

    return {
      type: 'bestmove',
      id,
      uci: data.uci,
      // The net does no search: one forward pass, one position looked at.
      // Reporting depth 1 / 1 node keeps the UI honest rather than borrowing
      // numbers that would imply lookahead it never did.
      depth: 1,
      nodes: 1,
      elapsed: Date.now() - started,
      score: { type: 'cp', value: data.cp ?? 0 },
    };
  } catch (error) {
    if (error?.name === 'AbortError') {
      return { type: 'error', id, message: 'Neural request cancelled' };
    }
    // Overwhelmingly the common case: the Python server is not running.
    return {
      type: 'error',
      id,
      message: `Cannot reach the neural server at ${endpoint()}. ` +
               'Start it with: python serve_model.py',
    };
  }
}

/** True when this level should be answered by the model instead of ai.js. */
export function isNeuralLevel(level) {
  return level === 'neural';
}
