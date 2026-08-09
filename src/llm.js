/**
 * Client for the chessLLM inference sidecar (llm_server.py).
 *
 * The neural bot is a character-level language model trained only on SAN move text. It has
 * never seen a board, a FEN, or a legality check -- it predicts the next characters of a
 * chess game the way a text model predicts the next word. Two consequences shape this file:
 *
 *   1. It needs the move history, not the position. A FEN tells it nothing.
 *   2. It can propose illegal moves. The server retries; if every retry fails we surface
 *      that rather than hiding it, because how often it happens is the interesting part.
 */

const ENDPOINT = 'http://127.0.0.1:5555';

/**
 * Ask the model for a move. `sanHistory` must be every move of the game so far, in order.
 * Resolves to {san, rawOutput, attempts} or {error, rawOutput, attempts}.
 */
export async function requestLlmMove(sanHistory, { bucket = 20, temp = 0.7, signal } = {}) {
  const response = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ san_history: sanHistory, bucket, temp }),
    signal,
  });
  if (!response.ok) throw new Error(`inference server returned ${response.status}`);
  const data = await response.json();
  return {
    san: data.san ?? null,
    rawOutput: data.raw_output ?? '',
    attempts: data.attempts ?? 0,
    error: data.error ?? null,
  };
}

/** True when the sidecar is reachable, so the UI can explain itself when it is not. */
export async function llmServerAvailable() {
  try {
    const probe = await requestLlmMove([], { temp: 0.7 });
    return Boolean(probe.san) || Boolean(probe.error);
  } catch {
    return false;
  }
}
