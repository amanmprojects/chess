/**
 * The AI runs here so that thinking never blocks the board.
 *
 * Two kinds of requests:
 *   - {type:'search', ...}  alpha-beta search (ai.js)
 *   - {type:'neural', ...}  the trained policy net, run in-process via nn.js
 *                           (weights ship with the static site — no server)
 */
import { Chess, moveToUci } from './engine.js';
import { Search, chooseMove, MATE, MATE_THRESHOLD } from './ai.js';
import { preloadNeural, requestNeuralMove } from './neural.js';

const search = new Search({ ttSizeMb: 48 });

/** Report neural model download/load progress to the app for its overlay. */
function reportModelStatus(status) {
  self.postMessage({ type: 'model-status', ...status });
}

/** Turn a raw centipawn score into something a person can read. */
function describeScore(score, turn) {
  if (Math.abs(score) > MATE_THRESHOLD) {
    const plies = MATE - Math.abs(score);
    return {
      type: 'mate',
      moves: Math.ceil(plies / 2),
      winning: score > 0 ? turn : turn ^ 1,
    };
  }
  return { type: 'cp', value: score };
}

/** One forward pass of the neural net. Shaped like the worker's search reply. */
async function handleNeural({ id, fen }) {
  try {
    const started = Date.now();
    const result = await requestNeuralMove(fen);
    self.postMessage({
      type: 'bestmove',
      id,
      source: 'neural-net',
      uci: result.uci,
      // The net does no search: one forward pass, one position looked at.
      depth: 1,
      nodes: 1,
      elapsed: result.ms ?? Date.now() - started,
      score: { type: 'cp', value: result.cp },
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      id,
      message: `Neural net: ${error?.message ?? error}`,
    });
  }
}

self.onmessage = (event) => {
  const { type, id, fen, level, history } = event.data;
  if (type === 'neural-preload') {
    // Fetch the neural weights in the background; progress streams back as
    // 'model-status' messages. Failures reset the cache, so a retry refetches.
    preloadNeural(reportModelStatus);
    return;
  }
  if (type === 'neural') {
    handleNeural({ id, fen });
    return;
  }
  if (type !== 'search') return;

  try {
    const game = new Chess(fen);
    // The FEN alone cannot express repetitions, so carry over the position keys
    // of the game so far. The search reads these at its root.
    if (Array.isArray(history) && history.length > 0) game.positions = history;

    const turn = game.turn;
    const started = Date.now();

    const result = chooseMove(game, level, {
      search,
      onIteration: (iteration) => {
        self.postMessage({
          type: 'progress',
          id,
          depth: iteration.depth,
          nodes: iteration.nodes,
          score: describeScore(iteration.score, turn),
        });
      },
    });

    self.postMessage({
      type: 'bestmove',
      id,
      source: 'worker-search',
      move: result.move,
      uci: result.move ? moveToUci(result.move) : null,
      depth: result.depth,
      nodes: result.nodes,
      elapsed: Date.now() - started,
      score: describeScore(result.score, turn),
    });
  } catch (error) {
    self.postMessage({ type: 'error', id, message: String(error?.message ?? error) });
  }
};
