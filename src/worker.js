/**
 * The AI runs here so that thinking never blocks the board.
 */
import { Chess, moveToUci } from './engine.js';
import { Search, chooseMove, MATE, MATE_THRESHOLD } from './ai.js';

const search = new Search({ ttSizeMb: 48 });

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

self.onmessage = (event) => {
  const { type, id, fen, level, history } = event.data;
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
