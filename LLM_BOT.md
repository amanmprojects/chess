# chessLLM bot

This branch adds a neural opponent: a 94M-parameter character-level language model
trained from scratch on Lichess games in SAN notation.

## Running it

Two processes — the static UI, and the model.

```sh
node server.mjs                  # the UI, as before        -> localhost:8000
python llm_server.py             # the model sidecar        -> localhost:5555
```

The sidecar needs the `llm` repo checked out next to this one (`../llm`) with its
virtualenv built, since it imports the model definition and the SAN tokenizer from there:

```sh
../llm/.venv/bin/python llm_server.py
```

Then pick **🧠 chessLLM (neural)** from the Level dropdown.

## What makes this bot different

The classical bot in `src/ai.js` searches: it looks at the position, generates legal moves,
and evaluates them. It cannot play an illegal move because it only ever considers legal ones.

The neural bot does none of that. It has never seen a board — not a FEN, not a piece list.
It was trained purely on the *text* of chess games, and plays by predicting the characters
that come next. Everything it knows about chess it inferred from move notation alone.

Two things follow, and the side panel is built to show both:

- **It needs the whole move history, not the position.** A FEN means nothing to it. This is
  why the neural bot cannot be used from a pasted FEN or a reviewed position — there is no
  history to hand it.
- **It can propose moves that do not exist.** The sidecar retries up to 8 times and the
  engine remains the final arbiter of legality, so an illegal suggestion can never reach the
  board. The panel counts them rather than hiding them.

## The controls

| Control | What it does |
|---|---|
| **Temperature** | 0 always takes the model's single most likely move — deterministic, and it will replay the same game every time. Higher values sample more freely: more variety, more blunders. |
| **Rating prompt** | The model was trained with a rating tag on each game (`<15>`–`<24>`, i.e. 1500–2499). This asks it to imitate that level. It is a request, not a guarantee. |
| **Raw output** | The exact characters the model emitted for this move. |
| **Move played** | What was actually played after legality checking. |
| **Legal-move tries** | 1 means it proposed a legal move immediately. Higher means it had to be resampled. |
| **Illegal this game** | Cumulative illegal proposals, the honest measure of how well it is tracking the board. |

## Measured behaviour

From the evaluation harness in the `llm` repo, at temperature 0.7:

- **99.2%** of generated moves are legal, over 11,216 moves of self-play
- **55.6 plies** of average survival before a first illegal move
- **~1131 Elo** against Stockfish

Expect it to play reasonable opening book, drift in the middlegame, and lose the thread in
long or unusual positions — its board representation degrades as the game goes on.
