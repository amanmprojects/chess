#!/usr/bin/env python3
"""
Inference sidecar for chessLLM. Listens on localhost:5555 and responds to JSON-RPC-like
requests: {"id": <int>, "san_history": ["e4", "e5", ...], "bucket": <int>, "temp": <float>}.

Returns {"id": <int>, "san": <str>, "raw_output": <str>, "attempts": <int>}
on success, or {"id": <int>, "error": <str>} on failure.

The model ONLY sees move history (no FEN), so the SAN list must be complete from move 1.
"""
import sys
import json
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# The chess/ module lives in the parent llm/ repo one level up, and has no __init__.py
# so `from chess import X` would resolve to the installed python-chess package.
sys.path.insert(0, "../llm/chess")
import chess_format as cf

# Same venv trick: torch lives in ../llm/.venv, not system python.
sys.path.insert(0, "../llm")
from model import MiniLLM, ModelConfig
import torch

CKPT_PATH = "../llm/data/chess_out/ckpt.pt"
DEVICE = "cpu"  # keep the GPU free for training; inference is fast enough on CPU


def load_model():
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ck["cfg"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = MiniLLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    print(f"[loaded] step {ck['step']}/{ck['total_steps']}  val {ck['best_val']:.4f}  "
          f"{model.num_params() / 1e6:.1f}M params  vocab={cfg.vocab_size}", flush=True)
    return model, cfg


MODEL, CFG = load_model()


def sample_one_move(history, bucket, temperature, max_chars=12):
    """Sample one full SAN move, with retries on illegal moves (up to 8 attempts).

    Returns (san, raw_output, attempts) on success, or (None, None, attempts) on exhaustion.
    `raw_output` is everything the model emitted for the move that succeeded (including the
    space that terminated it), so the UI can show "model said Nf6 , selected Nf6".
    """
    # Replay the game so far to build the context the model sees.
    text = f"<{bucket}>"
    if history:
        text += " " + " ".join(history)
    ids = [cf.GAME_START_ID] + cf.tokenize(text) + [cf.SPACE_ID]
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    import chess as pychess
    board = pychess.Board()
    for san in history:
        board.push_san(san)

    # Temperature 0 is deterministic, so a retry would resample the identical illegal move
    # forever. Nudge upward on retry so the retry budget can actually do something.
    for attempt in range(1, 9):
        temp = temperature if temperature > 0 else 0.0
        if attempt > 1 and temp <= 0.05:
            temp = 0.2
        cur = []
        step_idx = idx
        for _ in range(max_chars):
            with torch.no_grad():
                logits, _ = MODEL(step_idx[:, -CFG.seq_len:])
                logits = logits[:, -1, :].float()
            if temp <= 0:
                nxt = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits / temp, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            if nxt == cf.GAME_END_ID:
                # The model frequently writes a COMPLETE move (sometimes with a trailing
                # check/mate marker) and then emits the end token instead of the space that
                # would terminate it -- it is signing off because it believes the game is
                # over (e.g. it just found mate). If what it wrote parses as a legal move,
                # play it; the end token is then just the model saying "and that's mate".
                # Only an empty context is a genuine resignation.
                san = "".join(cur)
                if san:
                    try:
                        board.parse_san(san)
                        return san, san, attempt
                    except ValueError:
                        pass
                return None, san or "<end>", attempt
            if nxt == cf.SPACE_ID:
                san = "".join(cur)
                if not san:
                    continue
                try:
                    board.parse_san(san)
                except ValueError:
                    break  # illegal -- burn this attempt and resample from the same context
                return san, san, attempt
            cur.append(cf.decode([nxt]))
            step_idx = torch.cat(
                [step_idx, torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)], dim=1)
        # fell out of the char loop: truncated or illegal, try again

    return None, "".join(cur), 8


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        # The UI is served from a different port, so the browser needs CORS to call this.
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_POST(self):
        try:
            n = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(n))
            history = req.get("san_history", [])
            bucket = int(req.get("bucket", 20))
            temp = float(req.get("temp", 0.7))
            rid = req.get("id")

            san, raw, attempts = sample_one_move(history, bucket, temp)
            if san is None:
                self._send(200, {"id": rid, "error": "no legal move after 8 attempts",
                                 "raw_output": raw, "attempts": attempts})
                return
            self._send(200, {"id": rid, "san": san, "raw_output": raw, "attempts": attempts})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass  # the default logger writes a line per request, which drowns the load message


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    # Bound to localhost only: this exposes model inference with no authentication, so it
    # must not be reachable from the network.
    print(f"chessLLM inference server on http://127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
