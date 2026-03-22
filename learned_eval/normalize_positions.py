"""Normalize existing positions.pkl: divide eval scores by 20k, win scores by 1000."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIVISOR = 20_000
WIN_DIVISOR = 1000

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "positions.pkl")

with open(path, "rb") as f:
    positions = pickle.load(f)

print(f"Loaded {len(positions)} positions from {path}")
print(f"Tuple length: {len(positions[0])}")

normalized = []
for p in positions:
    if len(p) == 5:
        board, cp, eval_score, win_score, game_id = p
        normalized.append((board, cp, eval_score / EVAL_DIVISOR, win_score / WIN_DIVISOR, game_id))
    else:
        board, cp, eval_score, game_id = p
        normalized.append((board, cp, eval_score / EVAL_DIVISOR, game_id))

# Verify
import numpy as np
scores = np.array([p[2] for p in normalized])
normal = scores[np.abs(scores) < 4999]
forced = len(scores) - len(normal)
print(f"After normalization: {len(scores)} positions ({forced} forced wins)")
print(f"Non-forced: mean={normal.mean():.3f}, std={normal.std():.3f}, "
      f"range=[{normal.min():.2f}, {normal.max():.2f}]")

tmp = path + ".tmp"
with open(tmp, "wb") as f:
    pickle.dump(normalized, f)
os.replace(tmp, path)
print(f"Saved to {path}")
