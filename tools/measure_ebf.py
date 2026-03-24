"""Find the pruning sweet spot: what MAX_WIDTH consistently reaches the next depth?"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ai
from game import HexGame, Player

for width in [999, 60, 45, 35, 30, 25, 20]:
    ai._MAX_WIDTH = width
    bot = ai.MinimaxBot(time_limit=0.1)
    game = HexGame(win_length=6)

    depths = []
    for _ in range(30):
        if game.game_over:
            break
        q, r = bot.get_move(game)
        game.make_move(q, r)
        depths.append(bot.last_depth)

    avg_d = sum(depths) / len(depths) if depths else 0
    d_counts = {}
    for d in depths:
        d_counts[d] = d_counts.get(d, 0) + 1
    dist = "  ".join(f"d{d}:{c}" for d, c in sorted(d_counts.items()))
    print(f"width={width:3d}  avg_depth={avg_d:.1f}  {dist}")
