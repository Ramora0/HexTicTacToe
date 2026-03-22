"""Generate training positions via epsilon-greedy self-play.

Plays games between two minimax bots. Each stone has an independent
probability (epsilon) of being replaced with a random candidate move.
Only decisive games (with a winner) are kept; draws are discarded.

Output format: list of (board, current_player, search_score, win_score, game_id)

Usage: python -m learned_eval.generate_positions [--epsilon 0.1] [--target 47000]
"""

import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import HexGame, Player
from ai import MinimaxBot, get_candidates

MAX_MOVES = 200
TARGET_POSITIONS = 47_000
SAVE_INTERVAL = 50
SCORE_SCALE = 20_000


def play_game_collect(args):
    """Play one epsilon-greedy game, return positions if decisive."""
    time_limit, epsilon, game_idx = args
    rng = random.Random(game_idx)
    game = HexGame(win_length=6)
    bot_a = MinimaxBot(time_limit=time_limit)
    bot_b = MinimaxBot(time_limit=time_limit)

    positions = []
    total_stones = 0

    while not game.game_over and total_stones < MAX_MOVES:
        board_snap = dict(game.board)
        cp = game.current_player

        bot = bot_a if cp == Player.A else bot_b
        bot_moves = bot.get_move(game)

        if board_snap:
            search_score = bot.last_score / SCORE_SCALE
            positions.append((board_snap, cp, search_score))

        # Epsilon-greedy: independently randomize each stone
        moves = list(bot_moves)
        for i in range(len(moves)):
            if rng.random() < epsilon:
                candidates = get_candidates(game)
                alt = [c for c in candidates if c not in moves]
                if alt:
                    moves[i] = rng.choice(alt)

        for q, r in moves:
            if game.game_over:
                break
            if not game.make_move(q, r):
                break
            total_stones += 1

    winner = game.winner

    # Discard draws
    if winner == Player.NONE:
        return None, total_stones

    # Tag each position with win_score from current_player's perspective
    tagged = []
    for board_snap, cp, eval_score in positions:
        win_score = 1.0 if winner == cp else -1.0
        tagged.append((board_snap, cp, eval_score, win_score, game_idx))

    return tagged, total_stones


def _save(all_positions, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(all_positions, f)
    os.replace(tmp, path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--target", type=int, default=TARGET_POSITIONS)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"))
    args = parser.parse_args()

    workers = os.cpu_count() or 1

    all_positions = []
    games_played = 0
    games_decisive = 0
    games_drawn = 0
    game_idx = int(time.time() * 1000) % 1_000_000
    games_since_save = 0
    total_moves = 0

    print(f"Generating positions: epsilon={args.epsilon}, target={args.target}, "
          f"workers={workers}, time_limit={args.time_limit}")

    try:
        with Pool(workers) as pool:
            pending = []

            def _submit():
                nonlocal game_idx
                ar = pool.apply_async(play_game_collect,
                                      ((args.time_limit, args.epsilon, game_idx),))
                pending.append(ar)
                game_idx += 1

            for _ in range(workers * 2):
                _submit()

            while len(all_positions) < args.target:
                ar = pending.pop(0)
                game_positions, move_count = ar.get()

                games_played += 1
                games_since_save += 1
                total_moves += move_count

                if game_positions is not None:
                    games_decisive += 1
                    all_positions.extend(game_positions)
                else:
                    games_drawn += 1

                if games_played % 100 == 0:
                    pct_decisive = games_decisive / games_played * 100 if games_played else 0
                    print(f"  {len(all_positions):,}/{args.target:,} positions | "
                          f"{games_played} games ({pct_decisive:.0f}% decisive)")

                _submit()

                if games_since_save >= SAVE_INTERVAL:
                    _save(all_positions, args.output)
                    games_since_save = 0

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving {len(all_positions)} positions...")

    _save(all_positions, args.output)

    evals = [p[2] for p in all_positions]
    wins = [p[3] for p in all_positions]
    avg_moves = total_moves / games_played if games_played else 0
    win_count = sum(1 for w in wins if w > 0)
    loss_count = sum(1 for w in wins if w < 0)
    print(f"\nCollected {len(all_positions):,} positions from {games_decisive} decisive games "
          f"({games_drawn} draws discarded)")
    print(f"  Avg moves/game: {avg_moves:.1f}")
    print(f"  Eval range: [{min(evals):.2f}, {max(evals):.2f}]")
    print(f"  Eval mean: {sum(evals)/len(evals):.4f}")
    print(f"  Win/Loss: {win_count}/{loss_count}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
