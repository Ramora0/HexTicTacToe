"""Solve a saved position with unlimited search time.

Loads a position file saved by play.py and runs ai_cpp depth-by-depth,
printing live progress after each iteration.

Usage:
    python solve.py positions/position_20260324_150000.pkl
    python solve.py positions/position_20260324_150000.pkl --max-depth 50
    python solve.py positions/position_20260324_150000.pkl --time-limit 120
"""

import argparse
import importlib
import pickle
import time

from game import HexGame, Player

WIN_SCORE = 100_000_000


def load_position(path):
    """Load a position file. Supports both new dict format and legacy tuple format."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return data["board"], data["current_player"]

    # Legacy format: list of (board, player, score, uid) tuples — take first
    if isinstance(data, list):
        entry = data[0]
        board, player = entry[0], entry[1]
        return board, player

    raise ValueError(f"Unknown position format in {path}")


def format_moves(result, pair_moves):
    moves = [tuple(m) for m in result] if pair_moves else [tuple(result)]
    return " ".join(f"({q},{r})" for q, r in moves)


def format_score(score):
    if score >= WIN_SCORE:
        return f"+WIN"
    elif score <= -WIN_SCORE:
        return f"-WIN"
    else:
        return f"{score:+,.0f}"


def main():
    parser = argparse.ArgumentParser(description="Solve a position for forced wins.")
    parser.add_argument("position", help="Path to a saved position .pkl file")
    parser.add_argument(
        "--time-limit", type=float, default=60,
        help="Total search time limit in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=200,
        help="Maximum search depth (default: 200)",
    )
    parser.add_argument(
        "--bot", default="ai_cpp",
        help="AI module to use (default: ai_cpp)",
    )
    args = parser.parse_args()

    bot_mod = importlib.import_module(args.bot)

    board, current_player = load_position(args.position)

    game = HexGame(win_length=6)
    game.board = dict(board)
    game.current_player = current_player
    game.move_count = len(board)
    game.moves_left_in_turn = 2 if len(board) > 0 else 1

    a_count = sum(1 for p in board.values() if p == Player.A)
    b_count = sum(1 for p in board.values() if p == Player.B)
    player_name = "A (Red)" if current_player == Player.A else "B (Blue)"

    print(f"Position:  {args.position}")
    print(f"Board:     {len(board)} stones (A={a_count}, B={b_count})")
    print(f"To move:   {player_name}")
    print(f"Max depth: {args.max_depth}")
    print(f"Time:      {args.time_limit}s")
    print()
    print(f"{'Depth':>5}  {'Score':>8}  {'Nodes':>12}  {'Time':>7}  {'N/s':>10}  Best move")
    print(f"{'─'*5}  {'─'*8}  {'─'*12}  {'─'*7}  {'─'*10}  {'─'*20}")

    # Drive search depth-by-depth so we can report after each iteration.
    # The bot's TT persists across calls (same player, same position),
    # so deeper iterations benefit from earlier results.
    ai = bot_mod.MinimaxBot(time_limit=args.time_limit)
    t0 = time.time()
    total_nodes = 0
    best_result = None
    best_score = 0
    final_depth = 0

    for depth in range(1, args.max_depth + 1):
        elapsed_so_far = time.time() - t0
        remaining = args.time_limit - elapsed_so_far
        if remaining <= 0:
            break

        ai.time_limit = remaining
        ai.max_depth = depth

        dt0 = time.time()
        result = ai.get_move(game)
        dt = time.time() - dt0

        # If the bot searched deeper than requested (internal iterative
        # deepening), skip ahead to where it actually reached.
        reached = ai.last_depth
        if reached < depth:
            # Timed out before completing this depth — use last good result
            break

        total_nodes += ai._nodes
        nps = int(ai._nodes / dt) if dt > 0.001 else 0
        score_str = format_score(ai.last_score)
        move_str = format_moves(result, ai.pair_moves)

        print(f"{reached:>5}  {score_str:>8}  {ai._nodes:>12,}  {dt:>6.1f}s  {nps:>10,}  {move_str}")

        best_result = result
        best_score = ai.last_score
        final_depth = reached

        if abs(ai.last_score) >= WIN_SCORE:
            break

    elapsed = time.time() - t0
    print()
    print(f"Total: {total_nodes:,} nodes in {elapsed:.1f}s")

    if best_result is not None:
        if best_score >= WIN_SCORE:
            print(f"\n** FORCED WIN for {player_name} **")
        elif best_score <= -WIN_SCORE:
            opp = "B (Blue)" if current_player == Player.A else "A (Red)"
            print(f"\n** FORCED WIN for {opp} (to-move loses) **")
        else:
            print(f"\nNo forced win found (searched to depth {final_depth}).")


if __name__ == "__main__":
    main()
