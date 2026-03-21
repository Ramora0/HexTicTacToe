"""Analyze 6-cell line patterns from a positions.pkl file.

For each position, extracts all 6-cell windows along the 3 hex directions.
Cells are encoded as: X=current player, O=opponent, .=empty.
Patterns are canonicalized via flip (reverse) symmetry.

Reports win% for each pattern, sorted by win%.

Supports both formats:
  4-tuple: (board, current_player, eval_score, game_id)
  5-tuple: (board, current_player, eval_score, win_score, game_id)

For 4-tuple data, win is determined by eval_score > 0.
For 5-tuple data, win is determined by win_score > 0.

Usage: python -m learned_eval.analyze_patterns [path_to_positions.pkl]
       python -m learned_eval.analyze_patterns --min-games 10
"""

import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Player

HEX_DIRECTIONS = [(1, 0), (0, 1), (1, -1)]
WINDOW_LENGTH = 6


def extract_6_lines(board, current_player):
    """Extract all canonical 6-cell line patterns from the board.

    Returns set of canonical pattern tuples (after flip symmetry).
    Each cell: 0=empty, 1=current_player, 2=opponent.
    """
    patterns = set()
    occupied = set(board.keys())
    if not occupied:
        return patterns

    checked = set()
    for cq, cr in occupied:
        for dq, dr in HEX_DIRECTIONS:
            for j in range(WINDOW_LENGTH):
                start_q = cq - j * dq
                start_r = cr - j * dr
                key = (start_q, start_r, dq, dr)
                if key in checked:
                    continue
                checked.add(key)

                pat = []
                for i in range(WINDOW_LENGTH):
                    cell = board.get((start_q + i * dq, start_r + i * dr))
                    if cell is None:
                        pat.append(0)
                    elif cell == current_player:
                        pat.append(1)
                    else:
                        pat.append(2)

                pat_tuple = tuple(pat)
                if all(c == 0 for c in pat_tuple):
                    continue

                canon = min(pat_tuple, pat_tuple[::-1])
                patterns.add(canon)

    return patterns


def fmt_pat(pat):
    chars = {0: '.', 1: 'X', 2: 'O'}
    return ''.join(chars[c] for c in pat)


def analyze(positions_path, min_games=0):
    with open(positions_path, 'rb') as f:
        positions = pickle.load(f)

    print(f"File: {positions_path}")
    print(f"Total positions: {len(positions)}")

    # Detect format: 5-tuple (has win_score) or 4-tuple (eval only)
    has_win_score = len(positions[0]) == 5

    if has_win_score:
        game_ids = set(p[4] for p in positions)
        print(f"Format: 5-tuple (eval + win_score)")
    else:
        game_ids = set(p[3] for p in positions)
        print(f"Format: 4-tuple (eval only, using eval for win/loss)")

    print(f"Unique games/sources: {len(game_ids)}")

    pattern_stats = defaultdict(lambda: {
        "game_ids": set(), "wins": 0, "losses": 0, "total": 0
    })

    for idx, entry in enumerate(positions):
        if has_win_score:
            board, current_player, eval_score, win_score, game_id = entry
        else:
            board, current_player, eval_score, game_id = entry
            win_score = eval_score  # fall back to eval for win/loss

        pats = extract_6_lines(board, current_player)
        for pat in pats:
            s = pattern_stats[pat]
            s["game_ids"].add(game_id)
            s["total"] += 1
            if win_score > 0:
                s["wins"] += 1
            elif win_score < 0:
                s["losses"] += 1
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(positions)} positions...")

    print(f"\nUnique canonical patterns: {len(pattern_stats)}")

    # Build sorted list
    results = []
    for pat, s in pattern_stats.items():
        total = s["total"]
        win_pct = s["wins"] / total * 100 if total else 0
        n_games = len(s["game_ids"])
        results.append((pat, win_pct, s, n_games))

    results.sort(key=lambda x: x[1], reverse=True)

    hdr = f"{'Pattern':<10} {'Win%':>6} {'Wins':>6} {'Losses':>6} {'Positions':>10} {'Games':>6}"
    sep = "-" * len(hdr)

    # Filtered results
    filtered = [r for r in results if r[3] >= min_games] if min_games > 0 else results

    print(f"\n{'=' * len(hdr)}")
    if min_games > 0:
        print(f"TOP PATTERNS BY WIN% (min {min_games} games) — {len(filtered)} patterns")
    else:
        print(f"ALL PATTERNS BY WIN% — {len(results)} patterns")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    show_top = min(30, len(filtered))
    for pat, win_pct, s, n_games in filtered[:show_top]:
        print(f"{fmt_pat(pat):<10} {win_pct:>5.1f}% {s['wins']:>6} {s['losses']:>6} {s['total']:>10} {n_games:>6}")

    if len(filtered) > 60:
        print(f"\n... ({len(filtered) - 60} patterns omitted) ...\n")

    show_bottom = min(30, len(filtered))
    if len(filtered) > show_top:
        print(f"\nBOTTOM PATTERNS (worst for current player)")
        print(sep)
        for pat, win_pct, s, n_games in filtered[-show_bottom:]:
            print(f"{fmt_pat(pat):<10} {win_pct:>5.1f}% {s['wins']:>6} {s['losses']:>6} {s['total']:>10} {n_games:>6}")

    print(f"\nTotal patterns shown: {len(filtered)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze 6-cell line patterns in positions data")
    parser.add_argument("input", nargs="?",
                        default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"),
                        help="Path to positions.pkl file")
    parser.add_argument("--min-games", type=int, default=10,
                        help="Minimum number of unique games for a pattern to be shown (default: 10)")
    args = parser.parse_args()
    analyze(args.input, min_games=args.min_games)


if __name__ == "__main__":
    main()
