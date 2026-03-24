"""Profile ai.py search: pruning efficiency, time breakdown, TT stats, move ordering quality.

Usage: python profile_ai.py [--moves N] [--time-limit T]
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import math
import cProfile
import pstats
import io
from collections import defaultdict
from game import HexGame, Player
from ai import MinimaxBot, _EXACT, _LOWER, _UPPER


class InstrumentedBot(MinimaxBot):
    """MinimaxBot with detailed statistics collection."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.stats = {}

    def get_move(self, game):
        self._reset_stats()
        result = super().get_move(game)
        self._finalize_stats()
        return result

    def _reset_stats(self):
        self.stats = {
            # Node counts
            'total_nodes': 0,
            'nodes_by_depth': defaultdict(int),  # depth -> count
            'leaf_evals': 0,
            'depth0_evals': 0,

            # Alpha-beta pruning
            'cutoffs': 0,
            'no_cutoffs': 0,  # nodes where all children searched
            'first_move_cutoffs': 0,  # cutoff on the first child (perfect ordering)
            'cutoff_index_sum': 0,  # sum of child indices where cutoff happened
            'cutoff_count_for_avg': 0,

            # TT stats
            'tt_probes': 0,
            'tt_hits': 0,
            'tt_exact_hits': 0,
            'tt_bound_useful': 0,  # bound caused cutoff
            'tt_stores': 0,

            # Move ordering
            'tt_move_available': 0,
            'tt_move_caused_cutoff': 0,

            # Candidate counts
            'candidate_counts': [],  # list of candidate set sizes at each node

            # Time breakdown (sampled)
            'time_make': 0.0,
            'time_undo': 0.0,
            'time_move_ordering': 0.0,
            'time_eval_leaf': 0.0,
            'time_tt': 0.0,

            # Depth completed
            'completed_depths': [],
            'nodes_per_depth': [],  # (depth, nodes) for each completed ID iteration

            # Late move reductions
            'lmr_attempts': 0,
            'lmr_re_searches': 0,

            # Killer / history
            'history_max': 0,
        }

    def _finalize_stats(self):
        s = self.stats
        s['total_nodes'] = self._nodes
        if self._history:
            s['history_max'] = max(self._history.values())

    def _check_time(self):
        # Keep original behavior, just count
        self._nodes += 1
        if self._nodes % 128 == 0 and time.time() >= self._deadline:
            from ai import TimeUp
            raise TimeUp

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return 100000000
            elif game.winner != Player.NONE:
                return -100000000
            return 0

        # TT lookup with stats
        t0 = time.time()
        tt_key = self._tt_key(game)
        tt_entry = self._tt.get(tt_key)
        tt_move = None
        self.stats['tt_probes'] += 1
        if tt_entry:
            self.stats['tt_hits'] += 1
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == _EXACT:
                    self.stats['tt_exact_hits'] += 1
                    self.stats['time_tt'] += time.time() - t0
                    return tt_score
                elif tt_flag == _LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == _UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    self.stats['tt_bound_useful'] += 1
                    self.stats['time_tt'] += time.time() - t0
                    return tt_score
        self.stats['time_tt'] += time.time() - t0

        if depth == 0:
            self.stats['depth0_evals'] += 1
            score = self._eval_score
            self._tt[tt_key] = (0, score, _EXACT, None)
            return score

        if depth == 1:
            self.stats['leaf_evals'] += 1
            t0 = time.time()
            result = self._eval_leaf(game, alpha, beta)
            self.stats['time_eval_leaf'] += time.time() - t0
            return result

        orig_alpha = alpha
        orig_beta = beta
        maximizing = game.current_player == self._player

        # Move ordering with timing
        t0 = time.time()
        candidates = list(self._cand_set)
        self.stats['candidate_counts'].append(len(candidates))
        history = self._history
        is_a = game.current_player == Player.A
        root_deltas = self._root_deltas_a if is_a else self._root_deltas_b
        delta_sign = 0.001 if maximizing else -0.001
        candidates.sort(
            key=lambda m: history.get(m, 0) + root_deltas.get(m, 0) * delta_sign,
            reverse=True)
        has_tt_move = tt_move in self._cand_set
        if has_tt_move:
            self.stats['tt_move_available'] += 1
            idx = candidates.index(tt_move)
            candidates[0], candidates[idx] = candidates[idx], candidates[0]
        self.stats['time_move_ordering'] += time.time() - t0

        best_move = None
        children_searched = 0

        if maximizing:
            value = -math.inf
            for i, (q, r) in enumerate(candidates):
                player = game.current_player
                state = (player, game.moves_left_in_turn, game.winner, game.game_over)
                t0 = time.time()
                self._make(game, q, r)
                self.stats['time_make'] += time.time() - t0

                if i >= 3 and depth >= 3:
                    self.stats['lmr_attempts'] += 1
                    child_val = self._minimax(game, depth - 2, alpha, beta)
                    if child_val > alpha:
                        self.stats['lmr_re_searches'] += 1
                        child_val = self._minimax(game, depth - 1, alpha, beta)
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)

                t0 = time.time()
                self._undo(game, q, r, state, player)
                self.stats['time_undo'] += time.time() - t0

                children_searched += 1
                if child_val > value:
                    value = child_val
                    best_move = (q, r)
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.stats['cutoffs'] += 1
                    if i == 0:
                        self.stats['first_move_cutoffs'] += 1
                        if has_tt_move:
                            self.stats['tt_move_caused_cutoff'] += 1
                    self.stats['cutoff_index_sum'] += i
                    self.stats['cutoff_count_for_avg'] += 1
                    history[(q, r)] = history.get((q, r), 0) + depth * depth
                    break
            else:
                self.stats['no_cutoffs'] += 1
        else:
            value = math.inf
            for i, (q, r) in enumerate(candidates):
                player = game.current_player
                state = (player, game.moves_left_in_turn, game.winner, game.game_over)
                t0 = time.time()
                self._make(game, q, r)
                self.stats['time_make'] += time.time() - t0

                if i >= 3 and depth >= 3:
                    self.stats['lmr_attempts'] += 1
                    child_val = self._minimax(game, depth - 2, alpha, beta)
                    if child_val < beta:
                        self.stats['lmr_re_searches'] += 1
                        child_val = self._minimax(game, depth - 1, alpha, beta)
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)

                t0 = time.time()
                self._undo(game, q, r, state, player)
                self.stats['time_undo'] += time.time() - t0

                children_searched += 1
                if child_val < value:
                    value = child_val
                    best_move = (q, r)
                beta = min(beta, value)
                if alpha >= beta:
                    self.stats['cutoffs'] += 1
                    if i == 0:
                        self.stats['first_move_cutoffs'] += 1
                        if has_tt_move:
                            self.stats['tt_move_caused_cutoff'] += 1
                    self.stats['cutoff_index_sum'] += i
                    self.stats['cutoff_count_for_avg'] += 1
                    history[(q, r)] = history.get((q, r), 0) + depth * depth
                    break
            else:
                self.stats['no_cutoffs'] += 1

        self.stats['nodes_by_depth'][depth] += 1

        if value <= orig_alpha:
            flag = _UPPER
        elif value >= orig_beta:
            flag = _LOWER
        else:
            flag = _EXACT

        self.stats['tt_stores'] += 1
        self._tt[tt_key] = (depth, value, flag, best_move)
        return value


def play_sample_game(time_limit=0.5, num_moves=20):
    """Play a partial game and collect stats from each move."""
    game = HexGame(win_length=6)
    bot = InstrumentedBot(time_limit=time_limit)
    all_stats = []

    for move_num in range(num_moves):
        if game.game_over:
            break

        bot._player = game.current_player
        result = bot.get_move(game)
        moves = result if isinstance(result, list) else [result]

        stats = dict(bot.stats)
        stats['move_num'] = move_num
        stats['board_size'] = len(game.board)
        stats['depth_reached'] = bot.last_depth
        stats['ebf'] = bot.last_ebf
        stats['candidates_at_root'] = len(list(bot._cand_set)) if bot._cand_set else 0
        all_stats.append(stats)

        for q, r in moves:
            if game.game_over:
                break
            game.make_move(q, r)

    return all_stats


def run_cprofile(time_limit=0.5, num_moves=8):
    """Run cProfile on a few moves to find hotspots."""
    game = HexGame(win_length=6)
    bot = MinimaxBot(time_limit=time_limit)

    # Play a few moves to get an interesting position
    opening = [(0, 0), (-1, 0), (-1, 1), (1, -1), (1, 0)]
    for q, r in opening:
        game.make_move(q, r)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(num_moves):
        if game.game_over:
            break
        result = bot.get_move(game)
        moves = result if isinstance(result, list) else [result]
        for q, r in moves:
            if game.game_over:
                break
            game.make_move(q, r)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    return s.getvalue()


def print_report(all_stats):
    """Print a comprehensive optimization report."""
    print("=" * 70)
    print("  AI SEARCH PROFILING REPORT")
    print("=" * 70)

    # Aggregate across all moves
    total_nodes = sum(s['total_nodes'] for s in all_stats)
    total_cutoffs = sum(s['cutoffs'] for s in all_stats)
    total_no_cutoffs = sum(s['no_cutoffs'] for s in all_stats)
    total_first_move = sum(s['first_move_cutoffs'] for s in all_stats)
    total_tt_probes = sum(s['tt_probes'] for s in all_stats)
    total_tt_hits = sum(s['tt_hits'] for s in all_stats)
    total_tt_exact = sum(s['tt_exact_hits'] for s in all_stats)
    total_tt_bound = sum(s['tt_bound_useful'] for s in all_stats)
    total_tt_stores = sum(s['tt_stores'] for s in all_stats)
    total_tt_move_avail = sum(s['tt_move_available'] for s in all_stats)
    total_tt_move_cutoff = sum(s['tt_move_caused_cutoff'] for s in all_stats)
    total_cutoff_idx = sum(s['cutoff_index_sum'] for s in all_stats)
    total_cutoff_count = sum(s['cutoff_count_for_avg'] for s in all_stats)
    total_lmr = sum(s['lmr_attempts'] for s in all_stats)
    total_lmr_re = sum(s['lmr_re_searches'] for s in all_stats)
    total_leaf = sum(s['leaf_evals'] for s in all_stats)
    total_d0 = sum(s['depth0_evals'] for s in all_stats)

    all_cand = []
    for s in all_stats:
        all_cand.extend(s['candidate_counts'])

    # Time breakdown (from instrumented _minimax only, not cProfile)
    t_make = sum(s['time_make'] for s in all_stats)
    t_undo = sum(s['time_undo'] for s in all_stats)
    t_ordering = sum(s['time_move_ordering'] for s in all_stats)
    t_leaf = sum(s['time_eval_leaf'] for s in all_stats)
    t_tt = sum(s['time_tt'] for s in all_stats)

    interior_nodes = total_cutoffs + total_no_cutoffs

    # ---- Per-move summary ----
    print("\n  PER-MOVE SUMMARY")
    print("  " + "-" * 40)
    print(f"  {'Move':>4s} {'Board':>5s} {'Depth':>5s} {'EBF':>5s} {'Nodes':>8s} {'Cands':>5s} {'Cutoff%':>7s} {'TT hit%':>7s}")
    for s in all_stats:
        cutoff_pct = 0
        interior = s['cutoffs'] + s['no_cutoffs']
        if interior > 0:
            cutoff_pct = 100 * s['cutoffs'] / interior
        tt_pct = 100 * s['tt_hits'] / s['tt_probes'] if s['tt_probes'] else 0
        cands = s.get('candidates_at_root', 0)
        print(f"  {s['move_num']:4d} {s['board_size']:5d} {s['depth_reached']:5d} "
              f"{s['ebf']:5.1f} {s['total_nodes']:8d} {cands:5d} "
              f"{cutoff_pct:6.1f}% {tt_pct:6.1f}%")

    # ---- Alpha-Beta Pruning ----
    print(f"\n  ALPHA-BETA PRUNING")
    print("  " + "-" * 40)
    print(f"  Total nodes searched:       {total_nodes:>10,d}")
    print(f"  Interior nodes (depth>=2):  {interior_nodes:>10,d}")
    print(f"  Cutoffs:                    {total_cutoffs:>10,d}")
    if interior_nodes > 0:
        print(f"  Cutoff rate:                {100*total_cutoffs/interior_nodes:>9.1f}%")
    if total_cutoffs > 0:
        print(f"  First-move cutoff rate:     {100*total_first_move/total_cutoffs:>9.1f}%")
        # Perfect ordering = 100% first-move cutoffs
        # Good ordering = >70% first-move cutoffs
    if total_cutoff_count > 0:
        avg_idx = total_cutoff_idx / total_cutoff_count
        print(f"  Avg cutoff child index:     {avg_idx:>9.2f}")
        print(f"    (0.0=perfect ordering, lower=better)")

    # ---- Transposition Table ----
    print(f"\n  TRANSPOSITION TABLE")
    print("  " + "-" * 40)
    print(f"  Probes:                     {total_tt_probes:>10,d}")
    print(f"  Hits:                       {total_tt_hits:>10,d}")
    if total_tt_probes > 0:
        print(f"  Hit rate:                   {100*total_tt_hits/total_tt_probes:>9.1f}%")
    print(f"  Exact cutoffs:              {total_tt_exact:>10,d}")
    print(f"  Bound cutoffs:              {total_tt_bound:>10,d}")
    print(f"  Stores:                     {total_tt_stores:>10,d}")
    if total_tt_move_avail > 0:
        print(f"  TT move available:          {total_tt_move_avail:>10,d}")
        print(f"  TT move caused cutoff:      {total_tt_move_cutoff:>10,d} "
              f"({100*total_tt_move_cutoff/total_tt_move_avail:.1f}%)")

    # ---- Late Move Reductions ----
    print(f"\n  LATE MOVE REDUCTIONS")
    print("  " + "-" * 40)
    print(f"  LMR attempts:               {total_lmr:>10,d}")
    print(f"  LMR re-searches:            {total_lmr_re:>10,d}")
    if total_lmr > 0:
        print(f"  LMR success rate:           {100*(total_lmr-total_lmr_re)/total_lmr:>9.1f}%")
        print(f"    (higher = more nodes saved by reduced search)")

    # ---- Candidate Set ----
    print(f"\n  CANDIDATE SET SIZE")
    print("  " + "-" * 40)
    if all_cand:
        print(f"  Mean:                       {sum(all_cand)/len(all_cand):>9.1f}")
        all_cand_s = sorted(all_cand)
        print(f"  Median:                     {all_cand_s[len(all_cand_s)//2]:>9d}")
        print(f"  Min:                        {min(all_cand):>9d}")
        print(f"  Max:                        {max(all_cand):>9d}")
        # Distribution
        buckets = defaultdict(int)
        for c in all_cand:
            bucket = (c // 10) * 10
            buckets[bucket] += 1
        print(f"  Distribution:")
        for b in sorted(buckets):
            bar = "#" * min(50, buckets[b])
            print(f"    {b:3d}-{b+9:3d}: {buckets[b]:5d} {bar}")

    # ---- Eval Breakdown ----
    print(f"\n  EVAL BREAKDOWN")
    print("  " + "-" * 40)
    print(f"  Leaf evals (depth=1 scan):  {total_leaf:>10,d}")
    print(f"  Depth-0 evals (static):     {total_d0:>10,d}")
    total_evals = total_leaf + total_d0
    if total_nodes > 0:
        print(f"  Eval% of total nodes:       {100*total_evals/total_nodes:>9.1f}%")

    # ---- Time Breakdown (instrumented) ----
    print(f"\n  TIME BREAKDOWN (instrumented, depth>=2 nodes)")
    print("  " + "-" * 40)
    t_total = t_make + t_undo + t_ordering + t_leaf + t_tt
    if t_total > 0:
        for label, t in [("_make", t_make), ("_undo", t_undo),
                          ("move ordering", t_ordering),
                          ("_eval_leaf", t_leaf), ("TT lookup", t_tt)]:
            print(f"  {label:>20s}: {t:7.3f}s ({100*t/t_total:5.1f}%)")
        print(f"  {'TOTAL measured':>20s}: {t_total:7.3f}s")
    else:
        print("  (no timing data collected)")

    # ---- Effective Branching Factor ----
    print(f"\n  EFFECTIVE BRANCHING FACTOR")
    print("  " + "-" * 40)
    for s in all_stats:
        if s['ebf'] > 0:
            print(f"  Move {s['move_num']:2d}: depth={s['depth_reached']}, "
                  f"EBF={s['ebf']:.1f}, nodes={s['total_nodes']:,d}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--moves", type=int, default=12, help="Moves to profile")
    parser.add_argument("--time-limit", type=float, default=0.5, help="Time per move (s)")
    parser.add_argument("--cprofile", action="store_true", help="Also run cProfile")
    args = parser.parse_args()

    print(f"Running {args.moves} moves with {args.time_limit}s per move...\n")

    all_stats = play_sample_game(time_limit=args.time_limit, num_moves=args.moves)
    print_report(all_stats)

    if args.cprofile:
        print("\n\n  cPROFILE HOTSPOT ANALYSIS")
        print("=" * 70)
        print(run_cprofile(time_limit=args.time_limit, num_moves=6))
