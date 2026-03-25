"""Rank-pair heatmap analysis for the C++ engine on human positions.

Loads positions from learned_eval/data/positions_8k.pkl, runs the C++
engine on each to collect rank-pair statistics, and generates a log-scale
heatmap showing generated/searched/chosen distributions.

Usage:
    python tests/test_cpp_ranks.py [num_positions] [time_limit] [--no-tqdm]
    python tests/test_cpp_ranks.py 5000 0.1
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import defaultdict
from multiprocessing import Pool

from game import HexGame, Player

NUM_POSITIONS = 5000
TIME_LIMIT = 0.1
POSITIONS_PATH = os.path.join(os.path.dirname(__file__),
                              '..', 'learned_eval', 'data', 'positions_8k.pkl')
# Constants from C++ engine (engine_rank.h)
ROOT_CANDIDATE_CAP = 20
CANDIDATE_CAP = 15


def _load_positions(path, n):
    """Load and sample n positions from the pickle file."""
    with open(path, 'rb') as f:
        all_pos = pickle.load(f)
    # C++ engine uses 140x140 array with offset 70 => valid coords [-70, 69]
    max_coord = 69
    all_pos = [p for p in all_pos
               if len(p[0]) >= 4
               and all(abs(q) <= max_coord and abs(r) <= max_coord
                       for (q, r) in p[0])]
    if n < len(all_pos):
        all_pos = random.sample(all_pos, n)
    return all_pos


def _eval_one(args):
    """Evaluate a single position and return rank data."""
    (board_dict, cur_player, score, game_id), time_limit = args
    import ai_cpp_rank

    bot = ai_cpp_rank.MinimaxBot(time_limit)
    bot.track_ranks = True

    game = HexGame()
    for (q, r), player in board_dict.items():
        game.board[(q, r)] = player
    game.current_player = cur_player
    game.move_count = len(board_dict)
    game.moves_left_in_turn = 2

    bot.get_move(game)
    return bot.get_rank_data(), bot.get_scatter_data()


def main():
    parser = argparse.ArgumentParser(description="Rank-pair heatmap analysis")
    parser.add_argument('num_positions', nargs='?', type=int, default=NUM_POSITIONS)
    parser.add_argument('time_limit', nargs='?', type=float, default=TIME_LIMIT)
    parser.add_argument('--no-tqdm', action='store_true', help='Disable tqdm progress bar')
    args = parser.parse_args()

    num_pos = args.num_positions
    tl = args.time_limit
    use_tqdm = not args.no_tqdm

    print(f"Loading positions from {POSITIONS_PATH}...")
    positions = _load_positions(POSITIONS_PATH, num_pos)
    num_pos = len(positions)

    args = [(pos, tl) for pos in positions]
    workers = os.cpu_count() or 1
    print(f"Evaluating {num_pos} positions, time_limit={tl}s, "
          f"CAP={ROOT_CANDIDATE_CAP} (root) / {CANDIDATE_CAP} (inner), "
          f"{workers} workers")

    t0 = time.time()
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    SCATTER_CAP = 500_000  # max points per depth label across all positions
    all_scatter = defaultdict(list)  # depth_label -> [(val1, val2, nodes, chosen), ...]

    with Pool(workers) as pool:
        results_iter = pool.imap_unordered(_eval_one, args)
        if use_tqdm:
            from tqdm import tqdm
            results_iter = tqdm(results_iter, total=num_pos,
                                desc="Evaluating", unit="pos")

        for rank_data, scatter_data in results_iter:
            for cat, depth_data in rank_data.items():
                for label, counts in depth_data.items():
                    for pair, count in counts.items():
                        all_data[cat][label][tuple(pair)] += count
            for label, points in scatter_data.items():
                if len(all_scatter[label]) < SCATTER_CAP:
                    all_scatter[label].extend(points)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s  ({num_pos / elapsed:.0f} pos/s)")

    # Collect all depth labels across categories
    all_labels = set()
    for cat in all_data:
        all_labels |= set(all_data[cat].keys())
    labels = sorted(all_labels, key=lambda x: -1 if x == 'root' else int(x[1:]))

    # Print summary
    categories = ['generated', 'nodes', 'chosen']
    cat_labels = {'generated': 'generated', 'nodes': 'nodes', 'chosen': 'chosen'}
    for label in labels:
        parts = []
        for cat in categories:
            total = sum(all_data[cat].get(label, {}).values())
            parts.append(f"{cat}={total}")
        print(f"  {label}: {', '.join(parts)}")

    # --- Precompute global color range across all heatmap subplots ---
    global_hm_vmin = float('inf')
    global_hm_vmax = 0.0
    heatmap_cache = {}  # (row, col) -> (heatmap_pct, max_rank, total)

    for row, label in enumerate(labels):
        max_rank = ROOT_CANDIDATE_CAP if label == 'root' else CANDIDATE_CAP

        for col, cat in enumerate(categories):
            counts = all_data[cat].get(label, {})
            total = sum(counts.values())

            heatmap = np.zeros((max_rank, max_rank), dtype=float)
            for (r1, r2), count in counts.items():
                if r1 < max_rank and r2 < max_rank:
                    heatmap[r1, r2] += count
                    if r1 != r2:
                        heatmap[r2, r1] += count

            heatmap_pct = heatmap / total * 100 if total > 0 else heatmap
            heatmap_cache[(row, col)] = (heatmap_pct, max_rank, total)

            positives = heatmap_pct[heatmap_pct > 0]
            if positives.size:
                global_hm_vmin = min(global_hm_vmin, positives.min())
                global_hm_vmax = max(global_hm_vmax, positives.max())

    global_hm_vmin = max(global_hm_vmin, 0.001)
    if global_hm_vmax <= global_hm_vmin:
        global_hm_vmax = 1.0

    # --- Plot: 3 columns (generated/nodes/chosen) x N depth rows ---
    n_labels = len(labels)
    fig, axes = plt.subplots(n_labels, 3, figsize=(20, 5 * n_labels))
    if n_labels == 1:
        axes = [axes]

    for row, label in enumerate(labels):
        for col, cat in enumerate(categories):
            heatmap_pct, max_rank, total = heatmap_cache[(row, col)]

            ax = axes[row][col]

            # Log-scale colormap with global range
            heatmap_plot = np.ma.masked_where(heatmap_pct == 0, heatmap_pct)
            im = ax.imshow(heatmap_plot, cmap='YlOrRd', origin='upper',
                           aspect='equal',
                           norm=LogNorm(vmin=global_hm_vmin, vmax=global_hm_vmax))

            ax.set_title(f'{label} {cat} — n={total}')
            ax.set_xlabel('Move rank')
            ax.set_ylabel('Move rank')
            ax.set_xticks(range(max_rank))
            ax.set_yticks(range(max_rank))
            plt.colorbar(im, ax=ax, label='%', shrink=0.8)

            # Annotate cells with one decimal place
            for i in range(max_rank):
                for j in range(max_rank):
                    val = heatmap_pct[i, j]
                    if val > 0.05:
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=4,
                                color='black' if val < global_hm_vmax * 0.3 else 'white')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'rank_heatmap_cpp.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    # --- Scatter plot: value-space density of nodes vs chosen ---
    scatter_labels = [l for l in labels if all_scatter.get(l)]
    if scatter_labels:
        # Precompute global axis and color ranges
        global_xmin = global_ymin = float('inf')
        global_xmax = global_ymax = float('-inf')
        global_node_max = 0
        global_hist_vmin = float('inf')
        global_hist_vmax = 0.0
        bins = 30

        scatter_arrays = {}
        hist_cache = {}
        for label in scatter_labels:
            points = all_scatter[label]
            v1 = np.array([p[0] for p in points])
            v2 = np.array([p[1] for p in points])
            nd = np.array([p[2] for p in points])
            ch = np.array([p[3] for p in points])
            scatter_arrays[label] = (v1, v2, nd, ch)
            global_xmin = min(global_xmin, v1.min())
            global_xmax = max(global_xmax, v1.max())
            global_ymin = min(global_ymin, v2.min())
            global_ymax = max(global_ymax, v2.max())
            global_node_max = max(global_node_max, nd.max())

        # Compute histograms with shared bin edges
        xpad = (global_xmax - global_xmin) * 0.02 or 1.0
        ypad = (global_ymax - global_ymin) * 0.02 or 1.0
        shared_xedges = np.linspace(global_xmin - xpad, global_xmax + xpad, bins + 1)
        shared_yedges = np.linspace(global_ymin - ypad, global_ymax + ypad, bins + 1)

        for label in scatter_labels:
            v1, v2, nd, ch = scatter_arrays[label]
            if len(v1) > 10:
                h, _, _ = np.histogram2d(v1, v2, bins=[shared_xedges, shared_yedges],
                                         weights=nd)
                hist_cache[label] = h
                positives = h[h > 0]
                if positives.size:
                    global_hist_vmin = min(global_hist_vmin, positives.min())
                    global_hist_vmax = max(global_hist_vmax, positives.max())

        global_hist_vmin = max(global_hist_vmin, 1)
        if global_hist_vmax <= global_hist_vmin:
            global_hist_vmax = global_hist_vmin * 10

        n_scatter = len(scatter_labels)
        fig2, axes2 = plt.subplots(n_scatter, 2, figsize=(16, 6 * n_scatter))
        if n_scatter == 1:
            axes2 = [axes2]

        for row, label in enumerate(scatter_labels):
            v1, v2, nd, ch = scatter_arrays[label]
            unchosen = ~ch

            # Left: scatter with consistent axes and dot sizing
            ax = axes2[row][0]
            if unchosen.any():
                sizes = np.clip(nd[unchosen] / max(global_node_max, 1) * 100, 1, 200)
                ax.scatter(v1[unchosen], v2[unchosen], s=sizes,
                           alpha=0.15, c='steelblue', edgecolors='none', label='searched')
            if ch.any():
                ax.scatter(v1[ch], v2[ch], s=30, alpha=0.6,
                           c='red', edgecolors='darkred', linewidths=0.5,
                           marker='*', label='chosen', zorder=5)
            ax.set_xlim(global_xmin - xpad, global_xmax + xpad)
            ax.set_ylim(global_ymin - ypad, global_ymax + ypad)
            ax.set_title(f'{label} — value-space node density (n={len(v1)})')
            ax.set_xlabel('Candidate score (lower)')
            ax.set_ylabel('Candidate score (higher)')
            ax.legend(fontsize=8, loc='upper left')

            # Right: 2D histogram with consistent bins and color scale
            ax = axes2[row][1]
            if label in hist_cache:
                h = hist_cache[label]
                h_masked = np.ma.masked_where(h == 0, h)
                im = ax.pcolormesh(shared_xedges, shared_yedges, h_masked.T,
                                   cmap='YlOrRd',
                                   norm=LogNorm(vmin=global_hist_vmin,
                                                vmax=global_hist_vmax))
                plt.colorbar(im, ax=ax, label='total nodes', shrink=0.8)
                if ch.any():
                    ax.scatter(v1[ch], v2[ch], s=20, alpha=0.7,
                               c='cyan', edgecolors='black', linewidths=0.5,
                               marker='*', label='chosen', zorder=5)
                    ax.legend(fontsize=8, loc='upper left')
            ax.set_xlim(global_xmin - xpad, global_xmax + xpad)
            ax.set_ylim(global_ymin - ypad, global_ymax + ypad)
            ax.set_title(f'{label} — node cost heatmap in value space')
            ax.set_xlabel('Candidate score (lower)')
            ax.set_ylabel('Candidate score (higher)')

        plt.tight_layout()
        scatter_path = os.path.join(os.path.dirname(__file__), '..', 'rank_scatter_cpp.png')
        plt.savefig(scatter_path, dpi=150)
        print(f"Saved {scatter_path}")

    # --- Waste analysis: nodes spent on pairs never chosen ---
    print("\n--- Waste analysis: nodes on unchosen pairs ---")
    for label in labels:
        nodes = all_data['nodes'].get(label, {})
        chosen = all_data['chosen'].get(label, {})
        total_nodes = sum(nodes.values())
        if total_nodes == 0:
            continue

        # Pairs that consumed nodes but were never chosen
        waste = {}
        for pair, n_count in nodes.items():
            if chosen.get(pair, 0) == 0:
                waste[pair] = n_count

        total_waste = sum(waste.values())
        print(f"\n  {label}: {total_waste}/{total_nodes} nodes wasted "
              f"({100*total_waste/total_nodes:.1f}%)")

        sorted_waste = sorted(waste.items(), key=lambda x: -x[1])
        print(f"  Top 10 wasted pairs:")
        for (r1, r2), w in sorted_waste[:10]:
            print(f"    ({r1:2d},{r2:2d}): {w:>10d} nodes "
                  f"({100*w/total_nodes:.1f}%)")


if __name__ == '__main__':
    main()
