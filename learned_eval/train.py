"""Train pattern values via logistic regression on self-play positions.

Extracts 8-cell window features from positions, maps to canonical patterns,
and optimizes values so sigmoid(sum_of_values) predicts win probability.

Usage: python -m learned_eval.train [--input data/positions.pkl] [--epochs 200] [--lr 0.1]
"""

import json
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Player, HEX_DIRECTIONS
from learned_eval.pattern_table import (
    WINDOW_LENGTH, CANON_INDEX, CANON_SIGN, NUM_CANON,
    CANON_PATTERNS, pattern_to_int,
)

DIR_VECTORS = list(HEX_DIRECTIONS)


def extract_features(board, current_player):
    """Extract sparse feature vector {canon_index: sum_of_signs} for a position.

    Windows are 8 cells long. Patterns are read from current_player's perspective
    (current_player = 1, opponent = 2).
    """
    opponent = Player.B if current_player == Player.A else Player.A
    features = {}
    seen = set()

    for (q, r) in board:
        for d_idx, (dq, dr) in enumerate(DIR_VECTORS):
            for k in range(WINDOW_LENGTH):
                # Window starting at (q - k*dq, r - k*dr) along direction d_idx
                sq = q - k * dq
                sr = r - k * dr
                wkey = (d_idx, sq, sr)
                if wkey in seen:
                    continue
                seen.add(wkey)

                # Read the 8-cell pattern from current player's perspective
                pat_int = 0
                has_piece = False
                power = 1
                for j in range(WINDOW_LENGTH):
                    cell = board.get((sq + j * dq, sr + j * dr))
                    if cell is None:
                        v = 0
                    elif cell == current_player:
                        v = 1
                        has_piece = True
                    else:
                        v = 2
                        has_piece = True
                    pat_int += v * power
                    power *= 3

                if not has_piece:
                    continue

                ci = CANON_INDEX[pat_int]
                cs = CANON_SIGN[pat_int]
                if cs == 0:
                    continue  # self-symmetric or empty, forced zero

                if ci in features:
                    features[ci] += cs
                else:
                    features[ci] = cs

    return features


def build_dataset(positions):
    """Convert positions to sparse feature matrix, target vector, and weights.

    Each position has (board, current_player, wins, losses, draws).
    Target = empirical win rate = (wins + 0.5*draws) / total.
    Weight = total observations for that position.
    """
    print(f"Extracting features from {len(positions)} positions...")
    t0 = time.time()

    feat_indices = []
    feat_values = []
    targets = []
    weights = []

    for i, (board, cp, wins, losses, draws) in enumerate(positions):
        total = wins + losses + draws
        if total == 0:
            continue
        target = (wins + 0.5 * draws) / total

        features = extract_features(board, cp)
        if features:
            idx = np.array(list(features.keys()), dtype=np.int32)
            vals = np.array(list(features.values()), dtype=np.float64)
            feat_indices.append(idx)
            feat_values.append(vals)
            targets.append(target)
            weights.append(total)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(positions)} ({time.time()-t0:.1f}s)")

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.mean()  # normalize so mean weight = 1

    total_obs = sum(w + l + d for _, _, w, l, d in positions)
    print(f"  Done in {time.time()-t0:.1f}s, {len(targets)} usable positions from {total_obs} observations")
    return feat_indices, feat_values, np.array(targets, dtype=np.float64), weights


def train(feat_indices, feat_values, targets, weights, num_params, epochs=200, lr=0.1):
    """Train pattern values using Adam optimizer on weighted binary cross-entropy loss.

    Positions with more observations (wins+losses+draws) contribute more to the gradient.
    """
    params = np.zeros(num_params, dtype=np.float64)
    n = len(targets)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    print(f"\nTraining {num_params} parameters on {n} positions for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(epochs):
        # Forward pass: compute eval for each position
        evals = np.zeros(n, dtype=np.float64)
        for i in range(n):
            evals[i] = np.dot(params[feat_indices[i]], feat_values[i])

        # Sigmoid (clip to avoid overflow)
        evals_clipped = np.clip(evals, -30, 30)
        preds = 1.0 / (1.0 + np.exp(-evals_clipped))

        # Weighted BCE loss
        preds_safe = np.clip(preds, 1e-15, 1 - 1e-15)
        per_sample = -(targets * np.log(preds_safe) + (1 - targets) * np.log(1 - preds_safe))
        loss = np.mean(weights * per_sample)

        # Accuracy (threshold 0.5)
        correct = np.sum((preds > 0.5) == (targets > 0.5))
        acc = correct / n

        if epoch % 20 == 0 or epoch == epochs - 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}: loss={loss:.4f}  acc={acc:.3f}  ({elapsed:.1f}s)")

        # Backward pass: weighted gradient
        residuals = weights * (preds - targets) / n
        grad = np.zeros_like(params)
        for i in range(n):
            grad[feat_indices[i]] += residuals[i] * feat_values[i]

        # Adam update
        t_adam = epoch + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t_adam)
        v_hat = v / (1 - beta2 ** t_adam)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)

    print(f"\nFinal: loss={loss:.4f}  acc={acc:.3f}")
    return params


def save_results(params, output_dir):
    """Save the trained lookup table."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON: {pattern_string: value}
    result = {}
    for i, pat in enumerate(CANON_PATTERNS):
        pat_str = "".join(str(c) for c in pat)
        result[pat_str] = float(params[i])

    json_path = os.path.join(output_dir, "pattern_values.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save as numpy array
    npy_path = os.path.join(output_dir, "pattern_values.npy")
    np.save(npy_path, params)

    # Print top patterns
    sorted_idx = np.argsort(params)
    print(f"\nTop 20 most valuable patterns (for current player):")
    for i in sorted_idx[-20:][::-1]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        # Readable: . = empty, X = current player, O = opponent
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {params[i]:+.4f}")

    print(f"\nTop 20 worst patterns (for current player):")
    for i in sorted_idx[:20]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {params[i]:+.4f}")

    print(f"\nSaved to {json_path} and {npy_path}")
    return json_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {args.input}")

    feat_indices, feat_values, targets, weights = build_dataset(positions)
    params = train(feat_indices, feat_values, targets, weights, NUM_CANON,
                   epochs=args.epochs, lr=args.lr)
    save_results(params, args.output_dir)


if __name__ == "__main__":
    main()
