"""Evaluate all trained weights against ai.py using human starting positions."""
import glob
import os
import subprocess
import sys

results_dirs = sorted(glob.glob("learned_eval/results_*/pattern_values.json"))

# Also test the sweep results
sweep_dirs = sorted(glob.glob("learned_eval/sweep/*/pattern_values.json"))

all_paths = results_dirs + sweep_dirs

print(f"Testing {len(all_paths)} weight files against ai.py (400 games each, human positions)\n")
print(f"{'Weights':<55s} {'Wins':>5s} {'Loss':>5s} {'Win%':>6s}")
print(f"{'-'*55} {'-'*5} {'-'*5} {'-'*6}")

scores = []
for path in all_paths:
    name = path.replace("learned_eval/", "").replace("/pattern_values.json", "")
    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--pattern-a", path, "ai", "-n", "400", "--no-tqdm"],
        capture_output=True, text=True, timeout=600
    )
    # Parse output
    lines = result.stdout.split("\n") if result.stdout else result.stderr.split("\n")
    wins = losses = 0
    for line in lines:
        if path.replace("\\", "/") in line.replace("\\", "/") or "pattern_values" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "wins":
                    wins = int(parts[i-1])
                    break
        if "og_ai:" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "wins":
                    losses = int(parts[i-1])
                    break

    total = wins + losses
    pct = wins / total * 100 if total > 0 else 0
    scores.append((pct, wins, losses, name))
    print(f"  {name:<53s} {wins:>5d} {losses:>5d} {pct:>5.1f}%")
    sys.stdout.flush()

print(f"\n{'='*80}")
print(f"RANKED RESULTS:")
print(f"{'='*80}")
scores.sort(reverse=True)
for pct, wins, losses, name in scores:
    marker = " <-- BEST" if pct == scores[0][0] else ""
    print(f"  {pct:>5.1f}%  {wins:>3d}-{losses:<3d}  {name}{marker}")
