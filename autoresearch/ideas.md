# Optimization Ideas

## Current champion params (60 experiments, 22 kept)
```python
LINE_SCORES = [0, 0, 8, 1200, 3000, 50000, 100000]
_DEF_MULT = [0, 1.0, 1.0, 1.0, 1.8, 2.5, 1.0]
_CANDIDATE_CAP = 11
_ROOT_CANDIDATE_CAP = 13
_DELTA_WEIGHT = 1.5
time_check_interval = 1024
+ hot window sets for instant win/threat detection
avg depth: 2.4-2.5 (original was 2.1)
```

## All parameters confirmed near-optimal
Every parameter has been tested in both directions. Changes that work:
- Smaller increments may still find marginal gains
- Parameter interactions mean re-testing after other changes can help

## Speed ideas still untried
- [ ] Negamax refactor (eliminate duplicated max/min branches — less code, possibly faster)
- [ ] __slots__ on MinimaxBot
- [ ] Precompute move deltas into a list before sorting (avoid repeated method call overhead)
- [ ] Reduce Python overhead in _make/_undo (the main bottleneck: 27% + 24% of time)
