"""Pattern table for 6-cell hex window evaluation with symmetry reduction.

Each window is 6 cells along a hex axis. Cell states: 0=empty, 1=current player, 2=opponent.
Symmetries:
  - Flipping (reverse): same value
  - NO piece swap symmetry — offense and defense can have different weights
"""

WINDOW_LENGTH = 6
NUM_PATTERNS = 3 ** WINDOW_LENGTH  # 6561 including all-empty



def _int_to_pattern(n):
    """Convert integer [0, 6561) to 8-cell pattern tuple."""
    pat = []
    for _ in range(WINDOW_LENGTH):
        pat.append(n % 3)
        n //= 3
    return tuple(pat)


def _pattern_to_int(pat):
    """Convert 8-cell pattern tuple to integer [0, 6561)."""
    n = 0
    for i in range(WINDOW_LENGTH - 1, -1, -1):
        n = n * 3 + pat[i]
    return n


def _flip(pat):
    return pat[::-1]



def build_tables():
    """Build the canonical pattern table.

    Only flip (reverse) symmetry is used — no piece swap.
    This allows offense (1=me) and defense (2=opponent) to have
    independent weights (e.g., blocking can be worth more than extending).

    Returns:
        canon_patterns: list of canonical pattern tuples (the learnable set)
        pattern_map: dict mapping pattern_int -> (canon_index, sign)
            sign is always +1 (no negation from swap)
            canon_index is -1 for all-empty
    """
    canon_patterns = []
    canon_lookup = {}  # canonical pattern tuple -> index
    pattern_map = {}   # pattern_int -> (canon_index, sign)

    for i in range(NUM_PATTERNS):
        if i in pattern_map:
            continue

        pat = _int_to_pattern(i)

        # Skip all-empty
        if all(c == 0 for c in pat):
            pattern_map[i] = (-1, 0)
            continue

        # Only flip symmetry: pat and reversed pat share the same value
        p_flip = _flip(pat)
        canon = min(pat, p_flip)

        if canon not in canon_lookup:
            canon_lookup[canon] = len(canon_patterns)
            canon_patterns.append(canon)
        cidx = canon_lookup[canon]

        for p in (pat, p_flip):
            pi = _pattern_to_int(p)
            if pi not in pattern_map:
                pattern_map[pi] = (cidx, 1)

    return canon_patterns, pattern_map


# Pre-compute on import
CANON_PATTERNS, PATTERN_MAP = build_tables()
NUM_CANON = len(CANON_PATTERNS)

# Fast lookup array: index by pattern_int -> (canon_index, sign)
# For patterns not in map (shouldn't happen), default to (-1, 0)
CANON_INDEX = [0] * NUM_PATTERNS
CANON_SIGN = [0] * NUM_PATTERNS
for _pi, (_ci, _s) in PATTERN_MAP.items():
    CANON_INDEX[_pi] = _ci
    CANON_SIGN[_pi] = _s


def pattern_to_int(pat):
    return _pattern_to_int(pat)


if __name__ == "__main__":
    print(f"Total non-empty patterns: {NUM_PATTERNS - 1}")
    print(f"Canonical learnable patterns (flip-only): {NUM_CANON}")
    covered = sum(1 for ci, s in PATTERN_MAP.values() if s != 0)
    print(f"Patterns with sign: {covered}")
    print(f"Check: {covered} + 1 (empty) = {covered + 1} (should be {NUM_PATTERNS})")
