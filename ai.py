"""Minimax bot with iterative deepening, heuristic eval, and move ordering.

Uses alpha-beta pruning with a heuristic evaluation function that scores
positions based on contiguous line segments along the three hex axes.
Precomputes all 6-cell windows for fast evaluation.
"""

import math
import random
import time
from bot import Bot
from game import Player, HEX_DIRECTIONS


class TimeUp(Exception):
    pass


def hex_distance(dq, dr):
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))


# Precompute all cells on the board
RADIUS = 5
ALL_CELLS = set()
for _q in range(-RADIUS, RADIUS + 1):
    for _r in range(-RADIUS, RADIUS + 1):
        if abs(-_q - _r) <= RADIUS:
            ALL_CELLS.add((_q, _r))

# Precompute all 6-cell windows (as tuples of coordinates)
WIN_WINDOWS = []
for dq, dr in HEX_DIRECTIONS:
    visited = set()
    for cell in ALL_CELLS:
        if cell in visited:
            continue
        q, r = cell
        while (q - dq, r - dr) in ALL_CELLS:
            q -= dq
            r -= dr
        line = []
        cq, cr = q, r
        while (cq, cr) in ALL_CELLS:
            visited.add((cq, cr))
            line.append((cq, cr))
            cq += dq
            cr += dr
        for i in range(len(line) - 5):
            WIN_WINDOWS.append(tuple(line[i:i+6]))

# Scores for contiguous groups of length N (index = count)
LINE_SCORES = [0, 1, 10, 100, 1000, 10000, 100000]


def evaluate_position(game, player):
    """Score the position from player's perspective using precomputed windows."""
    opponent = Player.B if player == Player.A else Player.A
    board = game.board
    score = 0
    none = Player.NONE

    for window in WIN_WINDOWS:
        my_count = 0
        opp_count = 0
        for cell in window:
            v = board[cell]
            if v == player:
                my_count += 1
            elif v != none:
                opp_count += 1
        if my_count > 0 and opp_count == 0:
            score += LINE_SCORES[my_count]
        elif opp_count > 0 and my_count == 0:
            score -= LINE_SCORES[opp_count]

    return score


# Precompute neighbor offsets for distance 2
_NEIGHBOR_OFFSETS = []
for _dq in range(-2, 3):
    for _dr in range(-2, 3):
        if hex_distance(_dq, _dr) <= 2 and (_dq, _dr) != (0, 0):
            _NEIGHBOR_OFFSETS.append((_dq, _dr))


def get_candidates(game):
    """Return empty cells within hex-distance 2 of any occupied cell."""
    occupied = [pos for pos, p in game.board.items() if p != Player.NONE]
    if not occupied:
        return [(0, 0)]

    board = game.board
    none = Player.NONE
    candidates = set()
    for q, r in occupied:
        for dq, dr in _NEIGHBOR_OFFSETS:
            nq, nr = q + dq, r + dr
            if (nq, nr) in board and board[(nq, nr)] == none:
                candidates.add((nq, nr))
    return list(candidates)


class MinimaxBot(Bot):
    """Iterative-deepening minimax with alpha-beta pruning and heuristic eval."""

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0

    def get_move(self, game):
        self._deadline = time.time() + self.time_limit
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0

        candidates = get_candidates(game)
        if len(candidates) == 1:
            return candidates[0]

        random.shuffle(candidates)
        best_move = candidates[0]

        saved_board = dict(game.board)
        saved_state = game.save_state()
        saved_move_count = game.move_count

        for depth in range(1, 200):
            try:
                best_move = self._search_root(game, candidates, depth)
                self.last_depth = depth
            except TimeUp:
                game.board = saved_board
                game.move_count = saved_move_count
                (game.current_player, game.moves_left_in_turn,
                 game.winner, game.winning_cells, game.game_over) = saved_state
                break

        return best_move

    def _check_time(self):
        self._nodes += 1
        if self._nodes % 128 == 0 and time.time() >= self._deadline:
            raise TimeUp

    def _search_root(self, game, candidates, depth):
        maximizing = game.current_player == self._player
        best_move = candidates[0]
        best_score = -math.inf if maximizing else math.inf

        for q, r in candidates:
            self._check_time()
            state = game.save_state()
            game.make_move(q, r)
            score = self._minimax(game, depth - 1, -math.inf, math.inf)
            game.undo_move(q, r, state)

            if maximizing and score > best_score:
                best_score = score
                best_move = (q, r)
            elif not maximizing and score < best_score:
                best_score = score
                best_move = (q, r)

        return best_move

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return 100000000
            elif game.winner != Player.NONE:
                return -100000000
            return 0

        if depth == 0:
            return evaluate_position(game, self._player)

        candidates = get_candidates(game)
        maximizing = game.current_player == self._player

        if maximizing:
            value = -math.inf
            for q, r in candidates:
                state = game.save_state()
                game.make_move(q, r)
                value = max(value, self._minimax(game, depth - 1, alpha, beta))
                game.undo_move(q, r, state)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for q, r in candidates:
                state = game.save_state()
                game.make_move(q, r)
                value = min(value, self._minimax(game, depth - 1, alpha, beta))
                game.undo_move(q, r, state)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
