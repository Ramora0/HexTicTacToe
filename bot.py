"""Base class for Hex Tic-Tac-Toe bots."""

import random
from abc import ABC, abstractmethod
from game import Player


class Bot(ABC):
    """Abstract bot. Subclasses must implement get_move and respect time_limit."""

    def __init__(self, time_limit=0.05):
        self.time_limit = time_limit
        self.last_depth = 0  # depth reached on most recent get_move call

    @abstractmethod
    def get_move(self, game) -> tuple[int, int]:
        """Return (q, r) for the next move. Must return within self.time_limit seconds."""
        ...

    def __str__(self):
        return self.__class__.__name__


class RandomBot(Bot):
    """Places stones randomly. Useful as a baseline."""

    def get_move(self, game):
        empty = [pos for pos, p in game.board.items() if p == Player.NONE]
        return random.choice(empty)
