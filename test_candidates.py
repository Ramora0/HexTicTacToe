import random
from bot import Bot, RandomBot
from game import Player
from ai import get_candidates
from evaluate import evaluate


class CandidateRandomBot(Bot):
    def get_move(self, game):
        self.last_depth = 0
        return random.choice(get_candidates(game))


if __name__ == "__main__":
    evaluate(CandidateRandomBot(time_limit=0.5), RandomBot(time_limit=0.5), num_games=200)
