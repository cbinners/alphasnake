import json
import numpy as np
from snakeml import model, game

net = model.Net()


def simulate(game):
    if game.over():
        score = game.score()
        record(game.state(), score)
        return score

    return 0


def record(game, score):
    # TODO: Update ML learning algorithm
    pass


def heuristic(game):
  # TODO: Return value from ML
    return 0


def monte_carlo_value(game, N=100):
    scores = [simulate(game) for i in range(0, N)]
    return np.mean(scores)


def get_best_move(game):
    best_move = -1
    best_score = -1
    for i in range(4):
        score = monte_carlo_value(game)
        if score > best_score:
            best_score = score
            best_move = i

    return best_move, best_score


if __name__ == "__main__":
    with open("input.json") as f:
        payload = json.load(f)
        instance = game.Game(payload)
        print(get_best_move(instance))

        net.update(0, 0)

        print(net.predict())
