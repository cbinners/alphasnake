import json
import numpy as np
from snakeml import model, game
import random
import sys

sys.setrecursionlimit(2000)

net = model.Net()


def predict(game, playerMove):
    print("Simulating..")
    for snake in game.board['snakes']:
        print(snake.body)
    if game.over():
        score = game.score()
        record(game, score)
        return score

    (_, enemy_moves) = game.valid_moves()

    worst_score = 100.0
    worst_move = None
    for i in range(len(enemy_moves)):
        move = enemy_moves[i]

        move_to_execute = [playerMove]

        # Add the remaining enemy_moves for the bots
        for i in range(len(game.board['snakes'])-1):
            move_to_execute.append(move[i])

        game.make_move(move_to_execute)
        score = heuristic(game)
        game.undo_move()

        # Check if our score is worse
        if score < worst_score:
            worst_score = score
            worst_move = tuple(move_to_execute)

    # Actually apply the move and continue
    game.make_move(worst_move)
    value = simulate_move(game)
    game.undo_move()
    record(game, value)

    return value


def simulate_move(game):
    result = None
    done = False
    records = 1
    while not done:
        if game.over():
            result = game.score()
            record(game, result)
            done = True
            # We've recorded
            records -= 1
            game.print()
            break

        (player_moves, enemy_moves) = game.valid_moves()

        best_score = -100.0
        best_move = None
        for move in player_moves:
            for enemy_move in enemy_moves:
                move_to_execute = [move]

                for i in range(len(game.board['snakes'])-1):
                    move_to_execute.append(enemy_move[i])

                game.make_move(move_to_execute)
                score = heuristic(game)
                game.undo_move()

                # Check if our score is worse
                if score > best_score:
                    best_score = score
                    best_move = move

        worst_score = 100.0
        worst_move = None
        for i in range(len(enemy_moves)):
            move = enemy_moves[i]

            move_to_execute = [best_move]

            # Add the remaining enemy_moves for the bots
            for i in range(len(game.board['snakes'])-1):
                move_to_execute.append(move[i])

            game.make_move(move_to_execute)
            score = heuristic(game)
            game.undo_move()

            # Check if our score is worse
            if score < worst_score:
                worst_score = score
                worst_move = tuple(move_to_execute)

        # Actually apply the move and continue
        game.make_move(worst_move)
        records += 1

    # now we're done
    for i in range(records):
        record(game, result)
        game.undo_move()

    return result


def record(game, score):
    # TODO: Update ML learning algorithm
    pass


def heuristic(game):
    return random.random()


def monte_carlo_value(game, playerMove, N=100):
    scores = [predict(game, playerMove) for i in range(0, N)]
    return np.mean(scores)


def get_best_move(game):
    best_move = -1
    best_score = -1
    for i in range(4):
        score = monte_carlo_value(game, i)
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
