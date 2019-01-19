import json
import numpy as np
from snakeml import model, game as G
import random
import sys


def predict(net, game, playerMove):
    print("Simulating..")
    for snake in game.board['snakes']:
        print(snake.body)
    if game.over():
        score = game.score()
        record(net, game, score)
        return score

    (_, enemy_moves) = game.valid_moves()

    worst_score = 1.0
    worst_move = None
    for i in range(len(enemy_moves)):
        move = enemy_moves[i]

        move_to_execute = [playerMove]

        # Add the remaining enemy_moves for the bots
        for i in range(len(game.board['snakes'])-1):
            move_to_execute.append(move[i])

        game.make_move(move_to_execute)
        score = heuristic(net, game)
        game.undo_move()

        # Check if our score is worse
        if score < worst_score:
            worst_score = score
            worst_move = tuple(move_to_execute)

    # Actually apply the move and continue
    game.make_move(worst_move)
    value = simulate_move(net, game)
    game.undo_move()
    record(net, game, value)

    print("Score, saving model", value)

    return value


def simulate_move(net, game):
    result = None
    done = False
    records = 1
    while not done:
        if game.over():
            result = game.score()
            record(net, game, result)
            done = True
            # We've recorded
            records -= 1
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
                score = heuristic(net, game)
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
            score = heuristic(net, game)
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
        record(net, game, result)
        game.undo_move()

    return result


def record(net, game, score):
    net.update(game.state(), score)
    pass


def heuristic(net, game):
    return net.predict(game.state())


def monte_carlo_value(net, game, playerMove, N=100):
    scores = [predict(net, game, playerMove) for i in range(0, N)]
    return np.mean(scores)


def get_best_move(net, game, samples=100):
    best_move = G.MOVE_UP  # default to up
    best_score = -1
    head = game.board['snakes'][0].body[0]
    # get locations around the head
    up = (head[0], head[1] - 1)
    down = (head[0], head[1] + 1)
    left = (head[0]-1, head[1])
    right = (head[0]+1, head[1])
    for i in range(4):
        # Check if this kills us, if it does, don't simulate it...
        if i == G.MOVE_UP and not game.valid_first_move(up):
            continue
        if i == G.MOVE_LEFT and not game.valid_first_move(left):
            continue
        if i == G.MOVE_RIGHT and not game.valid_first_move(right):
            continue
        if i == G.MOVE_DOWN and not game.valid_first_move(down):
            continue
        score = monte_carlo_value(net, game, i, N=samples)
        if score > best_score:
            best_score = score
            best_move = i

    return best_move, best_score


if __name__ == "__main__":
    with open("input.json") as f:
        payload = json.load(f)
        instance = G.Game(payload)
        net = model.Net("models/tanh.model")

        print(get_best_move(net, instance, 2))
