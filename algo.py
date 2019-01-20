import json
import tensorflow as tf
import numpy as np
from snakeml import model, game as G
import random
import sys

win_scores = []
loss_scores = []
draw_scores = []


def predict(net, game, playerMove, is_training=True):
    if game.over():
        scores = game.score()
        if is_training:
            record(net, game, scores)
            apply_updates()
        return scores[0][1]

    moves = game.valid_moves()
    worst_move = [playerMove]

    moves_for_enemies = []
    # go through each enemy and find the best move for each individual, then construct the final move,
    # which consists of everybody playing best for them.
    for enemy in range(len(game.board['snakes'])-1):
        enemy_index = enemy + 1
        # we want the best for enemy_index
        score_for_move = {0: 10, 1: 10, 2: 10, 3: 10}
        for i in range(len(moves)):
            move = moves[i]
            if move[0] != playerMove:
                continue
            move_to_execute = [playerMove]

            # Add the remaining enemy_moves for the bots
            for i in range(len(game.board['snakes'])-1):
                move_to_execute.append(move[i])

            game.make_move(move_to_execute)
            score = heuristic(net, game, enemy_index)
            game.undo_move()

            move_index = move[enemy_index]
            # Check if our score is worse
            if score < score_for_move[move_index]:
                score_for_move[move_index] = score

        # we want to take the MAX of the scores
        best = -1
        best_direction = 0
        for i in range(4):
            if score_for_move[i] > best and score_for_move[i] <= 1:
                best = score_for_move[i]
                best_direction = i
        # Add this best move for the enemy
        moves_for_enemies.append(best_direction)

    worst_move += moves_for_enemies
    # Actually apply the move and continue
    game.make_move(tuple(worst_move))
    scores = simulate_move(net, game, is_training)
    game.undo_move()

    # If we're training, then update the neural net
    if is_training:
        record(net, game, scores)
        apply_updates()

    return scores[0][1]


def simulate_move(net, game, is_training):
    done = False
    records = 1
    snake_scores = None
    while not done:
        if game.over():
            snake_scores = game.score()
            if is_training:
                record(net, game, snake_scores)
            done = True
            records -= 1
            break

        moves = game.valid_moves()
        move_to_apply = []

        # Get the optimal move for each player
        for player in range(len(game.board['snakes'])):
            snake = game.board['snakes'][player]
            if snake.dead:
                # Default to a 0 move
                move_to_apply.append(0)
                continue

            # we want the best for enemy_index, initialize this out of bounds
            score_for_move = {0: 2, 1: 2, 2: 2, 3: 2}
            for i in range(len(moves)):
                move = moves[i]
                game.make_move(move)
                score = heuristic(net, game, player)
                game.undo_move()

                move_index = move[player]
                # minimize the scores
                if score < score_for_move[move_index]:
                    score_for_move[move_index] = score

            # we want to take the MAX of the scores
            best = -1
            best_direction = 0
            # Take the highest of the low
            for i in range(4):
                if score_for_move[i] > best and score_for_move[i] <= 1:
                    best = score_for_move[i]
                    best_direction = i
            # Add this best move for the player
            move_to_apply.append(best_direction)

        # Actually apply the move and continue
        game.make_move(move_to_apply)
        records += 1

    # Undo stack, add the inputs for each snake
    for i in range(records):
        # If we're training, update the model
        if is_training:
            for i in range(len(game.board['snakes'])):
                snake = game.board['snakes'][i]
                if snake.dead:
                    continue
                # Update the net for each snake in this position
                record(net, game, snake_scores)

        game.undo_move()

    return snake_scores


def record(net, game, snake_scores):
    # build up the records
    for (i, score) in snake_scores:
        state = game.state(i)
        if score == 1:
            win_scores.append((state, score))
        if score == -1:
            loss_scores.append((state, score))
        if score == 0:
            draw_scores.append((state, score))


def apply_updates():
    wins = len(win_scores)
    losses = len(loss_scores)

    # Make sure the win and loss set are equal size
    truncate_size = min(wins, losses)
    truncated_wins = win_scores[:truncate_size]
    truncated_losses = loss_scores[:truncate_size]

    # Construct the final training set, add all draws in (net 0)
    training = truncated_wins+truncated_losses+draw_scores

    # Shuffle the training data, this shuffles in place
    random.shuffle(training)

    # Gather our training examples and construct the label tensor
    ys = []
    x = []

    for example in training:
        y = np.full((len(example[0]), 1), example[1])
        ys.append(y)
        x.append(example[0])

    # Stack the data for the neural net
    X = np.vstack(x)
    Y = np.vstack(ys)

    # Fit the model
    net.update(X, Y)

    # Clear the records so we don't keep lingering data.
    win_scores.clear()
    loss_scores.clear()
    draw_scores.clear()


def heuristic(net, game, snake_id):
    # Add base case for death
    if game.board['snakes'][snake_id].dead:
        return -1
    return net.predict(game.state(snake_id))


def monte_carlo_value(net, game, playerMove, N=100):
    scores = [predict(net, game, playerMove) for i in range(N)]
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
    while True:
        instance = G.random_game(2, 7)
        net = model.Net("models/derpa.model")
        print(get_best_move(net, instance, 5))
