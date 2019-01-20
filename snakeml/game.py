import json
import itertools
import copy
import random
import numpy as np
import tensorflow as tf
import pprint
import os

pp = pprint.PrettyPrinter()

MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3


class Snake(object):
    def __init__(self, tag, health, body):
        self.id = tag
        self.dead = False
        self.health = health
        self.body = [(p['x'], p['y']) for p in body]

    def move(self, direction):
        self.health -= 1
        prefix = self.body[:-1]
        if direction == MOVE_UP:
            head = self.body[0]
            self.body = [(head[0], head[1]-1)] + prefix
            return
        if direction == MOVE_DOWN:
            head = self.body[0]
            self.body = [(head[0], head[1]+1)] + prefix
            return
        if direction == MOVE_LEFT:
            head = self.body[0]
            self.body = [(head[0]-1, head[1])] + prefix
            return
        if direction == MOVE_RIGHT:
            head = self.body[0]
            self.body = [(head[0]+1, head[1])] + prefix
            return

        print("ERROR NEVER GET HERE")


def generate_mutations(arr, game_size, max_size=19):
    output = []
    amount = max_size - game_size

    cur = np.roll(np.roll(arr, int(amount / 2), axis=0),
                  int(amount / 2), axis=1)

    for r in range(4):
        rotated = np.rot90(cur, r)
        output.append(rotated)

    return output


class Game(object):
    def __init__(self, payload):
        self.width = payload['board']['width']
        self.height = payload['board']['height']
        self.turn = payload['turn']
        self.history = []
        self.board = {
            "snakes": [],
            "food": []
        }

        self.board['snakes'].append(
            Snake(payload['you']['id'], payload['you']['health'], payload['you']['body']))

        for food in payload['board']['food']:
            self.board['food'].append((food['x'], food['y']))
        for snake in payload['board']['snakes']:
            if snake['id'] != payload['you']['id']:
                self.board['snakes'].append(
                    Snake(snake['id'], snake['health'], snake['body'])
                )

    def make_move(self, move):
        self.history.append(copy.deepcopy(self.board))
        self.turn += 1
        # Update our board
        for i in range(len(self.board['snakes'])):
            snake = self.board['snakes'][i]
            if snake.dead:
                continue

            # Apply the move to the snake
            snake.move(move[i])

        self.consume_food()
        self.kill_invalid_snakes()
        self.kill_starved_snakes()

    def valid_moves(self):
        used = set()
        for snake in self.board['snakes']:
            # Don't add the tail, we think it's a valid move
            for point in snake.body[:-1]:
                used.add(point)

        moves_for_snake = []
        for snake in self.board['snakes']:
            if snake.dead:
                moves_for_snake.append([-1])
                continue

            allowable = []
            # get the head
            head = snake.body[0]

            # get locations around the head
            up = (head[0], head[1] - 1)
            down = (head[0], head[1] + 1)
            left = (head[0]-1, head[1])
            right = (head[0]+1, head[1])

            if self.place_free(up, used):
                allowable.append(MOVE_UP)
            if self.place_free(down, used):
                allowable.append(MOVE_DOWN)
            if self.place_free(left, used):
                allowable.append(MOVE_LEFT)
            if self.place_free(right, used):
                allowable.append(MOVE_RIGHT)

            if len(allowable) == 0:
                allowable.append(MOVE_UP)

            moves_for_snake.append(allowable)

        # Use itertools to generate all possible moves
        return list(itertools.product(*moves_for_snake))

    def undo_move(self):
        self.turn -= 1
        self.board = self.history.pop()

    def place_free(self, point, used):
        if point in used:
            return False
        if point[0] < 0 or point[0] >= self.width:
            return False
        if point[1] < 0 or point[1] >= self.height:
            return False
        return True

    def valid_first_move(self, point):
        used = set()
        for snake in self.board['snakes']:
            if snake.dead:
                continue

            # skip the head and tail, we care about those
            for p in snake.body[1:-1]:
                used.add(p)

        return self.place_free(point, used)

    def consume_food(self):
        food = set()
        eaten = set()

        # Add the food
        for item in self.board['food']:
            food.add(item)

        # Grow the snakes and increase health
        for snake in self.board['snakes']:
            if snake.dead:
                continue
            head = snake.body[0]
            if head in food:
                eaten.add(head)
                snake.health = 100
                snake.body.append(snake.body[-1])

        leftover = food.difference(eaten)

        available = set()
        for i in range(self.width):
            for j in range(self.height):
                available.add((i, j))

        # Subtract existing food
        available = available.difference(leftover)

        snake_positions = set()

        for snake in self.board['snakes']:
            if snake.dead:
                continue
            for point in snake.body:
                snake_positions.add(point)

        available = available.difference(snake_positions)

        new_food_set = set(random.sample(available, len(eaten)))
        self.board['food'] = list(leftover.union(new_food_set))

    def kill_invalid_snakes(self):
        longest = dict()
        longestOwner = dict()
        bodies = set()
        for snake in self.board['snakes']:
            if snake.dead:
                continue

            head = snake.body[0]

            if head[0] < 0 or head[0] >= self.width:
                snake.dead = True
                snake.turn = self.turn

            if head[1] < 0 or head[1] >= self.height:
                snake.dead = True
                snake.turn = self.turn

            # if this head is new, just set it
            if head in longest:
                if len(snake.body) == longest[head]:
                    longest[head] = max(longest[head], len(snake.body))
                    longestOwner[head] = -1
                elif len(snake.body) > longest[head]:
                    longest[head] = max(longest[head], len(snake.body))
                    longestOwner[head] = snake.id
            else:
                longest[head] = len(snake.body)
                longestOwner[head] = snake.id

            for point in snake.body[1:]:
                bodies.add(point)

        # Now go through snakes and kill those in invalid
        for snake in self.board['snakes']:
            if snake.dead:
                continue

            head = snake.body[0]
            if head in bodies:
                snake.dead = True
                snake.turn = self.turn
            if head in longest:
                if longestOwner[head] != snake.id:
                    snake.turn = self.turn
                    snake.dead = True

        # dead snakes are dead

    def kill_starved_snakes(self):
        # TODO: Kill starved snakes
        for snake in self.board['snakes']:
            if not snake.dead and snake.health <= 0:
                snake.turn = self.turn
                snake.dead = True

    def over(self):
        alive = 0
        for snake in self.board['snakes']:
            if snake.dead:
                continue
            alive += 1

        return alive <= 1

    def score(self):
        alive = 0
        winner = -1
        scores = []
        greatest_death_turn = 0

        for i in range(len(self.board['snakes'])):
            snake = self.board['snakes'][i]
            if snake.dead:
                greatest_death_turn = max(greatest_death_turn, snake.turn)
                continue
            alive += 1
            winner = i

        for i in range(len(self.board['snakes'])):
            snake = self.board['snakes'][i]
            if winner == i:
                scores.append((i, 1))
            else:
                if winner == -1:
                    # tie, check if i'm longest
                    if greatest_death_turn == snake.turn:
                        scores.append((i, 0))
                    else:
                        scores.append((i, -1))
                else:
                    scores.append((i, -1))

        print("Scores:", scores)

        return scores

    def print(self):
        for snake in self.board['snakes']:
            if snake.dead:
                continue
            print(snake.body)
        print("Food")
        for food in self.board['food']:
            print(food)

    def state(self, player=0):
        output = np.zeros((19, 19, 3))
        for sId in range(len(self.board['snakes'])):
            snake = self.board['snakes'][sId]
            if snake.dead:
                continue
            for i in range(len(snake.body)):
                point = snake.body[i]
                if i == 0:
                    if sId == player:
                        # add the player head
                        output[point[0]][point[1]][2] = snake.health / 100.0
                    else:
                        # add the enemy head
                        output[point[0]][point[1]][1] = snake.health / 100.0
                else:
                    output[point[0]][point[1]][0] = 1.0

        for food in self.board['food']:
            output[food[0]][food[1]] = (.5, .5, .5)

        # set 1 outside
        i = self.width
        while i < 19:
            output[i, :, :] = np.ones((1, 3))
            output[:, i, :] = np.ones((1, 3))
            i += 1

        # Perform rotations
        outputs = generate_mutations(output, self.width, 19)

        return outputs

    def render(self):
        positions = {}
        for s in range(len(self.board['snakes'])):
            snake = self.board['snakes'][s]
            for i in range(len(snake.body)):
                point = snake.body[i]
                if i == 0:
                    positions[point] = str(s)
                else:
                    positions[point] = 's'

        for food in self.board['food']:
            positions[food] = "*"

        os.system('clear')
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                if (x, y) in positions:
                    line += positions[(x, y)]
                else:
                    line += ' '
            print(line)


def generate_items(players, board_size):
    foodcount = 4 + random.randrange(4)
    free_places = set()
    for i in range(board_size):
        for j in range(board_size):
            free_places.add((i, j))

    snakes = []
    for i in range(players):
        place = random.choice(list(free_places))
        bodypoint = {
            "x": place[0],
            "y": place[1]
        }
        free_places.remove(place)
        snakes.append({
            "id": "%d" % i,
            "body": [bodypoint, bodypoint, bodypoint],
            "health": 100
        })
    food = []
    for i in range(foodcount):
        place = random.choice(list(free_places))
        free_places.remove(place)
        food.append({
            "x": place[0],
            "y": place[1]
        })

    return (snakes, food)


def random_game(players=2, size=7):

    (snakes, food) = generate_items(players, size)
    payload = {
        "turn": 0,
        "board": {
            "height": size,
            "width": size,
            "snakes": snakes,
            "food": food
        },
        "you": random.choice(snakes)
    }

    instance = Game(payload)

    return instance
