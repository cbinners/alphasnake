import json
import itertools
import copy
import random

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

        for snake in payload['board']['snakes']:
            if snake['id'] != payload['you']['id']:
                self.board['snakes'].append(
                    Snake(snake['id'], snake['health'], snake['body'])
                )

    def over(self):
        return len(self.board["snakes"] <= 1)

    def make_move(self, move):
        if random.randint(0, 100) == 50:
            print(len(self.history))
            if len(self.history) > 2000:
                print(self.board['snakes'][0].body)
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
        return moves_for_snake[0], list(itertools.product(*moves_for_snake[1:]))

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

        for i in range(len(eaten)):
            new_food = random.choice(available)
            self.board['food'].append(new_food)

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

            if head[1] < 0 or head[1] >= self.height:
                snake.dead = True

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
            if head in longest:
                if longestOwner[head] != snake.id:
                    snake.dead = True

        # dead snakes are dead

    def kill_starved_snakes(self):
        # TODO: Kill starved snakes
        for snake in self.board['snakes']:
            if not snake.dead and snake.health <= 0:
                snake.dead = True

    def over(self):
        alive = 0
        for snake in self.board['snakes']:
            if snake.dead:
                continue
            alive += 1

        return alive <= 1

    def score(self):
        if self.board['snakes'][0].dead:
            return 0
        return 1

    def print(self):
        print(self.turn, self.board['snakes'], self.board['food'])

    def state(self):
        # TODO: Return state for tensorflow
        pass
