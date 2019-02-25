# alphasnake

An initial attempt at building an unsupervised ML snake based on alphazero

See https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/ for an explanation of what's going on.


## How to use

`python algo.py`

## Requirements

Requires tensorflow. I recommend using a GPU to train this.

## READ THIS

The current code will simulate games and try to build up a NN representation of `Heuristic(X)`. I suspect that the state space is much larger than Go, and as battlesnake is not a "classical game", I suspect you may never get convergence. I trained a model and loaded it for use in a Golang program, but execution times were much too high.

## TODO

The simulation environment here uses the old food rules and has the initial 2019 tail rule (not 2018).
