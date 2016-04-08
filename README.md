# Programming exercises for Reinforcement Learning

This is a set of exercises on reinforcement learning for our study group at Aalto University.

## Preparation

Clone this repository from github with

```bash
git clone git@github.com:tarvaina/rl-practice.git
```

Install numpy with

```bash
pip install numpy
```

## Environment description

The cliff world looks like this:

```
+-------------+
|.............|
|.............|
|.............|
|SxxxxxxxxxxxE|
+-------------+
```

where:

- `S` is the start state,
- `E` is the end state,
- `x` is a canyon
- `|` and `-` are walls

Agent can walk to any of the four main directions on any state. Walking to a wall retains the current state. Walking to a canyon causes reward -100 and the agent to teleport to the start state. All other steps cause reward -1.

## Exercise for lecture 5: TD-learning for control

Run the program with `python main.py` from the root directory. You will see output similar to this:

```
State value estimates
[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

Sampled actions on each state
[['d' 'l' 'r' 'r' 'u' 'u' 'u' 'u' 'd' 'u' 'r' 'l']
 ['l' 'l' 'd' 'u' 'd' 'l' 'l' 'd' 'd' 'r' 'd' 'd']
 ['r' 'r' 'r' 'l' 'd' 'r' 'l' 'l' 'd' 'u' 'u' 'u']
 ['r' 'l' 'l' 'l' 'r' 'l' 'u' 'r' 'l' 'l' 'l' 'u']]
```

Edit the file [rl/agents/td_agent.py](rl/agents/td_agent.py) and implement

- TD(0) or TD(λ) learning in the `learn` method.
- 𝜀-greedy policy in the `choose_action` method.

See if you can teach the agent to find the optimal route.
