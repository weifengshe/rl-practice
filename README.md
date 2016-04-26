# Programming exercises for Reinforcement Learning

This is a set of exercises on reinforcement learning for our study group at Aalto University.

## Preparation

Clone this repository from github with

```bash
git clone https://github.com/tarvaina/rl-practice.git
```

Install python libraries with

```bash
pip install numpy petname imageio
```

For exercise that use Atari 2600 games, see [installation instructions for Arcade Learning Environment](install_ale.md).

## Running the exercises

Run the exercise file with python. E.g. `python exercise5a.py`. You will see output similar to this:

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

Edit the exercise file to implement the missing parts and run the program
again. Repeat until you are happy with the results.

## Exercises

The numbering of exercises follows the lectures of [David Silver's Reinforcement Learning course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

### Lecture 3: Planning by Dynamic Programming

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf) from David Silver's course.

- [exercise3a.py](exercise3a.py): Iterative policy evaluation
- [exercise3b.py](exercise3b.py): Value iteration

### Lecture 4: Model-Free Prediction

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf) from David Silver's course.

- [exercise4a.py](exercise4a.py): TD policy evaluation

### Lecture 5: Model-Free Control

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf) from David Silver's course.

- [exercise5a.py](exercise5a.py): TD control with afterstates
- [exercise5b.py](exercise5b.py): Sarsa
- [exercise5c.py](exercise5c.py): Q-learning

### Lecture 6: Value Function Approximation

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/FA.pdf) from David Silver's course.

- (Work in progress: [exercise6a.py](exercise6a.py): DQN learning with Atari games)
