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
- TBD: Policy iteration
- [exercise3c.py](exercise3c.py): Value iteration

### Lecture 4: Model-Free Prediction

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf) from David Silver's course.

- TBD: Monte-carlo policy evaluation
- [exercise4b.py](exercise4b.py): TD policy evaluation

### Lecture 5: Model-Free Control

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf) from David Silver's course.

- [exercise5a.py](exercise5a.py): TD control with afterstates
- [exercise5b.py](exercise5b.py): Sarsa
- [exercise5c.py](exercise5c.py): Q-learning
