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

The numbering of exercises follows David Silver's Reinforcement Learning lectures.

### Lecture 3: Planning by Dynamic Programming

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/DP.pdf) from David Silver's course.

- [Exercise 3A](exercise3a.md): Iterative policy evaluation
- [Exercise 3B](exercise3b.md): Policy iteration
- [Exercise 3C](exercise3c.md): Value iteration

### Lecture 4: Model-Free Prediction

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf) from David Silver's course.

- [Exercise 4A](exercise4a.md): Monte-carlo policy evaluation
- [Exercise 4B](exercise4b.md): TD evaluation

### Lecture 5: Model-Free Control

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf) from David Silver's course.

- [Exercise 5A](exercise5a.md): TD control with afterstates
- [Exercise 5B](exercise5b.md): Sarsa
- [Exercise 5C](exercise5c.md): Q-learning
