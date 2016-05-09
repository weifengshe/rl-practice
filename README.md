# Programming exercises for Reinforcement Learning

This is a set of exercises on reinforcement learning for our study group at Aalto University.

## Preparation

Clone this repository from github with

```bash
git clone https://github.com/tarvaina/rl-practice.git
```

Install dependencies with

```bash
### OS X
brew install cmake ffmpeg
sudo easy_install --upgrade six
sudo pip install --ignore-installed pip
sudo pip install numpy petname gym[classic-control]
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl
```

```bash
### Ubuntu
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl ffmpeg
pip install --ignore-installed pip
pip install numpy petname gym[classic-control]
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
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
- [exercise3b.py](exercise3b.py): Value iteration

### Lecture 4: Model-Free Prediction

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf) from David Silver's course.

- [exercise4a.py](exercise4a.py): TD policy evaluation

### Lecture 5: Model-Free Control

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf) from David Silver's course.

- [exercise5a.py](exercise5a.py): TD control with afterstates
- [exercise5b.py](exercise5b.py): Sarsa
- [exercise5c.py](exercise5c.py): Q-learning

### Lecture 5: Value Function Approximation

Corresponding [slides](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/FA.pdf) from David Silver's course.

This exercise uses [OpenAI Gym](https://gym.openai.com/). See the installation instructions at the top of this page.

- [exercise6a.py](exercise6a.py): Improving Gradient Sarsa
