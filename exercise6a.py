from rl.environments import Breakout
from rl import Simulation
from rl import util
import random
import imageio
import numpy as np
import petname
import os
import fnmatch
import webbrowser

######
# Exercise: DQN to play Breakout... Work in progress

def save_animation(episode, filename):
  (images, actions, rewards) = zip(*episode)

  directory = os.path.dirname(filename)
  if not os.path.exists(directory):
      os.makedirs(directory)

  imageio.mimwrite(filename, images, fps=50)
  print "Saved video to %s" % filename

def write_html_index(directory):
  files = [file for file in os.listdir(directory) if fnmatch.fnmatch(file, '*.gif')]
  links = '\n'.join('<h2>%s</h2><div><img src="%s"></div>' % (file, file) for file in files)
  html = "<html><head><title>%s</title><body>%s</body></html>" % (directory, links)

  html_file_name = "%s/index.html" % directory
  with open(html_file_name, "w") as html_file:
    html_file.write(html)

  return html_file_name

def run_exercise():
  run_name = petname.Generate(3, '-')
  print "Starting run %s" % run_name
  print

  environment = Breakout()
  agent = DQN(environment)
  simulation = Simulation(environment, agent)

  for step in xrange(101):
    print "Starting episode %d" % step
    episode = simulation.run_episode()

    if step % 10 == 0:
      save_animation(episode, "gifs/%s/%07d.gif" % (run_name, step))

  html_file_name = write_html_index("gifs/%s" % run_name)
  webbrowser.open("file://" + os.path.abspath(html_file_name))


class DQN(object):
  def __init__(self, environment, exploration=True):
    # List of all available actions.
    self.environment = environment
    self.exploration = exploration

  def start_episode(self):
    self.episode_step = 0

  def choose_action(self, state):
    # Keep these two lines unchanged.
    self.episode_step += 1
    epsilon = self.epsilon(self.episode_step)

    #### TODO:
    # Change this function to implement an epsilon-greedy policy using
    #
    # - self.state_action_values[state][action] to get the
    #   state-action value estimates
    # - epsilon for the current epsilon value
    #
    # You can assume that discount factor is 1.
    return random.choice(self.environment.actions)

  def learn(self, state, action, reward, new_state):
    #### TODO:
    # Change this function to implement DQN-learning using
    #
    # - self.state_action_values[state][action] to get and
    #   update the state-action value estimates
    # - self.learning_rate for alpha value
    #
    # You can assume that discount factor is 1.
    pass

  def epsilon(self, k):
    if self.exploration:
      return 1.0 / k
      ### Alternative schedules to try:
      # return 0.1
      # return k**(-0.75)
    else:
      return 0


run_exercise()
