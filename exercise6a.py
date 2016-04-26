from rl.environments import Breakout
from rl import Simulation
from rl import util
import random

######
# Exercise: DQN to play Breakout... Work in progress

def run_exercise():
  run_name = util.generate_name()
  print "Starting run %s" % run_name
  print

  environment = Breakout()
  agent = DQN(environment)
  simulation = Simulation(environment, agent)

  for step in xrange(11):
    print "Starting episode %d" % step

    if step % 10 == 0:
      util.save_animation(simulation.episode_steps(), "videos/%s/%07d.gif" % (run_name, step))
    else:
      simulation.run_episode()

  util.open_videos_in_web_browser("videos/%s" % run_name)


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
