import numpy as np
import gym
import petname

######
# Exercise: Value function approximation
#
# Look at the code in the GradientQAgent class below and implement
# the missing parts of ActionValueApproximator class.
#
# The environment is described at
# https://gym.openai.com/envs/MountainCar-v0 .
def run_exercise():
  env_name = 'MountainCar-v0'
  run_name = petname.Generate(3, '-')
  print "Starting {} run {}".format(env_name, run_name)

  environment = gym.make(env_name)
  agent = GradientQAgent(environment)

  print
  print "Starting training"
  environment.monitor.start('runs/{}/{}/'.format(env_name, run_name))
  run_episodes(environment, agent, episodes=63)

  print
  print "Starting testing"
  agent.exploration = False
  run_episodes(environment, agent, episodes=1)
  environment.monitor.close()


def run_episodes(environment, agent, episodes):

  for episode_index in xrange(episodes):
    observation = environment.reset()

    for step_index in xrange(environment.spec.timestep_limit):

      action = agent.choose_action(observation)

      new_observation, reward, done, info = environment.step(action)
      agent.learn(observation, action, reward, new_observation, done)

      if done:
        break


class GradientQAgent(object):
  def __init__(self, environment, exploration=True):
    # The algorithm should theoretically work for any environment that has
    # continous observation space and discrete action space.
    assert type(environment.observation_space) is gym.spaces.Box
    assert type(environment.action_space) is gym.spaces.Discrete
    self.environment = environment

    # When False, we will use Greedy instead of EpsilonGreedy
    self.exploration = exploration

    # Tweak learning rate according to your taste
    self.learning_rate = 0.1

    # How many times learn() has been called
    self.learning_step = 0

    self.approximator = ActionValueApproximator(self.environment.action_space.n)

  def choose_action(self, observation):
    if self.exploration:
      return self.epsilon_greedy_action(observation)
    else:
      return self.greedy_action(observation)

  def epsilon_greedy_action(self, observation):
    if np.random.random() > self.epsilon(self.learning_step + 1):
      return self.environment.action_space.sample()
    else:
      return self.greedy_action(observation)

  def greedy_action(self, observation):
    return np.argmax(self.approximator.evaluate(observation))

  def learn(self, observation, action, reward, new_observation, done):
    self.learning_step += 1

    if done:
      target = reward
    else:
      target = reward + np.max(self.approximator.evaluate(new_observation))

    self.approximator.update(observation, action, target)

  def epsilon(self, t):
    return 1.0 / t


class ActionValueApproximator(object):
  def __init__(self, n_available_actions):
    self.n_available_actions = n_available_actions

  # Estimates the action-values of the given observation.
  # Returns an array of floats: one value for each action.
  def evaluate(self, observation):
    return np.random.random_sample(self.n_available_actions)

  # Estimates the action-values of the given observation.
  # Returns an array of floats: one value for each action.
  def update(self, observation, action, target):
    pass


run_exercise()
