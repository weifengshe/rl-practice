import numpy as np
import gym
import petname
import tensorflow as tf

######
# Exercise: Value function approximation
#
# Look at the code in the GradientSarsaAgent class below and implement
# the missing parts of ActionValueApproximator class.
#
# You can use whatever approximation you want. E.g. linear approximation
# with hand-crafted features or multi-layer neural network. You can
# also choose to use tensorflow or other libraries as you wish.
#
# The environment is described at
# https://gym.openai.com/envs/MountainCar-v0 .
def run_exercise():
  env_name = 'MountainCar-v0'
  run_name = petname.Generate(3, '-')
  print "Starting {} run {}".format(env_name, run_name)

  environment = gym.make(env_name)
  agent = GradientSarsaAgent(environment)

  print
  print "Starting training"
  environment.monitor.start('runs/{}/{}/'.format(env_name, run_name))
  run_episodes(environment, agent, episodes=10000000)


def run_episodes(environment, agent, episodes):
  for episode_index in xrange(episodes):
    observation = environment.reset()

    for step_index in xrange(environment.spec.timestep_limit):
      action = agent.choose_action(observation)

      new_observation, reward, done, info = environment.step(action)
      agent.learn(observation, action, reward, new_observation, done)

      if done:
        break


class GradientSarsaAgent(object):
  def __init__(self, environment, exploration=True):
    # The algorithm should theoretically work for any environment that has
    # continous observation space and discrete action space.
    assert type(environment.observation_space) is gym.spaces.Box
    assert type(environment.action_space) is gym.spaces.Discrete
    self.environment = environment

    # When False, we will use Greedy instead of EpsilonGreedy
    self.exploration = exploration

    self.observation_action_history = []

    self.sarsa_lambda = 0.9

    # How many times learn() has been called
    self.learning_step = 0

    self.approximator = ActionValueApproximator(
      self.environment.observation_space.shape,
      self.environment.action_space.n)

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
    self.observation_action_history.append((observation, action))

    current_value = self.approximator.evaluate(observation)[action]
    if done:
      td_error = reward - current_value
    else:
      new_value = self.approximator.evaluate(new_observation)[self.choose_action(new_observation)]
      td_error = reward + new_value - current_value

    for (index, (past_observation, past_action)) in enumerate(reversed(self.observation_action_history)):
      past_value = self.approximator.evaluate(past_observation)[past_action]
      target = past_value + td_error * (self.sarsa_lambda**index)
      self.approximator.update(past_observation, past_action, target)

    if done:
      self.observation_action_history = []

  def epsilon(self, t):
    return 0.1


class ActionValueApproximator(object):
  def __init__(self, observation_shape, n_available_actions):
    self.build_computation_graph(observation_shape, n_available_actions)
    self.session = tf.Session()
    self.session.run(self.initialize)

  def build_computation_graph(self, observation_shape, n_available_actions):
    assert observation_shape == (2,)
    assert n_available_actions == 3
    self.observation = tf.placeholder(tf.float32, [1, 2], name="observation")
    self.action = tf.placeholder(tf.int64, [1], name="action")
    self.target = tf.placeholder(tf.float32, [1], name="target")

    self.hidden = tf.nn.relu(affine_transformation(self.observation, 2))
    assert shape(self.hidden) == (1, 2)

    self.estimates = affine_transformation(self.hidden, 3)
    assert shape(self.estimates) == (1, 3)

    self.loss = self.build_loss(self.estimates, self.action, self.target, n_available_actions)
    assert shape(self.loss) == ()

    self.optimize = tf.train.AdamOptimizer().minimize(self.loss)

    self.initialize = tf.initialize_all_variables()

  def build_loss(self, estimates, action, target, n_available_actions):
    assert n_available_actions == 3
    assert shape(estimates) == (1, 3)
    assert shape(target) == (1,)
    assert shape(action) == (1,)
    assert action.dtype == np.int64

    action_mask = tf.one_hot(action, depth=3, on_value=1.0, off_value=0.0)
    assert shape(action_mask) == (1, 3)

    target_expanded = tf.expand_dims(target, 1)
    assert shape(target_expanded) == (1, 1)

    targets = tf.tile(target_expanded, [1, 3])
    assert shape(targets) == (1, 3)

    difference = action_mask * (estimates - targets)
    assert shape(difference) == (1, 3)

    loss = tf.nn.l2_loss(difference)
    assert shape(loss) == ()

    loss.action_mask = action_mask
    loss.targets = targets
    loss.difference = difference
    return loss

  # Estimates the action-values of the given observation.
  # Returns an array of floats: one value for each action.
  def evaluate(self, observation):
    return self.session.run(self.estimates,
      {self.observation: [observation]})[0]

  # Estimates the action-values of the given observation.
  # Returns an array of floats: one value for each action.
  def update(self, observation, action, target):
    return self.session.run(self.optimize, {
        self.observation: [observation],
        self.action: [action],
        self.target: [target]})

def affine_transformation(tensor, output_width):
  assert len(shape(tensor)) == 2
  input_width = shape(tensor)[1]

  W = tf.Variable(tf.random_normal([input_width, output_width], stddev=0.05), tf.float32, name="W")
  b = tf.Variable(tf.random_normal([output_width], stddev=0.05), tf.float32, name="b")
  output = tf.matmul(tensor, W) + b
  output.W = W
  output.b = b

  return output

def shape(tensor):
  assert type(tensor) is tf.Tensor, "tensor is not Tensor: %r" % tensor
  return tuple([dimension.value for dimension in tensor.get_shape()])

run_exercise()
