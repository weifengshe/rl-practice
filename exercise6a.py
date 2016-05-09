import numpy as np
import gym
import petname
import tensorflow as tf

######
# Exercise: Value function approximation
#
# You can see an inefficient Sarsa algorithm for the mountain car
# environment, based on the description at Sutton-Barto book
# chapter 8.4 Control with Function Approximation:
# https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node89.html
#
# Your task is to improve the algorithm. Write a more efficient
# implementation either in terms of computation time, the
# number of steps it takes to converge, or a more elegant solution.
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
  run_episodes(environment, agent, episodes=100)


def run_episodes(environment, agent, episodes):
  for episode_index in xrange(episodes):
    observation = environment.reset()

    for step_index in xrange(environment.spec.timestep_limit):
      action = agent.choose_action(observation)

      new_observation, reward, done, info = environment.step(action)
      agent.learn(observation, action, reward, new_observation, done)

      observation = new_observation

      if done:
        break


class GradientSarsaAgent(object):
  def __init__(self, environment):
    # The algorithm should theoretically work for any environment that has
    # continous observation space and discrete action space.
    assert type(environment.observation_space) is gym.spaces.Box
    assert type(environment.action_space) is gym.spaces.Discrete
    self.environment = environment

    self.observation_action_history = []

    self.sarsa_lambda = 0.9

    self.learning_step = 0

    self.grid_size = 9
    self.grid_count = 10
    grid_spans = zip(self.environment.observation_space.low, self.environment.observation_space.high)
    assert np.array(grid_spans).shape == (2, 2)

    self.tile_sizes = np.array([(high - low) / (self.grid_size - 1) for low, high in grid_spans])
    assert self.tile_sizes.shape == (2,), self.tile_sizes.shape

    self.tiling_offsets = np.array([
      [np.random.uniform(-0.5 * tile_size, 0.5 * tile_size) for tile_size in self.tile_sizes]
      for _ in xrange(10)])
    assert self.tiling_offsets.shape == (10, 2)

    self.approximator = ActionValueApproximator(
      2 * self.grid_count,
      self.grid_size,
      self.environment.action_space.n)

  def choose_action(self, observation):
    return np.argmax(self.evaluate(observation))

  def features(self, observation):
    def tile_coordinates(offsets):
      assert observation.shape == offsets.shape == self.tile_sizes.shape == (2,)
      tile_coordinates = ((observation - offsets - self.environment.observation_space.low) / self.tile_sizes).astype(int)
      assert tile_coordinates.shape == (2,)
      assert all(0 <= coordinate < self.grid_size for coordinate in tile_coordinates), tile_coordinates
      return tile_coordinates

    features = np.array([tile_coordinates(offset) for offset in self.tiling_offsets]).flatten()
    assert features.shape == (20,)
    return features

  def evaluate(self, observation):
    features = self.features(observation)
    return self.approximator.evaluate(features)

  def update(self, observation, action, target):
    assert observation.shape == (2,)
    features = self.features(observation)
    assert features.shape == (20,)
    self.approximator.update(features, action, target)

  def learn(self, observation, action, reward, new_observation, done):
    self.learning_step += 1
    self.observation_action_history.append((observation, action))

    current_value = self.evaluate(observation)[action]
    if done:
      td_error = reward - current_value
    else:
      new_value = self.evaluate(new_observation)[self.choose_action(new_observation)]
      td_error = reward + new_value - current_value

    for (index, (past_observation, past_action)) in enumerate(reversed(self.observation_action_history[-20:])):
      past_value = self.evaluate(past_observation)[past_action]
      target = past_value + td_error * (self.sarsa_lambda**index)
      self.update(past_observation, past_action, target)

    if done:
      self.observation_action_history = []

  def epsilon(self, t):
    return 0.0


class ActionValueApproximator(object):
  def __init__(self, n_features, feature_classes, n_available_actions):
    self.build_computation_graph(n_features, feature_classes, n_available_actions)
    self.session = tf.Session()
    self.session.run(self.initialize)

  def build_computation_graph(self, n_features, feature_classes, n_available_actions):
    assert n_features == 20, n_features
    assert n_available_actions == 3
    assert feature_classes == 9
    self.features = tf.placeholder(tf.int64, [1, 20], name="features")
    self.action = tf.placeholder(tf.int64, [1], name="action")
    self.target = tf.placeholder(tf.float32, [1], name="target")

    self.one_hot = tf.one_hot(self.features, feature_classes, 1.0, 0.0)
    assert shape(self.one_hot) == (1, 20, 9), self.one_hot

    self.one_hot_flattened = tf.reshape(self.one_hot, [1, 180])
    assert shape(self.one_hot_flattened) == (1, 180), self.one_hot_flattened

    self.estimates = affine_transformation(self.one_hot_flattened, 3)
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

  # Estimates the action-values of the given features.
  # Returns an array of floats: one value for each action.
  def evaluate(self, features):
    return self.session.run(self.estimates,
      {self.features: [features]})[0]

  # Estimates the action-values of the given features.
  # Returns an array of floats: one value for each action.
  def update(self, features, action, target):
    assert features.shape == (20,), features
    return self.session.run(self.optimize, {
        self.features: [features],
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
