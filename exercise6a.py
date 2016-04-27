from rl.environments import Breakout
from rl import Simulation
from rl import util
import random
import tensorflow as tf
from tensorflow.contrib import skflow
import numpy as np
from collections import deque

######
# Exercise: DQN to play Breakout... Work in progress

def run_exercise():
  run_name = util.generate_name()
  print "Starting run %s" % run_name
  print

  environment = Breakout()
  agent = DQN(environment, run_name)
  simulation = Simulation(environment, agent)

  for step in xrange(11):
    print
    print "Starting episode %d" % step

    if step % 1 == 0:
      util.save_animation(simulation.episode_steps(), "videos/%s/%07d.gif" % (run_name, step))
    else:
      simulation.run_episode()

  util.open_videos_in_web_browser("videos/%s" % run_name)


class DQN(object):
  # target_q_max_age = 5000 # 10000 in DeepMind paper
  replay_memory_size = 50000 # 1000000 in DeepMind paper
  replay_sample_size = 32 # 32 in DeepMind paper
  discount_factor = 0.9 # 0.99 in DeepMind paper

  def __init__(self, environment, run_name, exploration=True):
    self.actions = list(environment.actions)
    self.action_values = ConvNet(len(self.actions), run_name=run_name)
    self.learning_step = 1
    self.exploration = exploration
    self.replay_memory = deque()

  def start_episode(self):
    pass

  def choose_action(self, state):
    action_values = self.action_values.estimate(np.array(state))
    if self.learning_step % 10 == 0:
      print "Step %d action values: %s" % (self.learning_step, repr(action_values))

    if self.exploration and random.random() > self.epsilon(self.learning_step):
      return max(self.actions, key=lambda action: action_values[self.actions.index(action)])
    else:
      return random.choice(self.actions)

  def learn(self, state, action, reward, new_state):
    self.learning_step += 1

    # TODO: Update target_q

    if reward != 0:
      print "Step %d reward: %d" % (self.learning_step, reward)

    self.replay_memory.append((state, action, reward, new_state))

    if len(self.replay_memory) >= self.replay_sample_size:
      replay_sample = random.sample(self.replay_memory, self.replay_sample_size)

      states = np.array([state for (state, _, _, _) in replay_sample])
      rewards = np.array([reward for (_, _, reward, _) in replay_sample])
      new_states = np.array([state for (_, _, _, new_state) in replay_sample])
      new_values = np.array([max(estimate) for estimate in self.action_values.estimates(new_states)])
      action_idxs = np.array([self.actions.index(action) for (_, action, _, _) in replay_sample])
      targets = np.array([reward + self.discount_factor * new_value
          for reward, new_value in zip(rewards, new_values)])

      # TODO: Clip change to [-1, +1]
      self.action_values.update(states, action_idxs, targets)

    if len(self.replay_memory) > self.replay_memory_size:
      self.replay_memory.popleft()

  def epsilon(self, k):
    return 0.1


class ConvNet(object):
  screenshot_dimensions = (210, 160, 3)
  consecutive_screenshots = 2

  def __init__(self, n_actions, run_name):
    self.placeholders = self.build_placeholders(n_actions)
    assert shapes(self.placeholders.screenshots) == (None, 2, 210, 160, 3)
    assert shape(self.placeholders.targets) == (None,)
    assert shape(self.placeholders.action_idxs) == (None,)

    preprocessed = self.build_preprocessing(self.placeholders.screenshots)
    assert shape(preprocessed) == (None, 32, 32, 2)
    self.preprocessed = preprocessed

    conv1 = self.build_conv_layer(preprocessed, 8, 'conv1')
    assert shape(conv1) == (None, 16, 16, 8)

    conv2 = self.build_conv_layer(conv1, 8, 'conv2')
    assert shape(conv2) == (None, 8, 8, 8)

    conv3 = self.build_conv_layer(conv2, 8, 'conv3')
    assert shape(conv3) == (None, 4, 4, 8)

    flat = tf.reshape(conv3, [-1, 4 * 4 * 8])
    assert shape(flat) == (None, 128)

    self.estimators = self.build_linear_estimator(flat, n_actions)
    assert shape(self.estimators) ==  (None, 4)

    loss = self.build_loss(self.estimators, self.placeholders.action_idxs, self.placeholders.targets, n_actions)
    assert shape(loss) == ()

    self.optimizer = self.build_optimizer(loss)
    assert type(self.optimizer) is tf.Operation

    initialize = tf.initialize_all_variables()

    self.session = tf.Session()
    self.session.run(initialize)

  def estimate(self, screenshots):
    assert screenshots.shape == (2, 210, 160, 3), screenshots.shape
    estimates = self.__estimates([screenshots])
    assert estimates.shape == (1, 4), estimates
    return estimates[0]

  def estimates(self, screenshots):
    assert screenshots.shape == (32, 2, 210, 160, 3), screenshots.shape
    estimates = self.__estimates(screenshots)
    assert estimates.shape == (32, 4), estimates
    return estimates

  def __estimates(self, screenshots):
    feed_dict = {self.placeholders.screenshots: screenshots}
    return self.session.run(self.estimators, feed_dict)

  def update(self, screenshots, action_idxs, targets):
    assert screenshots.shape == (32, 2, 210, 160, 3)
    assert action_idxs.shape == (32,)
    assert targets.shape == (32,)

    feed_dict = {
      self.placeholders.screenshots: screenshots,
      self.placeholders.action_idxs: action_idxs,
      self.placeholders.targets: targets
    }

    self.session.run(self.optimizer, feed_dict)

  def build_placeholders(self, n_actions):
    input_tensor_dimensions = [None, self.consecutive_screenshots] + list(self.screenshot_dimensions)
    screenshots = tf.placeholder(tf.float32, input_tensor_dimensions, name="screenshots")
    assert shapes(screenshots) == (None, 2, 210, 160, 3)

    targets = tf.placeholder(tf.float32, [None], name="targets")
    assert shape(targets) == (None,)

    action_idxs = tf.placeholder(tf.int64, [None], name="action_idxs")
    assert shape(action_idxs) == (None,)

    class Placeholders(object):
      pass
    placeholders = Placeholders()
    placeholders.screenshots = screenshots
    placeholders.targets = targets
    placeholders.action_idxs = action_idxs

    return placeholders

  def build_preprocessing(self, screenshots):
    assert shapes(screenshots) == (None, 2, 210, 160, 3)

    grays = tf.image.rgb_to_grayscale(screenshots)
    assert shapes(grays) == (None, 2, 210, 160, 1)

    grays_combined = tf.transpose(tf.squeeze(grays, [4]), perm=[0, 2, 3, 1])
    assert shape(grays_combined) == (None, 210, 160, 2), grays_combined

    resized = tf.image.resize_bicubic(grays_combined, (32, 32))
    assert shape(resized) == (None, 32, 32, 2)

    result = (resized - 128.0) / 128.0
    assert shape(result) == (None, 32, 32, 2)

    result.grays = grays
    result.grays_combined = grays_combined
    result.resized = resized
    return result

  def build_conv_layer(self, input, n_filters, name):
    with tf.variable_scope(name):
      h = skflow.ops.conv2d(input, n_filters=n_filters, filter_shape=[3, 3],
          bias=True, activation=tf.nn.relu)
      maxpool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')
      maxpool.h = h
      return maxpool

  def build_linear_estimator(self, input, n_actions):
    with tf.variable_scope('estimate'):
      input_width= shape(input)[-1]
      W = tf.Variable(tf.random_normal([input_width, n_actions], stddev=0.05), tf.float32, name="W")
      b = tf.Variable(tf.random_normal([n_actions], stddev=0.05), tf.float32, name="b")
      result = tf.matmul(input, W) + b
      assert shape(result) == (None, 4)

      result.W = W
      result.b = b
      return result

  def build_loss(self, estimates, action_idxs, targets, n_actions):
    assert shape(estimates) == (None, 4)
    assert shape(targets) == (None,)
    assert shape(action_idxs) == (None,)
    assert action_idxs.dtype == np.int64

    action_mask = tf.one_hot(action_idxs, depth=n_actions, on_value=1.0, off_value=0.0)
    assert shape(action_mask) == (None, 4)

    targets = tf.expand_dims(targets, 1)
    assert shape(targets) == (None, 1)

    targets = tf.tile(targets, [1, n_actions])
    assert shape(targets) == (None, 4)

    difference = action_mask * (estimates - targets)
    assert shape(difference) == (None, 4)

    loss = tf.nn.l2_loss(difference)
    assert shape(loss) == ()

    return loss

  def build_optimizer(self, loss):
    assert shape(loss) == ()
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(loss)

def shapes(obj):
  def is_tensor_or_collection(t):
    if type(t) is list or type(t) is tuple:
      return all(map(is_tensor_or_collection, t))
    elif type(t) is dict:
      return all(map(is_tensor_or_collection, t.values()))
    else:
      return type(t) is tf.Tensor

  def _shapes(obj):
    if type(obj) is list:
      return map(_shapes, obj)
    if type(obj) is tuple:
      return tuple(map(_shapes, obj))
    elif type(obj) is dict:
      return { k: _shapes(obj[k]) for k in obj }
    else:
      return shape(obj)

  assert is_tensor_or_collection(obj)
  return _shapes(obj)


def shape(tensor):
  assert type(tensor) is tf.Tensor, "tensor is not Tensor: %r" % tensor
  return tuple([dimension.value for dimension in tensor.get_shape()])

run_exercise()


