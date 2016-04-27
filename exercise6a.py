from rl.environments import Breakout
from rl import Simulation
from rl import util
import random
import os
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

  for step in xrange(1001):
    print
    print "Starting episode %d" % step

    if step % 10 == 0:
      util.save_animation(simulation.episode_steps(), "runs/%s/videos/%07d.gif" % (run_name, step))
    else:
      simulation.run_episode()

  util.open_videos_in_web_browser("runs/%s/videos/" % run_name)


class DQN(object):
  target_q_max_age = 10000 # 10000 in DeepMind paper
  replay_memory_size = 50000 # 1000000 in DeepMind paper
  replay_sample_size = 32 # 32 in DeepMind paper
  discount_factor = 0.99 # 0.99 in DeepMind paper

  def __init__(self, environment, run_name, exploration=True):
    self.actions = list(environment.actions)
    graph = ConvNetGraph(len(self.actions))
    self.current_q = ConvNetSession(graph)
    self.target_q = ConvNetSession(graph)
    self.replay_memory = deque()

    self.exploration = exploration

    self.learning_step = 1
    self.save_path_prefix = "runs/%s/checkpoints/cp" % run_name
    self.rotate_target_q()

  def rotate_target_q(self):
    directory = os.path.dirname(self.save_path_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = self.current_q.save(self.save_path_prefix, self.learning_step)
    self.target_q.load(save_path)
    return save_path

  def start_episode(self):
    pass

  def choose_action(self, state):
    action_values = self.current_q.estimate(np.array(state))
    if self.learning_step % 10 == 0:
      print "Step %d action values: %s" % (self.learning_step, repr(action_values))

    if self.exploration and random.random() > self.epsilon(self.learning_step):
      return max(self.actions, key=lambda action: action_values[self.actions.index(action)])
    else:
      return random.choice(self.actions)

  def learn(self, state, action, reward, new_state):
    self.learning_step += 1

    if self.learning_step % self.target_q_max_age == 0:
      save_path = self.rotate_target_q()
      print "Step %d new target q: %s" % (self.learning_step, save_path)


    if reward != 0:
      print "Step %d reward: %d" % (self.learning_step, reward)

    preprocessed = self.current_q.preprocess(np.array([state, new_state]))
    self.replay_memory.append((preprocessed[0], action, reward, preprocessed[1]))

    if len(self.replay_memory) >= self.replay_sample_size:
      replay_sample = random.sample(self.replay_memory, self.replay_sample_size)

      preprocessed_states = np.array([state for (state, _, _, _) in replay_sample])
      rewards = np.array([reward for (_, _, reward, _) in replay_sample])
      new_preprocessed_states = np.array([state for (_, _, _, new_state) in replay_sample])
      new_values = np.array([max(estimate) for estimate in self.target_q.estimates(new_preprocessed_states)])
      action_idxs = np.array([self.actions.index(action) for (_, action, _, _) in replay_sample])
      targets = np.array([reward + self.discount_factor * new_value
          for reward, new_value in zip(rewards, new_values)])

      # TODO: Clip change to [-1, +1]
      self.current_q.update(preprocessed_states, action_idxs, targets)

    if len(self.replay_memory) > self.replay_memory_size:
      self.replay_memory.popleft()

  def epsilon(self, k):
    return 0.1


class ConvNetGraph(object):
  screenshot_dimensions = (210, 160, 3)
  consecutive_screenshots = 2

  def __init__(self, n_actions):
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

    self.initialize = tf.initialize_all_variables()
    self.saver = tf.train.Saver()

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


class ConvNetSession(object):
  def __init__(self, graph):
    self.graph = graph
    self.session = tf.Session()
    self.session.run(self.graph.initialize)

  def estimate(self, screenshots):
    assert screenshots.shape == (2, 210, 160, 3), screenshots.shape
    feed_dict = {self.graph.placeholders.screenshots: [screenshots]}
    estimates = self.session.run(self.graph.estimators, feed_dict)
    assert estimates.shape == (1, 4), estimates
    return estimates[0]

  def estimates(self, preprocessed):
    assert preprocessed.shape == (32, 32, 32, 2), preprocessed.shape
    feed_dict = {self.graph.preprocessed: preprocessed}
    return self.session.run(self.graph.estimators, feed_dict)
    assert estimates.shape == (32, 4), estimates
    return estimates

  def update(self, preprocessed, action_idxs, targets):
    assert preprocessed.shape == (32, 32, 32, 2)
    assert action_idxs.shape == (32,)
    assert targets.shape == (32,)

    feed_dict = {
      self.graph.preprocessed: preprocessed,
      self.graph.placeholders.action_idxs: action_idxs,
      self.graph.placeholders.targets: targets
    }

    self.session.run(self.graph.optimizer, feed_dict)

  def preprocess(self, screenshots):
    assert screenshots.shape == (2, 2, 210, 160, 3)

    feed_dict = {
      self.graph.placeholders.screenshots: screenshots
    }

    result = self.session.run(self.graph.preprocessed, feed_dict)
    assert result.shape == (2, 32, 32, 2)
    return result

  def save(self, path_prefix, step):
    return self.graph.saver.save(self.session, path_prefix, global_step=step)

  def load(self, path):
    self.graph.saver.restore(self.session, path)


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


