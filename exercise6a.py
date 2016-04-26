from rl.environments import Breakout
from rl import Simulation
from rl import util
import random
import tensorflow as tf
from tensorflow.contrib import skflow
import numpy as np

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
  def __init__(self, environment, run_name, exploration=True):
    self.actions = environment.actions
    self.action_values = ConvNet(self.actions, run_name=run_name)
    self.discount_factor = 0.9
    self.learning_step = 1
    self.exploration = exploration

  def start_episode(self):
    pass

  def choose_action(self, state):
    epsilon = self.epsilon(self.learning_step)

    if self.learning_step % 100 == 0:
      print "Step %d action values: %s" % (self.learning_step, repr({
        action: self.action_values.estimate(state, action)
        for action in self.actions
      }))

    if self.exploration and random.random() > epsilon:
      return self.greedy_action(state)
    else:
      return random.choice(self.actions)

  def greedy_action(self, state):
    def action_value(action):
      return self.action_values.estimate(state, action)
    return max(self.actions, key=action_value)

  def learn(self, state, action, reward, new_state):
    self.learning_step += 1
    if reward != 0:
      print "Step %d reward: %d" % (self.learning_step, reward)
    new_action = self.greedy_action(new_state)
    new_value = self.action_values.estimate(new_state, new_action)
    self.action_values.update(state, action, reward + self.discount_factor * new_value)

  def epsilon(self, k):
    return 0.1


class ConvNet(object):
  screenshot_dimensions = (210, 160, 3)
  consecutive_screenshots = 2

  def __init__(self, actions, run_name):
    self.placeholders = self.build_placeholders(actions)
    assert shapes(self.placeholders.screenshots) == [(None, 210, 160, 3), (None, 210, 160, 3)]

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

    self.estimates = self.build_linear_estimators(flat, actions)
    assert shapes(self.estimates) == shapes(self.placeholders.targets) ==  { 0: (None,), 1: (None,), 3: (None,), 4: (None,) }

    losses = self.build_losses(self.estimates, self.placeholders.targets)
    assert shapes(losses) == { 0: (), 1: (), 3: (), 4: () }

    self.optimizers = self.build_optimizers(losses)
    assert len(self.optimizers) == len(actions)
    assert all(type(optimizer) is tf.Operation for optimizer in self.optimizers.values())

    initialize = tf.initialize_all_variables()

    self.session = tf.Session()
    self.session.run(initialize)

  def estimate(self, screenshots, action):
    assert len(screenshots) == len(self.placeholders.screenshots) == self.consecutive_screenshots

    feed_dict = {
      placeholder: [screenshot]
      for (placeholder, screenshot) in
      zip(self.placeholders.screenshots, screenshots)
    }
    assert len(feed_dict) == 2

    estimate = self.session.run(self.estimates[action], feed_dict)
    assert len(estimate) == 1, estimate
    return estimate[0]

  def update(self, screenshots, action, target):
    assert len(screenshots) == len(self.placeholders.screenshots) == self.consecutive_screenshots

    feed_dict = {
      placeholder: [screenshot]
      for (placeholder, screenshot)
      in zip(self.placeholders.screenshots, screenshots)
    }
    feed_dict[self.placeholders.targets[action]] = [target]

    self.session.run(self.optimizers[action], feed_dict)

  def build_placeholders(self, actions):
    input_tensor_dimensions = [None] +  list(self.screenshot_dimensions)

    screenshots = [
      tf.placeholder(tf.float32, input_tensor_dimensions, name="screenshots")
      for _ in xrange(self.consecutive_screenshots)
    ]
    assert shapes(screenshots) == [(None, 210, 160, 3), (None, 210, 160, 3)]

    targets = {
      action: tf.placeholder(tf.float32, [None], name="targets")
      for action in actions
    }
    assert shapes(targets) == { 0: (None,), 1: (None,), 3: (None,), 4: (None,) }

    class Placeholders(object):
      pass
    placeholders = Placeholders()
    placeholders.screenshots = screenshots
    placeholders.targets = targets

    return placeholders

  def build_preprocessing(self, screenshots):
    assert shapes(screenshots) == [(None, 210, 160, 3), (None, 210, 160, 3)]

    grays = [tf.image.rgb_to_grayscale(screenshot) for screenshot in screenshots]
    assert shapes(grays) == [(None, 210, 160, 1), (None, 210, 160, 1)]

    grays_combined = tf.concat(3, grays)
    assert shape(grays_combined) == (None, 210, 160, 2)

    result = tf.image.resize_bicubic(grays_combined, (32, 32))
    assert shape(result) == (None, 32, 32, 2)

    # TODO: Whiten

    result.grays = grays
    result.grays_combined = grays_combined
    result.resized = result
    return result

  def build_conv_layer(self, input, n_filters, name):
    with tf.variable_scope(name):
      h = skflow.ops.conv2d(input, n_filters=n_filters, filter_shape=[3, 3],
          bias=True, activation=tf.nn.relu)
      maxpool = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')
      maxpool.h = h
      return maxpool

  def build_linear_estimators(self, input, actions):
    estimators = {
      action: self.build_linear_estimator(input, name="estimator_%d" % action)
      for action in actions
    }

    assert len(estimators) == len(actions)
    assert all(shape(estimator) == (None,)
        for estimator in estimators.values())

    return estimators

  def build_linear_estimator(self, input, name):
    with tf.variable_scope(name):
      input_width= shape(input)[-1]
      W = tf.Variable(tf.random_normal([input_width, 1], stddev=0.05), tf.float32, name="W")
      b = tf.Variable(tf.random_normal([1], stddev=0.05), tf.float32, name="b")
      result = tf.reshape((tf.matmul(input, W) + b), [-1])
      assert shape(result) == (None,)

      result.W = W
      result.b = b
      return result

  def build_losses(self, estimators, targets):
    assert estimators.keys() == targets.keys()

    return {
      action: tf.nn.l2_loss(estimators[action] - targets[action])
      for action in estimators.keys()
    }

  def build_optimizers(self, losses):
    optimizer = tf.train.AdamOptimizer()
    return {
      action: optimizer.minimize(losses[action])
      for action in losses.keys()
    }


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


