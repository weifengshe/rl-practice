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
  agent = DQN(environment)
  simulation = Simulation(environment, agent)

  for step in xrange(11):
    print "Starting episode %d" % step

    if step % 1 == 0:
      util.save_animation(simulation.episode_steps(), "videos/%s/%07d.gif" % (run_name, step))
    else:
      simulation.run_episode()

  util.open_videos_in_web_browser("videos/%s" % run_name)


class DQN(object):
  def __init__(self, environment, exploration=True):
    self.actions = environment.actions
    self.action_values = ConvNet(self.actions)
    self.discount_factor = 0.9
    self.learning_step = 1
    self.exploration = exploration

  def start_episode(self):
    pass

  def choose_action(self, state):
    epsilon = self.epsilon(self.learning_step)

    if self.learning_step % 10 == 0:
      print "Step:", self.learning_step
      print "Action values:", repr({
        action: self.action_values.estimate(state, action)
        for action in self.actions
      })

    if random.random() > epsilon:
      return self.greedy_action(state)
    else:
      print "r",
      return random.choice(self.actions)

  def greedy_action(self, state):
    def action_value(action):
      return self.action_values.estimate(state, action)
    return max(self.actions, key=action_value)

  def learn(self, state, action, reward, new_state):
    self.learning_step += 1
    if reward != 0:
      print reward,
    new_action = self.greedy_action(new_state)
    new_value = self.action_values.estimate(new_state, new_action)
    self.action_values.update(state, action, reward + self.discount_factor * new_value)

  def epsilon(self, k):
    if self.exploration:
      # return 1.0 / k
      ### Alternative schedules to try:
      return 0.1
      # return (k/10.0)**(-0.8)
    else:
      return 0


class ConvNet(object):
  def __init__(self, actions):
    self.action_estimator = {
      action: self.estimator('action-' + str(action))
      for action in actions
    }

  def estimator(self, name):
    with tf.variable_scope(name):
      estimator = skflow.TensorFlowEstimator(
          model_fn=self.conv_model,
          n_classes=1, batch_size=1, steps=1,
          continue_training=True, learning_rate=0.0001)

    estimator.fit(np.zeros((1, 210, 160, 3)), [0])
    return estimator

  def conv_model(self, X, y):
    rescaled = tf.image.resize_bicubic(tf.image.rgb_to_grayscale(X), (32, 32))
    conv1 = self.conv_layer(rescaled, 8, 'conv1') # (16, 16)
    conv2 = self.conv_layer(conv1, 8, 'conv2') # (8, 8)
    flat = tf.reshape(conv2, [-1, 8 * 8 * 8])
    return skflow.models.linear_regression(flat, y)

  def conv_layer(self, input, n_filters, name):
    with tf.variable_scope(name):
      h = skflow.ops.conv2d(input, n_filters=n_filters, filter_shape=[3, 3],
          bias=True, activation=tf.nn.relu)
      return self.max_pool_2x2(h)

  def max_pool_2x2(self, tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

  def update(self, image, action, target):
    self.action_estimator[action].fit(
        np.array([image]),
        np.array([target]),
        logdir='tensorflow_logs/action' + action)

  def estimate(self, image, action):
    [[result]] = self.action_estimator[action].predict(np.array([image]))
    return result

run_exercise()
