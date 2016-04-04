import random
import math

class EpsilonPolicy:
  """
  Choose an action at random with probability epsilon, otherwise
  choose with inner_policy. The epsilon is one of

  - A float between 0 and 1
  - "inverse decay" (default) for the inverse of current episode number
  - "inverse sqrt decay" for the inverse of square root of current episode number
  - A function taking the current episode number as parameter
  """
  def __init__(self, inner_policy, epsilon = "inverse_decay"):
    self.inner_policy = inner_policy
    self.episode_count = 0

    if callable(epsilon):
      self.epsilon = epsilon
    elif epsilon == "inverse_decay":
      self.epsilon = lambda k: 1.0 / k
    elif epsilon == "inverse_sqrt_decay":
      self.epsilon = lambda k: 1.0 / math.sqrt(k)
    else:
      self.epsilon = lambda k: float(epsilon)

  def start_episode(self):
    self.episode_count += 1
    self.inner_policy.start_episode()

  def choose_action(self, state):
    if random.random() > self.epsilon(self.episode_count):
      return self.inner_policy.choose_action(state)
    else:
      return random.choice(self.inner_policy.available_actions(state))
