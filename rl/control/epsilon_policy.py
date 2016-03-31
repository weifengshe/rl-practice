import random


class EpsilonPolicy:
  def __init__(self, inner_policy):
    self.inner_policy = inner_policy
    self.episode_count = 0

  def start_episode(self):
    self.episode_count += 1
    self.inner_policy.start_episode()

  def choose_action(self, state):
    if random.random() > 1.0 / self.episode_count:
      return self.inner_policy.choose_action(state)
    else:
      return random.choice(self.inner_policy.available_actions(state))
