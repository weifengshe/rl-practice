import random


class Epsilon:
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
      followups = self.inner_policy.get_followups(state)
      (action, _, _) = random.choice(followups)
      return action


class Greedy:
  def __init__(self, get_followups, state_values):
    self.get_followups = get_followups
    self.state_values = state_values

  def start_episode(self):
    pass

  def choose_action(self, state):
    def value(followup):
      (action, reward, next_state) = followup
      return reward + self.state_values[next_state]

    followups = self.get_followups(state)
    (action, _, _) = max(*followups, key=value)
    return action
