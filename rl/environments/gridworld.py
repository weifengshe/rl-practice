import numpy as np
import operator

class GridWorld(object):
  actions_and_updates = [
    ('up', (-1, 0)),
    ('down', (1, 0)),
    ('left', (0, -1)),
    ('right', (0, 1))
  ]

  actions = [name for (name, _) in actions_and_updates]
  coordinate_updates = [direction for (_, direction) in actions_and_updates]

  def __init__(self,
      dimensions=(3, 4),
      start_state=(0, 0),
      end_states=[(0, 3), (1, 3)],
      nonstates = [(1, 1)],
      state_rewards = {(0, 3): 1, (1, 3): -1},
      step_reward = -0.1,
      max_steps = 100):
    self.dimensions = dimensions
    self.states = set(np.ndindex(dimensions)).difference(set(nonstates))
    self.start_state = start_state
    self.end_states = end_states
    self.state_rewards = state_rewards
    self.step_reward = step_reward
    self.max_steps = max_steps
    assert self.is_state(self.start_state)
    assert all(self.is_state(state) for state in self.end_states)
    assert all(not self.is_state(state) for state in nonstates)
    self.start_episode()

  @property
  def state_actions(self):
    return {state: self.actions for state in self.states}

  def start_episode(self):
    self.current_state = self.start_state
    self.step = 0
    assert self.is_state(self.current_state)

  def take_action(self, action):
    assert not self.terminated
    (reward, new_state) = self.get_followup(self.current_state, action)
    self.current_state = new_state
    self.step += 1
    return (reward, new_state)

  @property
  def terminated(self):
    return (self.current_state in self.end_states) or (self.step >= self.max_steps)

  def get_followups(self, state):
    if state in self.end_states:
      return []
    else:
      return [(action,) + self.get_followup(state, action)
          for action in self.actions]

  def get_followup(self, state, action):
    def keep_within_states(new_state):
      return new_state if self.is_state(new_state) else state

    def reward(new_state):
      return self.step_reward + self.state_rewards.get(new_state, 0)

    update = self.coordinate_updates[self.actions.index(action)]
    updated_coords = map(operator.add, state, update)
    new_state = keep_within_states(tuple(updated_coords))
    assert self.is_state(new_state)
    return reward(new_state), new_state

  def is_state(self, maybe_state):
    return maybe_state in self.states

