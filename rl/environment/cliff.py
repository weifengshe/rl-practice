import numpy as np
import operator

class Cliff(object):
  actions_and_updates = [
    ('up', (-1, 0)),
    ('down', (1, 0)),
    ('left', (0, -1)),
    ('right', (0, 1))
  ]

  actions = [name for (name, _) in actions_and_updates]
  coordinate_updates = [direction for (_, direction) in actions_and_updates]

  def __init__(self, dimensions, start_state, goal_state,
      cliff_states = [],
      cliff_reward = -100, step_reward = -1, max_steps = 100):
    self.dimensions = dimensions
    self.states = set(np.ndindex(dimensions))
    self.start_state = start_state
    self.goal_state = goal_state
    self.cliff_states = cliff_states
    self.cliff_reward = cliff_reward
    self.step_reward = step_reward
    self.max_steps = max_steps
    assert self.is_state(self.start_state)
    assert self.is_state(self.goal_state)
    assert all(self.is_state(state) for state in self.cliff_states)
    assert self.start_state not in self.cliff_states
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
    return (self.current_state == self.goal_state) or (self.step >= self.max_steps)

  def get_followups(self, state):
    return [(action,) + self.get_followup(state, action)
        for action in self.actions]

  def get_followup(self, state, action):
    def jump_from_cliff_states(new_state):
      return self.start_state if new_state in self.cliff_states else new_state

    def keep_within_bounds(coordinate, dimension_size):
      return max(0, min(coordinate, dimension_size - 1))

    def reward(new_state):
      if new_state in self.cliff_states:
        return self.cliff_reward
      else:
        return self.step_reward

    update = self.coordinate_updates[self.actions.index(action)]
    updated_coords = map(operator.add, state, update)
    bounded_coords = map(keep_within_bounds, updated_coords, self.dimensions)
    new_state = tuple(bounded_coords)
    assert self.is_state(new_state)
    return reward(new_state), jump_from_cliff_states(new_state)

  def is_state(self, maybe_state):
    return maybe_state in self.states

