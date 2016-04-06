import unittest
from ..greedy_afterstate_policy import GreedyAfterstatePolicy


class EnvironmentStub(object):
  def __init__(self, followup_dictionary):
    self.followup_dictionary = followup_dictionary

  def get_followups(self, state):
    return self.followup_dictionary[state]

class StateValuesStub(object):
  def __init__(self, state_values):
    self.state_values = state_values

  def value(self, state):
    return self.state_values[state]


class TestGreedyAfterstatePolicy(unittest.TestCase):
  def setUp(self):
    environment = EnvironmentStub({
      'a': [('to_a', 0, 'a'),
            ('to_b', 0, 'b')],
      'b': [('to_a', 100, 'a'),
            ('to_b', -100, 'b')]
    })

    state_values = StateValuesStub({
      'a': 1,
      'b': 10
    })

    self.greedy = GreedyAfterstatePolicy(
        environment=environment,
        state_values=state_values)


  def test_chooses_action_with_greatest_value(self):
    action = self.greedy.choose_action('a')
    self.assertEqual(action, 'to_b')

    action = self.greedy.choose_action('b')
    self.assertEqual(action, 'to_a')

  def test_lists_all_available_actions(self):
    self.assertEqual(set(self.greedy.available_actions('a')), set(['to_a', 'to_b']))
    self.assertEqual(set(self.greedy.available_actions('b')), set(['to_a', 'to_b']))
