import unittest
from ..greedy_action_policy import GreedyActionPolicy


class ValueStub(object):
  def __init__(self, state_action_values):
    self.state_action_values = state_action_values

  def action_values(self, state):
    return self.state_action_values[state]


class TestGreedyActionPolicy(unittest.TestCase):
  def setUp(self):
    state_action_values = ValueStub({
      'a': {'foo': 1, 'bar': 5},
      'b': {'foo': 10, 'bar': -10}})

    self.greedy = GreedyActionPolicy(
        state_action_values=state_action_values)


  def test_chooses_action_with_greatest_value(self):
    action = self.greedy.choose_action('a')
    self.assertEqual(action, 'bar')

    action = self.greedy.choose_action('b')
    self.assertEqual(action, 'foo')

  def test_lists_all_available_actions(self):
    self.assertEqual(set(self.greedy.choices('a')), set(['foo', 'bar']))
    self.assertEqual(set(self.greedy.choices('b')), set(['foo', 'bar']))
