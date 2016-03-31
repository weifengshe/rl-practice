import unittest
from ..sarsa import Sarsa

class TestSarsa(unittest.TestCase):
  def setUp(self):
    self.sarsa = Sarsa(
        state_actions={
          'a': ['to_a', 'to_b'],
          'b': ['to_a'],
        },
        td_lambda=0.5,
        learning_rate=1.0/3)

    self.sarsa.policy = PolicyStub({
        'a': 'to_b', 'b': 'to_a'})

  def test_initial_values_are_zero(self):
    self.assertEqual(self.sarsa.value('a', 'to_a'), 0)
    self.assertEqual(self.sarsa.value('a', 'to_b'), 0)
    self.assertEqual(self.sarsa.value('b', 'to_a'), 0)

  def test_list_action_values_of_one_state(self):
    self.assertEqual(self.sarsa.action_values('a'), {'to_a': 0, 'to_b': 0})
    self.assertEqual(self.sarsa.action_values('b'), {'to_a': 0})

  def test_broadcast_td_error_to_to_previous_state_actions(self):
    self.sarsa.learn(
      state='a',
      action='to_b',
      reward=3,
      new_state='b')
    # TD error = 3 + 0 - 0 = 3

    # eligibility = 0.5**0 = 1
    # change = 1/3 * 1 * 3 = 1
    # new value = 0 + 1 = 1
    self.assertAlmostEqual(self.sarsa.value('a', 'to_b'), 1)

    self.sarsa.learn(
      state='b',
      action='to_a',
      reward=11,
      new_state='a')
    # TD error = 11 + 1 - 0 = 12

    # eligibility = 0.5**1 = 0.5
    # change = 1/3 * 0.5 * 12 = 2
    # new value = 1 + 2 = 3
    self.assertAlmostEqual(self.sarsa.value('a', 'to_b'), 3)

    # eligibility = 0.5**0 = 1
    # change = 1/3 * 1 * 12 = 4
    # new value = 0 + 4 = 4
    self.assertAlmostEqual(self.sarsa.value('b', 'to_a'), 4)

    self.sarsa.learn(
      state='a',
      action='to_b',
      reward=23,
      new_state='b')
    # TD error = 23 + 4 - 3 = 24

    # eligibility = 0.5**2 + 0.5**0 = 1.25
    # change = 1/3 * 1.25 * 24 = 10
    # new value = 3 + 10 = 13
    self.assertAlmostEqual(self.sarsa.value('a', 'to_b'), 13)

    # eligibility = 0.5**1 = 0.5
    # change = 1/3 * 0.5 * 24 = 4
    # new value = 4 + 4 = 8
    self.assertAlmostEqual(self.sarsa.value('b', 'to_a'), 8)

  def test_knows_max_action_values_per_state(self):
    self.sarsa.learn(
      state='a',
      action='to_b',
      reward=3,
      new_state='b')
    self.assertAlmostEqual(self.sarsa.max_values, {'a': 1, 'b': 0})


class PolicyStub(object):
  def __init__(self, action_lookup):
    self.action_lookup = action_lookup

  def choose_action(self, state):
    return self.action_lookup[state]
