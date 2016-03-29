import unittest
from td0 import TD0

class TestTD0(unittest.TestCase):
  def setUp(self):
    self.td = TD0(
        states=set(['a', 'b']),
        learning_rate=0.1)

  def test_initial_values_are_zero(self):
    self.assertEqual(self.td['a'], 0)
    self.assertEqual(self.td['b'], 0)

  def test_move_closer_to_given_reward(self):
    self.assertEqual(self.td['a'], 0)

    self.td.learn(
      state='a',
      action='foo',
      reward=3,
      new_state='b')

    self.assertAlmostEqual(self.td['a'], 0.3)

  def test_move_closer_to_follower_state_value(self):
    # given
    self.td.learn('b', 'foo', 100, 'b')
    self.assertAlmostEqual(self.td['b'], 10.0)

    # when
    self.td.learn(
      state='a',
      action='foo',
      reward=0,
      new_state='b')

    # then
    self.assertAlmostEqual(self.td['a'], 1.0)
