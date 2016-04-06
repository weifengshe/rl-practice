import unittest
from ..td import TD

class TestTD(unittest.TestCase):
  def setUp(self):
    self.td = TD(
        states=set(['a', 'b', 'c']),
        td_lambda=0.5,
        learning_rate=1.0/3)

  def test_initial_values_are_zero(self):
    self.assertEqual(self.td.value('a'), 0)
    self.assertEqual(self.td.value('b'), 0)
    self.assertEqual(self.td.value('c'), 0)

  def test_broadcast_td_error_to_to_previous_states(self):
    self.td.learn(
      state='a',
      action='foo',
      reward=3,
      new_state='b')
    # TD error = 3 + 0 - 0 = 3

    # eligibility = 0.5**0 = 1
    # change = 1/3 * 1 * 3 = 1
    # new value = 0 + 1 = 1
    self.assertAlmostEqual(self.td.value('a'), 1)

    self.td.learn(
      state='b',
      action='foo',
      reward=11,
      new_state='a')
    # TD error = 11 + 1 - 0 = 12

    # eligibility = 0.5**1 = 0.5
    # change = 1/3 * 0.5 * 12 = 2
    # new value = 1 + 2 = 3
    self.assertAlmostEqual(self.td.value('a'), 3)

    # eligibility = 0.5**0 = 1
    # change = 1/3 * 1 * 12 = 4
    # new value = 0 + 4 = 4
    self.assertAlmostEqual(self.td.value('b'), 4)

    self.td.learn(
      state='a',
      action='foo',
      reward=27,
      new_state='c')
    # TD error = 27 + 0 - 3 = 24

    # eligibility = 0.5**2 + 0.5**0 = 1.25
    # change = 1/3 * 1.25 * 24 = 10
    # new value = 3 + 10 = 13
    self.assertAlmostEqual(self.td.value('a'), 13)

    # eligibility = 0.5**1 = 0.5
    # change = 1/3 * 0.5 * 24 = 4
    # new value = 4 + 4 = 8
    self.assertAlmostEqual(self.td.value('b'), 8)
