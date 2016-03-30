import unittest
from ..greedy import Greedy


class TestGreedy(unittest.TestCase):
  def setUp(self):
    def get_followups(state):
      if state == 'a':
        return [
          ('to_a', 0, 'a'),
          ('to_b', 0, 'b')
        ]
      elif state == 'b':
        return [
          ('to_a', 100, 'a'),
          ('to_b', -100, 'b')
        ]
      else:
        assert False


    state_values = {
      'a': 1,
      'b': 10
    }

    self.greedy = Greedy(
        get_followups=get_followups,
        state_values=state_values)


  def test_chooses_action_with_greatest_value(self):
    action = self.greedy.choose_action('a')
    self.assertEqual(action, 'to_b')

    action = self.greedy.choose_action('b')
    self.assertEqual(action, 'to_a')
