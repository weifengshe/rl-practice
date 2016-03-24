import unittest
from gridworld import GridWorld


class TestGridWorld(unittest.TestCase):
  def setUp(self):
    self.world = GridWorld(
      dimensions = (3, 3),
      start_state = (1, 1),
      goal_state = (2, 2))

  def test_lists_all_actions(self):
    self.assertEqual(sorted(self.world.actions),
      sorted(['left', 'right', 'up', 'down']))

  def test_lists_all_possible_states(self):
    self.assertEqual(len(self.world.states), 9)

  def test_move(self):
    self.assertEqual(self.world.current_state, (1, 1))
    
    self.world.take_action('up')
    self.assertEqual(self.world.current_state, (0, 1))

    self.world.take_action('left')
    self.assertEqual(self.world.current_state, (0, 0))

    self.world.take_action('down')
    self.assertEqual(self.world.current_state, (1, 0))

    self.world.take_action('right')
    self.assertEqual(self.world.current_state, (1, 1))

  def test_moves_on_boundaries(self):
    self.world = GridWorld((1, 1), (0, 0), (0, 0))
    for action in self.world.actions:
      self.world.take_action(action)
      self.assertEqual(self.world.current_state, (0, 0))
