import unittest
from gridworld import GridWorld


class TestGridWorld(unittest.TestCase):
  def setUp(self):
    self.world = GridWorld((3, 3), (1, 1), (2, 2))

  def test_lists_all_directions(self):
    self.assertEqual(sorted(self.world.direction_names),
      sorted(['left', 'right', 'up', 'down']))
    self.assertEqual(len(self.world.directions), 4)

  def test_lists_all_positions(self):
    self.assertEqual(len(self.world.all_positions), 9)

  def test_move(self):
    self.assertEqual(self.world.current, (1, 1))
    
    self.world.move('up')
    self.assertEqual(self.world.current, (0, 1))

    self.world.move('left')
    self.assertEqual(self.world.current, (0, 0))

    self.world.move('down')
    self.assertEqual(self.world.current, (1, 0))

    self.world.move('right')
    self.assertEqual(self.world.current, (1, 1))

  def test_moves_on_boundaries(self):
    self.world = GridWorld((1, 1), (0, 0), (0, 0))
    for direction_name in self.world.direction_names:
      self.world.move(direction_name)
      self.assertEqual(self.world.current, (0, 0))
