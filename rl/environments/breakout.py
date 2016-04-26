from ale_python_interface import ALEInterface
import os

rom_directory = os.path.dirname(os.path.realpath(__file__)) + "/roms"

class Breakout(object):
  def __init__(self):
    self.ale = ALEInterface()
    self.ale.setInt('random_seed', 123)
    self.ale.setBool("display_screen", False)
    self.ale.setBool("sound", False)
    self.ale.loadROM("%s/breakout.bin" % rom_directory)
    self.current_state = [
      self.ale.getScreenRGB(), self.ale.getScreenRGB()
    ]

  def start_episode(self):
    self.ale.reset_game()

  def take_action(self, action):
    assert not self.terminated
    reward = self.ale.act(action)
    self.roll_state()
    return (reward, self.current_state)

  def roll_state(self):
    assert len(self.current_state) == 2
    self.current_state = [self.current_state[1], self.ale.getScreenRGB()]
    assert len(self.current_state) == 2

  @property
  def actions(self):
    return self.ale.getMinimalActionSet()

  @property
  def terminated(self):
    return self.ale.game_over()
