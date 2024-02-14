import unittest

import numpy as np
import pygame
from MotorEnv import MotorEnv

class TestMotorEnv(unittest.TestCase):
    def setUp(self):
        self.env = MotorEnv()

    def test_step(self):
        initial_state = self.env.reset()
        action = self.env.action_space.sample()
        state, reward, terminated, truncated, info = self.env.step(action)
        print(info)
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_reset(self):
        state, _ = self.env.reset()
        self.assertIsInstance(state, np.ndarray)

    def test_render(self):
        self.env.reset()
        self.env.render()

if __name__ == '__main__':
    unittest.main()
