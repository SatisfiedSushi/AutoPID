import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from Motor import Motor
from PIDController import PIDController
from MotorSimulation import MotorSimulation

class MotorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MotorEnv, self).__init__()
        self.action_space = spaces.Box(low=np.array([1, -0.2]), high=np.array([2, 0.2]), dtype=np.float32)  # PID values
        self.observation_space = spaces.Box(low=np.array([-180]), high=np.array([180]), dtype=np.float32)

        self.motor = Motor.from_name('NEO')
        self.pid_controller = None
        self.setpoint = 0
        self.dt = 0.1
        self.screen = None
        self.simulation = None
        self.simulation_duration = 10
        self.threshold = 0.1  # Threshold for staying at setpoint

    def step(self, action):
        if action[0] == 1:
            self.pid_controller.P += action[1]
        elif action[0] == 2:
            self.pid_controller.D += action[1]
        self.pid_controller.I = 0
        self.simulation = MotorSimulation(self.motor, self.pid_controller)

        error_scale = 0.0001

        terminated = False
        truncated = False
        reward = 0

        for _ in range(int(self.simulation_duration / self.dt)):
            self.simulation.update(self.setpoint, self.dt)

            # Calculate the shortest angular distance
            error = self.setpoint - self.simulation.angle
            error = (error + 180) % 360 - 180  # Adjust for circular nature

            if abs(error) <= self.threshold:
                reward += 1  # Positive reward for being within threshold

            # Scale the error for the reward calculation
            error_scale = 0.01  # Adjust this scale factor as needed
            reward -= (error ** 2) * error_scale  # Squared error to penalize larger errors more

            if terminated or truncated:
                break

        self.state = np.array([self.simulation.angle])
        info = {'error': error}

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.pid_controller = PIDController(0.001,0,0)
        self.simulation = MotorSimulation(self.motor, self.pid_controller)
        self.consecutive_threshold_count = 0  # Reset count on reset
        self.state = np.array([self.simulation.angle])
        return self.state, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Motor Simulation')
            self.simulation.set_screen(self.screen)

        if mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.simulation.render()
            pygame.display.flip()