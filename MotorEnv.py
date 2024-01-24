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
        self.action_space = spaces.Box(low=0, high=5, shape=(3,), dtype=np.float32)  # PID values
        self.observation_space = spaces.Box(low=np.array([-180, -np.inf]), high=np.array([180, np.inf]), dtype=np.float32)

        self.motor = Motor.from_name('NEO')
        self.pid_controller = None  # Initialized in reset
        self.setpoint = 0
        self.dt = 0.1
        self.screen = None
        self.clock = None
        self.simulation = None
        self.simulation_duration = 50  # Duration of each episode
        self.elapsed_time = 0

    def step(self, action):
        # No PID update during the episode, just simulate the motor
        self.simulation.update(self.setpoint, self.dt)
        angle = self.simulation.angle
        angular_velocity = self.simulation.angular_velocity

        self.elapsed_time += self.dt
        terminated = self.elapsed_time >= self.simulation_duration
        truncated = False
        reward = 0

        if terminated:
            # Compute the reward based on the performance at the end of the episode
            error = np.mean([abs(self.setpoint - self.simulation.angle) for _ in range(100)])  # Example calculation
            reward = -error  # Negative reward for higher error

        self.state = np.array([angle, angular_velocity])
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # Set PID values based on action at the beginning of the episode
        self.pid_controller = PIDController(*self.action_space.sample())
        self.simulation = MotorSimulation(self.motor, self.pid_controller)
        self.elapsed_time = 0

        initial_angle = self.simulation.angle
        initial_angular_velocity = self.simulation.angular_velocity
        self.state = np.array([initial_angle, initial_angular_velocity])
        return self.state, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Motor Simulation')
            self.clock = pygame.time.Clock()
            self.simulation.screen = self.screen  # Set the screen in MotorSimulation

        if mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            self.simulation.render()
            self.clock.tick(60)

# Example usage
# env = MotorEnv()
# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Only sampled once per episode
#     obs, reward, done, info = env.step(action)
#     env.render()
# env.close()
