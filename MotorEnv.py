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
        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-360, -np.inf]), high=np.array([360, np.inf]), dtype=np.float32)

        # Motor and PID Controller Initialization
        self.motor = Motor.from_name('NEO')
        self.pid_controller = PIDController(0.0, 0.0, 0.0)

        # Simulation Parameters
        self.setpoint = 0
        self.dt = 0.1
        self.screen = None
        self.clock = None
        self.simulation = None

        # Time near setpoint tracking
        self.time_near_setpoint = 0.0
        self.goal_time = 10.0  # Time in seconds to stay near setpoint for max reward
        # Add a timer for total elapsed time
        self.total_elapsed_time = 0.0
        self.max_time_without_goal = 200.0  # Maximum time allowed without reaching the goal

    def step(self, action):
        # Update PID Controller with action
        self.pid_controller.updateConstants(*action)

        # Update Motor Simulation
        self.simulation.update(self.setpoint, self.dt)
        angle = self.simulation.angle
        angular_velocity = self.simulation.angular_velocity

        # Update total elapsed time
        self.total_elapsed_time += self.dt

        # Calculate reward and done condition
        error = abs(self.setpoint - angle)
        if error < 1:  # Threshold for being "at setpoint"
            self.time_near_setpoint += self.dt
        else:
            self.time_near_setpoint = 0.0  # Reset if not near setpoint

        reward = -error
        done = False

        # Check for goal condition or timeout
        if self.time_near_setpoint >= self.goal_time:
            reward = 100  # Assign a positive reward for achieving the goal
            done = True
        elif self.total_elapsed_time >= self.max_time_without_goal:
            reward = -100  # Assign a negative reward for taking too long
            done = True

        self.state = np.array([angle, angular_velocity])
        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment state
        self.pid_controller = PIDController(0.0, 0.0, 0.0)
        self.simulation = MotorSimulation(self.motor, self.pid_controller)
        self.time_near_setpoint = 0.0  # Reset the counter
        self.total_elapsed_time = 0.0  # Reset total elapsed time

        initial_angle = self.simulation.angle
        initial_angular_velocity = self.simulation.angular_velocity
        self.state = np.array([initial_angle, initial_angular_velocity])
        return self.state

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

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# Example usage
env = MotorEnv()
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
