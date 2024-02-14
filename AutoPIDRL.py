import os.path
import numpy as np
import torch
from gym.spaces import Box
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
from stable_baselines3 import DDPG, SAC
from MotorEnv import MotorEnv
import pygame
from MotorSimulation import MotorSimulation
from Motor import Motor
from PIDController import PIDController

# Configuration parameters
USE_SAC = True  # Set to False to use DDPG
LEARNING_RATE = 1e-2
BATCH_SIZE = 256
ACTION_NOISE_STD_DEV = 0.2
LAYER_SIZES = [400, 300]  # Customize your MLP layers here

# Check if CUDA is available
print(torch.cuda.is_available())

if USE_SAC:
    models_dir = "models/SAC"
else:
    models_dir = "models/DDPG"
logdir = "logs"

# Create directories if they don't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize environment
env = MotorEnv()
n_actions = env.action_space.shape[-1]

# Define action noise
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=ACTION_NOISE_STD_DEV * np.ones(n_actions))

# Select and configure model
if USE_SAC:
    model = SAC("MlpPolicy", env, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                device="cuda", tensorboard_log=logdir, verbose=1)
    # model = SAC.load(f'{models_dir}/Season 0', env=env, device='cuda', batch_size=BATCH_SIZE,
    #                  learning_rate=LEARNING_RATE, tensorboard_log=logdir, verbose=1)
else:
    model = DDPG("MlpPolicy", env, action_noise=action_noise, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, device="cuda", tensorboard_log=logdir, verbose=1)

seasons = 50
time_steps = 10000

# Training
for i in range(seasons):
    model.learn(total_timesteps=time_steps, progress_bar=True)
    model.save(os.path.join(models_dir, f"Season {i}"))

# Testing the model
# def test():
#     while True:
#         obs, info = env.reset()
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncation, info = env.step(action)
#         pid_controller = PIDController(action[0], 0, action[1])
#
#         print(f'P: {action[0]}, I: 0, D: {action[1]}')
#
#         pygame.init()
#         screen = pygame.display.set_mode((800, 600))
#         pygame.display.set_caption('Motor Simulation')
#
#         simulation = MotorSimulation(Motor.from_name('NEO'), pid_controller, screen=screen)
#
#         for _ in range(100):
#             simulation.update(0, 0.1)
#             simulation.render()
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     return
#
# test()

