import torch
import torch.nn as nn
import random
import pygame
from copy import deepcopy

from Motor import Motor
from PIDController import PIDController
from MotorSimulation import MotorSimulation

class PIDNet(nn.Module):
    def __init__(self):
        super(PIDNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # Input layer: 3 inputs for the state
        self.fc2 = nn.Linear(10, 3)  # Output layer: 3 outputs for P, I, D
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(x)  # Apply ReLU to ensure non-negative output
        return x

class Individual:
    def __init__(self, net):
        self.net = net
        self.fitness = None

    def evaluate_fitness(self):
        pid_values = self.net(torch.FloatTensor([0, 0, 0]))  # Example state for fitness evaluation, TODO: change
        # this for real world application using vision
        self.fitness = evaluate_pid_performance(pid_values[0].item(), pid_values[1].item(), pid_values[2].item())

def crossover(ind1, ind2):
    child1 = Individual(deepcopy(ind1.net))
    child2 = Individual(deepcopy(ind2.net))

    # Cross over the weights
    for ((name1, param1), (name2, param2)) in zip(child1.net.named_parameters(), child2.net.named_parameters()):
        if 'weight' in name1:
            temp = param1.data.clone()
            param1.data = param2.data
            param2.data = temp

    return child1, child2

def mutate(individual, mutation_rate):
    for param in individual.net.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param) * 0.1

def create_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        net = PIDNet()
        individual = Individual(net)
        population.append(individual)
    return population

def evaluate_pid_performance(P, I, D):
    pid_controller = PIDController(P, I, D)
    simulation = MotorSimulation(Motor.from_name("NEO"), pid_controller)

    # Fitness evaluation parameters
    setpoint = 0  # Example setpoint for fitness evaluation
    max_time = 10
    dt = 0.1
    total_error = 0
    time_at_setpoint = 0

    for _ in range(int(max_time / dt)):
        simulation.update(setpoint, dt)
        error = abs(simulation.angle - setpoint)
        total_error += error * dt
        if error < 1:  # Within threshold of the setpoint
            time_at_setpoint += dt

    fitness = -total_error + time_at_setpoint * 5  # Fitness function
    return fitness

def run_ga(generations, pop_size, mutation_rate):
    population = create_initial_population(pop_size)

    for _ in range(generations):
        for individual in population:
            individual.evaluate_fitness()

        population.sort(key=lambda x: x.fitness, reverse=True)  # Maximize fitness

        survivors = population[:pop_size // 2]
        next_generation = survivors[:]

        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(survivors, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])

        population = next_generation

    best_individual = population[0]
    return best_individual.net

def main():
    generations = 50
    pop_size = 20
    mutation_rate = 0.2

    best_net = run_ga(generations, pop_size, mutation_rate)

    best_pid_values = best_net(torch.FloatTensor([0, 0, 0]))
    # print(f"Best PID values: P={best_pid_values[0].item()}, I={best_pid_values[1].item()}, D={best_pid_values[2].item()}")

    # Simulation with the best PID controller
    # pid_controller = PIDController(best_pid_values[0].item(), best_pid_values[1].item(), best_pid_values[2].item())
    # pygame.init()
    # screen = pygame.display.set_mode((800, 600))
    # simulation = MotorSimulation(Motor.from_name("NEO"), pid_controller, screen=screen)

    # while True:
    #     simulation.update(0, 0.1),  # Setpoint for the simulation
    #     simulation.render()
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             return

# if __name__ == "__main__":
#     main()

