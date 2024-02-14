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
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.fitness = None

    def evaluate_fitness(self):
        self.fitness = evaluate_pid_performance(self.P, self.I, self.D)


def crossover(ind1, ind2):
    # Simple one-point crossover
    child1 = Individual((ind1.P + ind2.P) / 2, (ind1.I + ind2.I) / 2, (ind1.D + ind2.D) / 2)
    child2 = Individual((ind1.P + ind2.P) / 2, (ind1.I + ind2.I) / 2, (ind1.D + ind2.D) / 2)
    return child1, child2


def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        individual.P = min(max(individual.P + random.uniform(-0.1, 0.1), 0), 1)
    if random.random() < mutation_rate:
        individual.D = min(max(individual.D + random.uniform(-0.1, 0.1), 0), 2)

def create_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        P = random.uniform(0, 1)  # Adjust range as needed
        I = 0
        D = random.uniform(0, 2)  # Adjust range as needed
        individual = Individual(P, I, D)
        population.append(individual)
    return population


def evaluate_pid_performance(P, I, D):
    pid_controller = PIDController(P, I, D)
    simulation = MotorSimulation(Motor.from_name("NEO"), pid_controller)

    setpoint = 0
    max_time = 3
    dt = 0.1
    total_error = 0

    for t in range(int(max_time / dt)):
        simulation.update(setpoint, dt)
        error = abs(simulation.angle - setpoint)
        total_error += error

    # The fitness is inversely proportional to the total error
    fitness = -total_error

    return fitness


def run_ga(generations, pop_size, mutation_rate):
    population = create_initial_population(pop_size)

    for _ in range(generations):
        print(f'Generation {_}')
        # Evaluate fitness
        for individual in population:
            individual.evaluate_fitness()

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Selection and breeding
        survivors = population[:pop_size // 2]
        next_generation = survivors[:]
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(survivors, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_generation.extend([child1, child2])

        population = next_generation
        print(f'Best fitness: {population[0].fitness} || {population[0].P}, {population[0].I}, {population[0].D}')

    best_individual = population[0]
    return best_individual



def main():
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))

    # Parameters for simulation
    generations = 10000
    pop_size = 1000
    mutation_rate = 0.1
    setpoint = 0  # Setpoint for the simulation
    sim_duration = 3 # Duration of the simulation in seconds

    # Run GA
    best_individual = run_ga(generations, pop_size, mutation_rate)
    print(f'best_pid || P:{best_individual.P}, I:{best_individual.I}, D:{best_individual.D}')

    # Rest of your simulation code using best_individual's PID values

    while True:  # Main loop to restart the simulation
        pid_controller = PIDController(best_individual.P, best_individual.I, best_individual.D)
        simulation = MotorSimulation(Motor.from_name("NEO"), pid_controller, screen=screen)

        clock = pygame.time.Clock()
        DT = 0.1
        FPS = 165
        start_time = pygame.time.get_ticks()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Exit the main function and end the program

            current_time = pygame.time.get_ticks()
            if (current_time - start_time) / 1000 > sim_duration:
                break  # Break out of the simulation loop to restart it

            simulation.update(setpoint, DT)
            simulation.render()

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()


