import numpy as np
import random
import pygame
import multiprocessing
from functools import partial

# Assuming Motor, PIDController, MotorSimulation, and Individual classes are defined as before
from Motor import Motor
from PIDController import PIDController
from MultiMotorSimulation import MotorSimulation


class Individual:
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.fitness = None

    def evaluate_fitness(self, index, generation):
        print(f"Evaluating Individual {index + 1} in Generation {generation + 1}: P={self.P}, I={self.I}, D={self.D}")
        self.fitness = evaluate_pid_performance(self.P, self.I, self.D)
        print(f"Fitness: {self.fitness}\n")



def evaluate_pid_performance(P, I, D):
    # Create a PID controller with the given parameters
    pid_controller = PIDController(P, I, D)

    # Assuming a motor is defined elsewhere in your code
    motor = Motor.from_name("NEO")  # Replace with actual motor initialization if needed

    # Set up the simulation
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    simulation = MotorSimulation(motor, pid_controller, screen=screen)

    # Run the simulation for a certain period
    total_time = 100  # total simulation time in seconds
    dt = 0.1  # simulation time step
    total_error = 0

    for _ in range(int(total_time / dt)):
        simulation.update(0, dt)  # Assuming the setpoint is 0
        error = abs(simulation.angle)  # Calculate error
        total_error += error * dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return float('inf')  # Return a high error if simulation is closed prematurely

        # simulation.render()

    pygame.quit()

    # Fitness could be total error, lower is better
    return total_error

def create_initial_population(pop_size, P_range, I_range, D_range):
    population = []
    for _ in range(pop_size):
        individual = Individual(
            P=random.uniform(*P_range),
            I=random.uniform(*I_range),
            D=random.uniform(*D_range)
        )
        population.append(individual)
    return population

def crossover(parent1, parent2):
    child1 = Individual(
        P=(parent1.P + parent2.P) / 2,
        I=(parent1.I + parent2.I) / 2,
        D=(parent1.D + parent2.D) / 2
    )
    child2 = Individual(
        P=(parent1.P + parent2.P) / 2,
        I=(parent1.I + parent2.I) / 2,
        D=(parent1.D + parent2.D) / 2
    )
    return child1, child2

def mutate(individual, P_range, I_range, D_range, mutation_rate):
    if random.random() < mutation_rate:
        individual.P = random.uniform(*P_range)
    if random.random() < mutation_rate:
        individual.I = random.uniform(*I_range)
    if random.random() < mutation_rate:
        individual.D = random.uniform(*D_range)

def run_ga(generations, pop_size, P_range, I_range, D_range, mutation_rate):
    population = create_initial_population(pop_size, P_range, I_range, D_range)

    # Create a pool of processes
    pool = multiprocessing.Pool()

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1}/{generations} ---")

        # Evaluate fitness in parallel
        evaluate_fitness_partial = partial(evaluate_individual_fitness, generation=gen)
        fitness_results = pool.map(evaluate_fitness_partial, enumerate(population))
        for i, fitness in enumerate(fitness_results):
            population[i].fitness = fitness

        # Sort by fitness (assuming lower fitness is better)
        population.sort(key=lambda x: x.fitness)

        # Display best fitness in current generation
        print(f"Best Fitness in Generation {gen + 1}: {population[0].fitness}")

        # Select the top half of individuals
        survivors = population[:pop_size // 2]

        # Create next generation
        next_generation = []
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(survivors, 2)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, P_range, I_range, D_range, mutation_rate)
            mutate(child2, P_range, I_range, D_range, mutation_rate)
            next_generation.extend([child1, child2])

        population = next_generation

    pool.close()
    pool.join()

    # Return the best individual
    best_individual = population[0]
    return best_individual.P, best_individual.I, best_individual.D

def evaluate_individual_fitness(individual_tuple, generation):
    index, individual = individual_tuple
    print(f"Evaluating Individual {index + 1} in Generation {generation + 1}: P={individual.P}, I={individual.I}, D={individual.D}")
    fitness = evaluate_pid_performance(individual.P, individual.I, individual.D)
    print(f"Fitness: {fitness}\n")
    return fitness

def main():
    # Example GA parameters
    generations = 50
    pop_size = 20
    P_range = (0, 2)
    I_range = (0, 2)
    D_range = (0, 2)
    mutation_rate = 0.3

    # Run GA with multiprocessing
    best_P, best_I, best_D = run_ga(generations, pop_size, P_range, I_range, D_range, mutation_rate)
    print(f"\nBest PID Found: P={best_P}, I={best_I}, D={best_D}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


