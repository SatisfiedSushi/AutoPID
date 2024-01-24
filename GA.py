import pygad
import pygame

from Motor import Motor
from PIDController import PIDController
from MotorSimulation import MotorSimulation

# Constants
NUM_GENERATIONS = 500
NUM_PARENTS_MATING = 5
SOL_PER_POP = 10
NUM_GENES = 3
GENE_RANGE = [{'low': 0.0, 'high': 2.0}] * 3  # Range for P, I, D
SETPOINT = 0  # Target angle
DT = 0.1  # Time step for simulation
FPS = 165  # Frames per second
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SIMULATION_TIME = 20  # Total simulation time in seconds


def fitness_function(ga_instance, solution, solution_idx):
    pid = PIDController(P=solution[0], I=solution[1], D=solution[2])
    motor_type = 'NEO'  # or 'NEO_550', depending on what motor you want to simulate
    motor = Motor.from_name(motor_type)
    motor_simulation = MotorSimulation(motor, pid)
    total_error = 0

    for _ in range(100):
        motor_simulation.update(SETPOINT, DT)
        error = abs(SETPOINT - motor_simulation.angle)
        total_error += error

    return -total_error


def tune_pid():
    ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                           num_parents_mating=NUM_PARENTS_MATING,
                           fitness_func=fitness_function,
                           sol_per_pop=SOL_PER_POP,
                           num_genes=NUM_GENES,
                           gene_type=float,
                           gene_space=GENE_RANGE,
                           mutation_type=None)  # Disable mutation

    ga_instance.run()
    best_solution, _, _ = ga_instance.best_solution()
    return PIDController(P=best_solution[0], I=best_solution[1], D=best_solution[2])


def run_simulation(pid_controller, motor=Motor.from_name('NEO')):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('PID Motor Simulation')

    motor_sim = MotorSimulation(motor, pid_controller, screen=screen)
    clock = pygame.time.Clock()
    running = True
    start_time = pygame.time.get_ticks()  # Get the start time

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()
        if (current_time - start_time) / 1000 > SIMULATION_TIME:  # Check if the simulation time has elapsed
            break

        motor_sim.update(SETPOINT, DT)
        motor_sim.render()
        clock.tick(FPS)

        # print(f"Angle: {motor_sim.angle}, Time: {(current_time - start_time) / 1000}")  # Debugging line

    pygame.quit()



def test_pid(p, i, d):
    pid_controller = PIDController(P=p, I=i, D=d)
    run_simulation(pid_controller)


def main():
    # Create motors
    neo_motor = Motor.from_name('NEO')
    neo_550_motor = Motor.from_name('NEO_550')

    # best_pid = tune_pid()
    # print(f'best_pid || P:{best_pid.P}, I:{best_pid.I}, D:{best_pid.D}')
    # run_simulation(best_pid, motor=neo_motor)

    # Test PID controllers
    test_pid(1.1257244030607048, 0.16324408761227135,1.3923534593773446)


if __name__ == "__main__":
    main()
