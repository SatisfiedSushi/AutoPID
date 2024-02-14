import numpy as np
import pygame

from Motor import Motor
from MotorSimulation import MotorSimulation
from PIDController import PIDController


class HeuristicPIDTuner:
    def __init__(self, motor_simulation, kp_range, ki_range, kd_range, evaluation_metric):
        self.motor_simulation = motor_simulation
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        self.evaluation_metric = evaluation_metric
        self.best_params = {'kp': 0, 'ki': 0, 'kd': 0}
        self.best_performance = float('inf')

    def evaluate_performance(self, kp, ki, kd):
        # Update PID constants in the simulation's PID controller
        self.motor_simulation.pid_controller.Kp = kp
        self.motor_simulation.pid_controller.Ki = 0
        self.motor_simulation.pid_controller.Kd = kd

        # Reset the simulation state
        self.motor_simulation.reset()

        # Run the simulation
        for _ in range(100):  # Define the number of steps or a stopping condition
            self.motor_simulation.update(0, 0.1)  # Example update call, adjust as needed

        # Evaluate the performance based on the simulation results
        return self.evaluation_metric(self.motor_simulation)

    def tune_pid(self):
        for kp in self.kp_range:
            print(f"Testing kp: {kp}")
            for kd in self.kd_range:
                performance = self.evaluate_performance(kp, 0, kd)
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.best_params = {'kp': kp, 'ki': 0, 'kd': kd}
        print(f"Best PID: {self.best_params} with performance: {self.best_performance}")


# Example usage
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
motor_simulation = MotorSimulation(Motor.from_name('NEO_550'), PIDController(0, 0, 0), screen=screen)

def evaluation_metric(motor_simulation):
    # Assuming motor_simulation has a method to return the history of angles
    # and there's a desired setpoint to compare against
    setpoint = 0  # Define your setpoint here
    error_sum = sum((setpoint - angle) ** 2 for angle in motor_simulation.get_angle_history())
    return error_sum

# Define ranges for kp, ki, and kd with a step of 0.001
kp_range = np.arange(0, 1 + 0.001, 0.001)
ki_range = [0,0]  # no i term
kd_range = np.arange(0, 2 + 0.001, 0.001)

# Initialize the tuner with the new ranges
tuner = HeuristicPIDTuner(
    motor_simulation,
    kp_range=kp_range,
    ki_range=ki_range,
    kd_range=kd_range,
    evaluation_metric=evaluation_metric
)

tuner.tune_pid()
