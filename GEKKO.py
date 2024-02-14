from gekko import GEKKO
import numpy as np
from MotorSimulation import MotorSimulation
from Motor import Motor
from PIDController import PIDController
import pygame

# Ensure Pygame doesn't block waiting for events
pygame.init()
screen = pygame.display.set_mode((800, 600))

def motor_model(Kp, Ki, Kd, sim_time=100, dt=0.1):
    m = GEKKO(remote=False)
    m.time = np.linspace(0, sim_time, int(sim_time / dt) + 1)

    # PID parameters as GEKKO parameters
    Kp, Ki, Kd = [m.Param(value=val) for val in (Kp, Ki, Kd)]

    # Process variable, setpoint, and control variable
    y, setpoint, u = m.Var(value=0), m.Param(value=0), m.Var(value=0)

    # Simulate motor dynamics as a first-order system
    tau, K = 2, 1  # Time constant and gain
    m.Equation(tau * y.dt() == -y + K * u)

    # Error and its derivative
    e = m.Var(value=0)
    m.Equation(e == setpoint - y)

    # PID formula
    m.Equation(u == Kp * e + Ki * m.integral(e) + Kd * e.dt())

    # Objective to minimize the integral of the absolute error (IAE)
    m.Obj(m.integral(m.abs3(setpoint - y)))

    # Solve
    m.options.IMODE = 6
    m.solve(disp=False)

    return y.value

def tune_pid(sim_time=1000, dt=0.1):
    m = GEKKO(remote=False)

    # Randomize the starting points for PID parameters
    initial_Kp = np.random.uniform(0, 1)
    initial_Ki = np.random.uniform(0, 0)
    initial_Kd = np.random.uniform(0, 3)

    # Tunable PID parameters with initial values and bounds
    Kp, Ki, Kd = [m.FV(value=val, lb=0, ub=ub) for val, ub in zip((initial_Kp, initial_Ki, initial_Kd), (1, 0, 3))]
    for var in (Kp, Ki, Kd): var.STATUS = 1

    y = motor_model(Kp, Ki, Kd, sim_time, dt)
    m.Obj((0 - y[-1]) ** 2)  # Minimize final error

    m.options.IMODE = 3
    m.solve(disp=False)

    return Kp.NEWVAL, Ki.NEWVAL, Kd.NEWVAL

def continuous_tuning():
    while True:
        Kp_opt, Ki_opt, Kd_opt = tune_pid()
        pid_controller = PIDController(Kp_opt, Ki_opt, Kd_opt)
        print(f"Optimized PID parameters: P={Kp_opt}, I={Ki_opt}, D={Kd_opt}")

        simulation = MotorSimulation(Motor.from_name('NEO'), pid_controller, screen=screen)

        for _ in range(100):
            simulation.update(0, 0.1)
            simulation.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

continuous_tuning()
