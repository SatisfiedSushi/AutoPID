import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from Motor import Motor
from PIDController import PIDController
from MotorSimulation import MotorSimulation

# Constants
SETPOINT = 0  # Target angle
DT = 0.1  # Time step for simulation
FPS = 165  # Frames per second
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SIMULATION_TIME = 20  # Total simulation time in seconds
NUM_EPOCHS = 1000
NUM_ITERATIONS = 1000

# Neural Network for PID Tuning
class PIDNet(nn.Module):
    def __init__(self):
        super(PIDNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Input layer (1 input feature)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # Output layer (3 outputs for P, I, D)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(x)  # Apply ReLU to ensure non-negative output
        return x

# Function to calculate the fitness
def calculate_fitness(motor_simulation):
    total_error = 0
    last_error = 0
    oscillation_count = 0
    time_to_converge = 0
    converged = False

    for step in range(100):  # Number of steps in simulation
        motor_simulation.update(SETPOINT, DT)
        error = abs(SETPOINT - motor_simulation.angle)
        total_error += error

        # Check for oscillation
        if last_error * error < 0:
            oscillation_count += 1

        # Check for convergence
        if not converged and error < 1:  # Threshold for convergence
            time_to_converge = step * DT
            converged = True

        last_error = error

    # Calculate fitness with penalties
    fitness = total_error + oscillation_count * 50 + (100 - time_to_converge) * 10
    return fitness

# Training Function
def train_network():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = PIDNet().to(device)  # Move the model to GPU if available
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    motor = Motor.from_name('NEO')
    pid_controller = PIDController(0.0, 0.0, 0.0)

    best_pid_params = None
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        for iteration in range(NUM_ITERATIONS):
            input_feature = torch.tensor([[SETPOINT]], dtype=torch.float).to(device)  # Move input to GPU
            pid_params = net(input_feature)
            pid_controller.updateConstants(*pid_params.cpu().detach().numpy()[0])  # Move data back to CPU for other operations

            motor_simulation = MotorSimulation(motor, pid_controller)
            fitness = calculate_fitness(motor_simulation)
            loss = -fitness

            # Convert loss to PyTorch tensor and move to GPU
            loss_tensor = torch.tensor([loss], requires_grad=True).to(device)

            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()

            if loss_tensor.item() < best_loss:
                best_loss = loss_tensor.item()
                best_pid_params = pid_params.cpu().detach().numpy()[0]  # Move data back to CPU

            print(f'\rEpoch: {epoch+1}/{NUM_EPOCHS}, Iteration: {iteration+1}/{NUM_ITERATIONS}, Loss: {loss_tensor.item():.4f}', end='', flush=True)

    print()
    return best_pid_params


def run_simulation_for_error(pid_controller, motor):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('PID Motor Simulation')

    motor_sim = MotorSimulation(motor, pid_controller, screen=screen)
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()

    total_error = 0
    last_error = 0
    oscillation_count = 0
    converged = False
    time_to_converge = 0

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        if (current_time - start_time) / 1000 > SIMULATION_TIME:
            break

        motor_sim.update(SETPOINT, DT)
        error = abs(SETPOINT - motor_sim.angle)
        total_error += error

        # Check for oscillation
        if last_error * error < 0:
            oscillation_count += 1

        # Check for convergence
        if not converged and error < 1:  # Threshold for convergence
            time_to_converge = (current_time - start_time) / 1000
            converged = True

        last_error = error

        motor_sim.render()
        clock.tick(FPS)

    pygame.quit()

    # Calculate fitness with penalties (similar to training function)
    fitness = total_error + oscillation_count * 50 + (SIMULATION_TIME - time_to_converge) * 10
    return fitness


def main():
    # Train the network and get the best PID parameters
    best_pid_params = train_network()
    print(f"Trained PID Parameters: P={best_pid_params[0]}, I={best_pid_params[1]}, D={best_pid_params[2]}")

    # Create a PID controller with the trained parameters
    trained_pid_controller = PIDController(*best_pid_params)

    # Create a motor instance (assuming NEO motor)
    neo_motor = Motor.from_name('NEO')

    # Run the simulation with the trained PID controller
    run_simulation_for_error(trained_pid_controller, neo_motor)


if __name__ == "__main__":
    main()


