import math

import pygame

from PIDController import PIDController


class MotorSimulation:
    def __init__(self, motor, pid_controller, radius=100, center=(400, 300), screen=None):
        self.motor = motor
        self.pid_controller = pid_controller
        self.angle = 120  # Initial angle of the motor
        self.angular_velocity = 0  # Initial angular velocity in degrees per second
        self.radius = radius
        self.center = center
        self.screen = screen
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)
        self.current_angular_acceleration = 0  # Current angular acceleration
        self.max_angular_acceleration = 70  # Maximum angular acceleration, adjust as needed
        self.max_angular_acceleration_change = 10  # Maximum change in angular acceleration per second
        self.angle_history = []

    def update(self, setpoint, dt):
        # Get the target angular acceleration from the PID controller
        target_angular_acceleration = self.pid_controller.update(setpoint, self.angle, dt)

        # Apply limit to the target angular acceleration
        target_angular_acceleration = max(min(target_angular_acceleration, self.max_angular_acceleration),
                                          -self.max_angular_acceleration)

        # Calculate the change in acceleration
        acceleration_change = target_angular_acceleration - self.current_angular_acceleration

        # Limit the change in acceleration
        limited_acceleration_change = max(min(acceleration_change, self.max_angular_acceleration_change * dt),
                                          -self.max_angular_acceleration_change * dt)

        # Update current angular acceleration
        self.current_angular_acceleration += limited_acceleration_change

        # Update angular velocity based on current angular acceleration
        self.angular_velocity += self.current_angular_acceleration * dt

        # Limit angular velocity to the motor's maximum
        self.angular_velocity = max(min(self.angular_velocity, self.motor.max_angular_velocity),
                                    -self.motor.max_angular_velocity)

        # Update angle based on angular velocity
        self.angle += self.angular_velocity * dt

        # Normalize the angle to be within -180 to 180 degrees
        self.angle = (self.angle + 180) % 360 - 180

        # Add the angle to the angle history
        self.angle_history.append(self.angle)

    def set_screen(self, screen):
        self.screen = screen

    def get_acceleration(self):
        return self.current_angular_acceleration

    def get_position(self):
        return self.angle

    def get_velocity(self):
        return self.angular_velocity

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Motor Simulation')
            self.set_screen(self.screen)

        self.screen.fill((0, 0, 0))  # Fill the screen with black

        # Draw a circle representing the motor's position
        x = int(self.center[0] + self.radius * math.cos(math.radians(self.angle)))
        y = int(self.center[1] + self.radius * math.sin(math.radians(self.angle)))
        pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 10)

        # Render the angle as text
        angle_text = self.font.render(f"Angle: {self.angle:.2f}", True, (255, 255, 255))
        self.screen.blit(angle_text, (10, 10))  # Position the text at the top-left corner

        # Update the display
        pygame.display.flip()

    def reset(self):
        self.angle = 120
        self.angular_velocity = 0
        self.current_angular_acceleration = 0
        self.pid_controller = PIDController(0.001,0,0)
        self.angle_history = []

    def get_angle_history(self):
        return self.angle_history
