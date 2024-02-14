import math

import pygame


class MotorSimulation:
    def __init__(self, motor, pid_controller, x=0, y=0, width=600, height=600, radius=100, center=(0, 0), screen=None):
        self.motor = motor
        self.pid_controller = pid_controller
        self.angle = 290  # Initial angle of the motor
        self.angular_velocity = 0  # Initial angular velocity in degrees per second
        self.radius = radius
        self.center = center
        self.screen = screen
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)
        self.current_angular_acceleration = 0  # Current angular acceleration
        self.max_angular_acceleration = 20  # Maximum angular acceleration, adjust as needed
        self.max_angular_acceleration_change = 10  # Maximum change in angular acceleration per second

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

    def render(self):
        # Calculate the relative position of the motor within the simulation box
        radius = min(self.width, self.height) // 4
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2

        # Draw a circle representing the motor's position
        x = int(center_x + radius * math.cos(math.radians(self.angle)))
        y = int(center_y + radius * math.sin(math.radians(self.angle)))
        pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 10)

        # Render the angle as text within the simulation box
        angle_text = self.font.render(f"Angle: {self.angle:.2f}", True, (255, 255, 255))
        self.screen.blit(angle_text, (self.x + 10, self.y + 10))
