import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 800
screen_height = 600

# Create the screen
screen = pygame.display.set_mode((screen_width, screen_height))

# Title and Icon
pygame.display.set_caption("Sprocket Arm Simulation")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

# Sprocket settings (initial values)
sprocket1_teeth = 5
sprocket2_teeth = 52
sprocket_diameter = 20  # Base diameter, will scale with the number of teeth
arm_length = 200  # Length of the arm connected to the second sprocket

# Motor settings
motor_speed = 1  # Revolutions per second
motor_direction = 1  # 1 for clockwise, -1 for anti-clockwise

def draw_sprocket(center, teeth, diameter):
    for i in range(teeth):
        angle = i * (360 / teeth)
        radian = math.radians(angle)
        x = center[0] + math.cos(radian) * diameter
        y = center[1] + math.sin(radian) * diameter
        pygame.draw.circle(screen, black, (int(x), int(y)), 3)

def draw_arm(sprocket_center, length, angle):
    end_x = sprocket_center[0] + math.cos(math.radians(angle)) * length
    end_y = sprocket_center[1] + math.sin(math.radians(angle)) * length
    pygame.draw.line(screen, red, sprocket_center, (end_x, end_y), 5)

# Main loop
running = True
clock = pygame.time.Clock()
angle = 0  # Initial angle

while running:
    screen.fill(white)
    dt = clock.tick(60) / 1000  # Delta time in seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                motor_speed += 0.1  # Increase speed
            elif event.key == pygame.K_DOWN and motor_speed > 0.1:
                motor_speed -= 0.1  # Decrease speed
            elif event.key == pygame.K_r:
                motor_direction *= -1  # Reverse direction
            elif event.key == pygame.K_q:
                sprocket1_teeth += 1  # Increase sprocket1 teeth
            elif event.key == pygame.K_a and sprocket1_teeth > 1:
                sprocket1_teeth -= 1  # Decrease sprocket1 teeth
            elif event.key == pygame.K_w:
                sprocket2_teeth += 1  # Increase sprocket2 teeth
            elif event.key == pygame.K_s and sprocket2_teeth > 1:
                sprocket2_teeth -= 1  # Decrease sprocket2 teeth

    # Update angle based on motor speed and direction
    angle += (motor_speed * motor_direction * 360 * dt) * (sprocket2_teeth / sprocket1_teeth) % 360

    # Draw sprockets and arm
    sprocket1_center = (200, 300)
    sprocket2_center = (600, 300)
    draw_sprocket(sprocket1_center, sprocket1_teeth, sprocket1_teeth * sprocket_diameter / 20)
    draw_sprocket(sprocket2_center, sprocket2_teeth, sprocket2_teeth * sprocket_diameter / 20)
    draw_arm(sprocket2_center, arm_length, math.radians(angle))

    pygame.display.update()

pygame.quit()
