class PIDController:
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.integral = 0
        self.previous_error = 0

    def update(self, setpoint, actual_value, dt):
        error = setpoint - actual_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.P * error + self.I * self.integral + self.D * derivative

    def updateConstants(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.integral = 0
        self.previous_error = 0

    def getConstants(self):
        return (self.P, self.I, self.D)
