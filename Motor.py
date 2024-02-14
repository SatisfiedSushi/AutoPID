class Motor:
    predefined_specs = {
        'NEO': {'max_rpm': 5676, 'stall_torque': 2.6, 'stall_current': 105, 'free_current': 1.3},
        'NEO_550': {'max_rpm': 11000, 'stall_torque': 0.97, 'stall_current': 100, 'free_current': 0.4}
    }

    def __init__(self, max_rpm, stall_torque, stall_current, free_current):
        self.max_rpm = max_rpm
        self.stall_torque = stall_torque
        self.stall_current = stall_current
        self.free_current = free_current
        self.free_speed = max_rpm  # Assuming free speed is max RPM

        # Convert max RPM to degrees per second for simulation
        # self.max_angular_velocity = (max_rpm * 360) / 60  # 1 RPM = 360 degrees/60 seconds
        self.max_angular_velocity = 1000

    @classmethod
    def from_name(cls, motor_name):
        specs = cls.predefined_specs.get(motor_name)
        if specs is not None:
            return cls(**specs)
        else:
            raise ValueError(f"Unknown motor name: {motor_name}")

    def calculate_torque(self, current_speed):
        return self.stall_torque * (1 - current_speed / self.free_speed)
