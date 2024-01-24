import json

import torch
from networktables import NetworkTables
from GAWithMutationAndNN import run_ga

roborio_ip = "roborio-4787-frc.local"  # Replace with your roborio's IP address or hostname

# Initialize NTCore client mode
NetworkTables.initialize(server=roborio_ip)

# Create a table specifically for PID values
pid_table = NetworkTables.getTable("AutoPIDValues")

generations = 50
pop_size = 20
mutation_rate = 0.2

best_net = run_ga(generations, pop_size, mutation_rate)

best_pid_values = best_net(torch.FloatTensor([0, 0, 0]))
pid_values = {
    "P": best_pid_values[0].item(),
    "I": best_pid_values[1].item(),
    "D": best_pid_values[2].item()
}

# Serialize the dictionary to a JSON string
pid_values_json = json.dumps(pid_values)

# Store the JSON string in NetworkTables
pid_table = NetworkTables.getTable("PIDValues")
pid_table.putString("PIDSettings", pid_values_json)

# To retrieve and deserialize on the receiving end
received_json = pid_table.getString("PIDSettings", "{}")
received_dict = json.loads(received_json)
print(received_dict)
