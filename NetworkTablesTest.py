import json

import torch
from networktables import NetworkTables
from GAWithMutationAndNN import run_ga

roborio_ip = "ni-roborio2-0-03249427"  # Replace with your roborio's IP address or hostname

# Initialize NTCore client mode
NetworkTables.initialize(server=roborio_ip)

# generations = 50
# pop_size = 20
# mutation_rate = 0.2
#
# best_net = run_ga(generations, pop_size, mutation_rate)
#
# best_pid_values = best_net(torch.FloatTensor([0, 0, 0]))
# pid_values = {
#     "P": best_pid_values[0].item(),
#     "I": best_pid_values[1].item(),
#     "D": best_pid_values[2].item()
# }

pid_values = {
    "P": 0.001,
    "I": 0,
    "D": 0.5
}

# Serialize the dictionary to a JSON string
pid_values_json = json.dumps(pid_values)

# Store the JSON string in NetworkTables
pid_table = NetworkTables.getTable("AutoPIDValues")
# Directly set individual PID values in NetworkTables
# Python side remains the same as in your original code
# Directly set individual PID values in NetworkTables
pid_table.putNumber("P", pid_values["P"])
pid_table.putNumber("I", pid_values["I"])
pid_table.putNumber("D", pid_values["D"])


# To retrieve and deserialize on the receiving end
received_num = pid_table.getNumber("P", "0")
print(received_num)
