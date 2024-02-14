#!/usr/bin/env python3

import json
import torch
import ntcore
from ntcore import NetworkTableInstance, DoubleTopic, PubSubOptions, _now, Topic
from GAWithMutationAndNN import run_ga

TEAM = 4787  # Your team number


class PIDPublisher:
    def __init__(self, inst: NetworkTableInstance):
        self.pidTable = inst.getTable("PIDValues")
        topic = Topic("PIDSettings"), self.pidTable, PubSubOptions(0, 0, 0, 0)
        self.pidTopic = DoubleTopic(topic)
        self.pidPub = self.pidTopic.publish()

    def publishPID(self, pid_values):
        # Serialize the dictionary to a JSON string
        pid_values_json = json.dumps(pid_values)
        self.pidPub.set(pid_values_json)

    def close(self):
        self.pidPub.close()


def main():
    inst = NetworkTableInstance.getDefault()

    # Set up client mode with team number
    inst.startClientTeam(TEAM)
    inst.startDSClient()  # Recommended if running on DS computer

    pidPublisher = PIDPublisher(inst)

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

    pidPublisher.publishPID(pid_values)
    pidPublisher.close()


if __name__ == "__main__":
    main()
