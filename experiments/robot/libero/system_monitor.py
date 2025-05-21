import os
import csv
import time
import torch

class SystemMonitor:
    def __init__(self, log_dir="rollouts", task="unknown_task"):
        self.log_dir = log_dir
        self.latency_log_path = os.path.join(self.log_dir, "system_metrics.csv")
        self.task = task
        self.episode = 0
        self.step = 0

        os.makedirs(self.log_dir, exist_ok=True)

        self.reset()

    def reset(self, episode=None):
        self.episode = 0
        self.step = 0
        self.entries = []

    def start_timing(self):
        self._start_time = time.time()

    def stop_and_log(self):
        latency = time.time() - self._start_time
        vram_bytes = torch.cuda.max_memory_allocated()
        vram_gb = vram_bytes / 1e9 

        self.entries.append({
            "task": self.task,
            "episode": self.episode,
            "step": self.step,
            "latency": latency,
            "vram_gb": vram_gb,
        })
        self.step += 1

    def export(self):
        write_header = not os.path.exists(self.latency_log_path)
        with open(self.latency_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "episode", "step", "latency_s", "vram_gb"])
            if write_header:
                writer.writeheader()
            for row in self.entries:
                writer.writerow(row)
