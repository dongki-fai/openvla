import os
import csv
import time
import torch

class SystemMonitor:
    def __init__(self, log_dir="rollouts", task="unknown_task"):
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, "system_metrics.csv")
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
        torch.cuda.reset_peak_memory_stats()
        self._start_time = time.time()

    def stop_and_log(self):
        latency = time.time() - self._start_time
        vram_bytes = torch.cuda.max_memory_allocated()
        vram_gb = vram_bytes / (1024 ** 3)

        row = {
            "task": self.task,
            "episode": self.episode,
            "step": self.step,
            "latency_s": latency,
            "vram_gb": vram_gb,
        }

        self.append_to_csv(row)
        self.step += 1

    def append_to_csv(self, row):
        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "episode", "step", "latency_s", "vram_gb"])
            if not file_exists or os.stat(self.log_path).st_size == 0:
                writer.writeheader()
            writer.writerow(row)