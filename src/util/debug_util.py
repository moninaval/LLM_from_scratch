# src/utils/debug_utils.py

class DebugLogger:
    def __init__(self, enabled=True, prefix="[DEBUG]"):
        self.enabled = enabled
        self.prefix = prefix

    def log(self, message):
        if self.enabled:
            print(f"{self.prefix} {message}")
