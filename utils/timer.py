import time
import logging
from contextlib import contextmanager


class PerformanceTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def measure_time(self, operation_name="Operation"):
        """Context manager to measure execution time"""
        self.start_time = time.time()
        self.logger.info(f"Starting {operation_name}...")

        try:
            yield self
        finally:
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            self.logger.info(f"{operation_name} completed in {elapsed_time:.2f} seconds")

    def get_elapsed_time(self):
        """Get elapsed time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None