import logging
from utils.csv_manager import CSVManager
from utils.timer import PerformanceTimer
from filters.symbol_filter import SymbolFilter
from filters.date_check_filter import DateCheckFilter
from filters.data_fill_filter import DataFillFilter


class Pipeline:
    def __init__(self):
        self.csv_manager = CSVManager()
        self.timer = PerformanceTimer()
        self.logger = logging.getLogger(__name__)

        # Initialize filters
        self.filters = [
            SymbolFilter(self.csv_manager),
            DateCheckFilter(self.csv_manager),
            DataFillFilter(self.csv_manager)
        ]

    def execute(self):
        """Execute the complete pipe and filter pipeline"""
        self.logger.info("Starting pipeline execution...")

        data = None
        for i, filter_obj in enumerate(self.filters, 1):
            self.logger.info(f"Executing Filter {i}...")
            data = filter_obj.process(data)
            self.logger.info(f"Filter {i} completed")

        return data


# Example usage
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.execute()