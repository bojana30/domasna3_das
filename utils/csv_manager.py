import pandas as pd
import os
from datetime import datetime
import logging
from config import SYMBOLS_DIR, HISTORICAL_DIR, METRICS_DIR, CSV_ENCODING, CSV_DELIMITER


class CSVManager:
    def __init__(self):
        self._ensure_directories()
        self.logger = logging.getLogger(__name__)

    def _ensure_directories(self):
        """Ensure all data directories exist"""
        os.makedirs(SYMBOLS_DIR, exist_ok=True)
        os.makedirs(HISTORICAL_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    def save_symbols(self, symbols):
        """Save cryptocurrency symbols to CSV"""
        filename = os.path.join(SYMBOLS_DIR, f"crypto_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        df = pd.DataFrame(symbols)
        df.to_csv(filename, index=False, encoding=CSV_ENCODING)
        self.logger.info(f"Saved {len(symbols)} symbols to {filename}")
        return filename

    def get_last_symbols_file(self):
        """Get the most recent symbols CSV file"""
        try:
            files = [f for f in os.listdir(SYMBOLS_DIR) if f.endswith('.csv')]
            if not files:
                return None
            latest_file = max(files)
            return os.path.join(SYMBOLS_DIR, latest_file)
        except Exception as e:
            self.logger.error(f"Error getting latest symbols file: {e}")
            return None

    def load_symbols(self):
        """Load symbols from the latest CSV file"""
        symbols_file = self.get_last_symbols_file()
        if not symbols_file:
            return []

        try:
            df = pd.read_csv(symbols_file, encoding=CSV_ENCODING)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Error loading symbols: {e}")
            return []

    def save_historical_data(self, crypto_id, historical_data):
        """Save historical data for a cryptocurrency"""
        if not historical_data:
            return None

        filename = os.path.join(HISTORICAL_DIR, f"{crypto_id}_historical.csv")

        # Check if file exists to append or create new
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename, encoding=CSV_ENCODING)
                new_df = pd.DataFrame(historical_data)

                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
                combined_df.to_csv(filename, index=False, encoding=CSV_ENCODING)
            except Exception as e:
                self.logger.error(f"Error appending historical data for {crypto_id}: {e}")
                # Save as new file if append fails
                df = pd.DataFrame(historical_data)
                df.to_csv(filename, index=False, encoding=CSV_ENCODING)
        else:
            df = pd.DataFrame(historical_data)
            df.to_csv(filename, index=False, encoding=CSV_ENCODING)

        self.logger.info(f"Saved {len(historical_data)} historical records for {crypto_id}")
        return filename

    def save_daily_metrics(self, crypto_id, metrics_data):
        """Save daily metrics for a cryptocurrency"""
        if not metrics_data:
            return None

        filename = os.path.join(METRICS_DIR, f"{crypto_id}_metrics.csv")

        # Convert single metric to list for consistency
        if isinstance(metrics_data, dict):
            metrics_data = [metrics_data]

        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename, encoding=CSV_ENCODING)
                new_df = pd.DataFrame(metrics_data)

                # Combine and remove duplicates
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date'], keep='last')
                combined_df = combined_df.sort_values('date')
                combined_df.to_csv(filename, index=False, encoding=CSV_ENCODING)
            except Exception as e:
                self.logger.error(f"Error appending metrics for {crypto_id}: {e}")
                # Save as new file if append fails
                df = pd.DataFrame(metrics_data)
                df.to_csv(filename, index=False, encoding=CSV_ENCODING)
        else:
            df = pd.DataFrame(metrics_data)
            df.to_csv(filename, index=False, encoding=CSV_ENCODING)

        self.logger.info(f"Saved {len(metrics_data)} metrics records for {crypto_id}")
        return filename

    def get_last_historical_date(self, crypto_id):
        """Get the last available date for a cryptocurrency's historical data"""
        filename = os.path.join(HISTORICAL_DIR, f"{crypto_id}_historical.csv")

        if not os.path.exists(filename):
            return None

        try:
            df = pd.read_csv(filename, encoding=CSV_ENCODING)
            if df.empty or 'date' not in df.columns:
                return None

            last_date = df['date'].max()
            return last_date
        except Exception as e:
            self.logger.error(f"Error reading last date for {crypto_id}: {e}")
            return None

    def crypto_historical_exists(self, crypto_id):
        """Check if historical data exists for a cryptocurrency"""
        filename = os.path.join(HISTORICAL_DIR, f"{crypto_id}_historical.csv")
        return os.path.exists(filename)