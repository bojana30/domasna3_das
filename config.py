import os
from datetime import datetime, timedelta

# API Configuration
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# File Storage Configuration
DATA_DIR = "data"
SYMBOLS_DIR = os.path.join(DATA_DIR, "symbols")
HISTORICAL_DIR = os.path.join(DATA_DIR, "historical")
METRICS_DIR = os.path.join(DATA_DIR, "metrics")

# Application Settings
MAX_CRYPTOCURRENCIES = 1000  # Start with 100 for testing
HISTORICAL_YEARS = 10
START_DATE = (datetime.now() - timedelta(days=HISTORICAL_YEARS * 365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Request Settings
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 7

# CSV Configuration
CSV_ENCODING = 'utf-8'
CSV_DELIMITER = ','  # Fixed the delimiter