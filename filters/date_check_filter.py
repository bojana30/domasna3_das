import logging
from datetime import datetime


class DateCheckFilter:
    def __init__(self, csv_manager):
        self.csv_manager = csv_manager
        self.logger = logging.getLogger(__name__)

    def process(self, cryptocurrencies):
        """Filter 2: Check last date of available data for each cryptocurrency"""
        self.logger.info("Starting date check filter")

        crypto_date_info = []

        for crypto in cryptocurrencies:
            crypto_id = crypto['id']

            # Check if we have historical data for this crypto
            last_date = self.csv_manager.get_last_historical_date(crypto_id)

            if last_date:
                # Check if we need to update (data is more than 1 day old)
                last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')
                current_date = datetime.now()

                needs_update = (current_date - last_date_dt).days > 1
            else:
                # No data exists, need full download
                needs_update = True

            crypto_date_info.append({
                'crypto': crypto,
                'last_date': last_date,
                'needs_update': needs_update
            })

        # Log statistics
        needs_update = len([c for c in crypto_date_info if c['needs_update']])
        self.logger.info(f"Date check completed: {needs_update}/{len(crypto_date_info)} need updates")

        return crypto_date_info