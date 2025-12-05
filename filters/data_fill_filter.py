# import requests
# import time
# import logging
# from datetime import datetime
# from config import COINGECKO_API_URL, RATE_LIMIT_DELAY
#
#
# class DataFillFilter:
#     def __init__(self, csv_manager):
#         self.csv_manager = csv_manager
#         self.logger = logging.getLogger(__name__)
#
#     def process(self, crypto_date_info):
#         """Filter 3: Fill missing data for each cryptocurrency"""
#         self.logger.info("Starting data fill filter")
#
#         processed_count = 0
#         success_count = 0
#
#         for crypto_info in crypto_date_info:
#             if crypto_info['needs_update']:
#                 crypto = crypto_info['crypto']
#                 last_date = crypto_info['last_date']
#
#                 try:
#                     self.logger.info(f"Processing {crypto['id']} - {crypto['name']}")
#
#                     # Download historical data
#                     historical_data = self._download_historical_data(crypto['id'], last_date)
#
#                     # Download current metrics
#                     current_metrics = self._download_current_metrics(crypto['id'])
#
#                     # Save to CSV files
#                     if historical_data:
#                         self.csv_manager.save_historical_data(crypto['id'], historical_data)
#
#                     if current_metrics:
#                         self.csv_manager.save_daily_metrics(crypto['id'], current_metrics)
#
#                     success_count += 1
#                     self.logger.info(f"Successfully processed {crypto['id']}")
#
#                 except Exception as e:
#                     self.logger.error(f"Error processing {crypto['id']}: {e}")
#
#                 processed_count += 1
#
#                 # Rate limiting
#                 time.sleep(RATE_LIMIT_DELAY)
#
#         self.logger.info(f"Data fill completed: {success_count} successful")
#         return {
#             'processed_count': processed_count,
#             'success_count': success_count
#         }
#
#     def _download_historical_data(self, crypto_id, last_date):
#         """Download historical OHLCV data from CoinGecko"""
#         try:
#             url = f"{COINGECKO_API_URL}/coins/{crypto_id}/ohlc"
#
#             # Determine days parameter
#             days_param = 'max' if not last_date else '30'
#
#             params = {
#                 'vs_currency': 'usd',
#                 'days': days_param
#             }
#
#             response = requests.get(url, params=params, timeout=30)
#             response.raise_for_status()
#
#             ohlc_data = response.json()
#             formatted_data = self._format_ohlc_data(ohlc_data, last_date)
#
#             return formatted_data
#
#         except requests.RequestException as e:
#             self.logger.error(f"Error downloading historical data for {crypto_id}: {e}")
#             return []
#
#     def _format_ohlc_data(self, ohlc_data, last_date):
#         """Format OHLC data from CoinGecko API response"""
#         formatted_data = []
#
#         for ohlc in ohlc_data:
#             # CoinGecko OHLC format: [timestamp, open, high, low, close]
#             timestamp = ohlc[0] / 1000  # Convert from milliseconds to seconds
#             date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
#
#             # Skip if we already have this date
#             if last_date and date <= last_date:
#                 continue
#
#             formatted_data.append({
#                 'date': date,
#                 'open': ohlc[1],
#                 'high': ohlc[2],
#                 'low': ohlc[3],
#                 'close': ohlc[4],
#                 'volume': None
#             })
#
#         return formatted_data
#
#     def _download_current_metrics(self, crypto_id):
#         """Download current market metrics"""
#         try:
#             url = f"{COINGECKO_API_URL}/coins/{crypto_id}"
#             params = {
#                 'localization': 'false',
#                 'tickers': 'false',
#                 'market_data': 'true',
#                 'community_data': 'false',
#                 'developer_data': 'false'
#             }
#
#             response = requests.get(url, params=params, timeout=30)
#             response.raise_for_status()
#
#             coin_data = response.json()
#             return self._format_current_metrics(coin_data)
#
#         except requests.RequestException as e:
#             self.logger.error(f"Error downloading current metrics for {crypto_id}: {e}")
#             return None
#
#     def _format_current_metrics(self, coin_data):
#         """Format current market metrics"""
#         market_data = coin_data.get('market_data', {})
#
#         return {
#             'date': datetime.now().strftime('%Y-%m-%d'),
#             'price': market_data.get('current_price', {}).get('usd'),
#             'volume_24h': market_data.get('total_volume', {}).get('usd'),
#             'high_24h': market_data.get('high_24h', {}).get('usd'),
#             'low_24h': market_data.get('low_24h', {}).get('usd'),
#             'market_cap': market_data.get('market_cap', {}).get('usd')
#         }

import requests
import time
import logging
from datetime import datetime, timedelta
from config import COINGECKO_API_URL, RATE_LIMIT_DELAY, HISTORICAL_YEARS


class DataFillFilter:
    def __init__(self, csv_manager):
        self.csv_manager = csv_manager
        self.logger = logging.getLogger(__name__)

    def process(self, crypto_date_info):
        """Filter 3: Fill missing data for each cryptocurrency"""
        self.logger.info("Starting data fill filter")

        processed_count = 0
        success_count = 0

        for crypto_info in crypto_date_info:
            if crypto_info['needs_update']:
                crypto = crypto_info['crypto']
                last_date = crypto_info['last_date']

                try:
                    self.logger.info(f"Processing {crypto['id']} - {crypto['name']}")

                    # Download historical data - ALWAYS get max for new coins
                    if not last_date:
                        # For new coins, get maximum historical data
                        historical_data = self._download_historical_data(crypto['id'], None, 'max')
                    else:
                        # For existing coins, get data since last date
                        historical_data = self._download_historical_data(crypto['id'], last_date, 'max')

                    # Download current metrics
                    current_metrics = self._download_current_metrics(crypto['id'])

                    # Save to CSV files
                    if historical_data:
                        self.csv_manager.save_historical_data(crypto['id'], historical_data)
                        self.logger.info(f"Saved {len(historical_data)} historical records for {crypto['id']}")

                    if current_metrics:
                        self.csv_manager.save_daily_metrics(crypto['id'], current_metrics)

                    success_count += 1
                    self.logger.info(f"Successfully processed {crypto['id']}")

                except Exception as e:
                    self.logger.error(f"Error processing {crypto['id']}: {e}")

                processed_count += 1

                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)

        self.logger.info(f"Data fill completed: {success_count} successful")
        return {
            'processed_count': processed_count,
            'success_count': success_count
        }

    def _download_historical_data(self, crypto_id, last_date, days_param='max'):
        """Download historical OHLCV data from CoinGecko"""
        try:
            url = f"{COINGECKO_API_URL}/coins/{crypto_id}/ohlc"

            params = {
                'vs_currency': 'usd',
                'days': days_param
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            ohlc_data = response.json()

            # Check if we got valid data
            if not ohlc_data:
                self.logger.warning(f"No historical data available for {crypto_id}")
                return []

            formatted_data = self._format_ohlc_data(ohlc_data, last_date)

            if formatted_data:
                self.logger.info(f"Downloaded {len(formatted_data)} records for {crypto_id}")
            else:
                self.logger.info(f"No new historical data for {crypto_id}")

            return formatted_data

        except requests.RequestException as e:
            self.logger.error(f"Error downloading historical data for {crypto_id}: {e}")
            return []

    def _format_ohlc_data(self, ohlc_data, last_date):
        """Format OHLC data from CoinGecko API response"""
        formatted_data = []

        for ohlc in ohlc_data:
            # CoinGecko OHLC format: [timestamp, open, high, low, close, volume]
            # Note: CoinGecko OHLC doesn't include volume in this endpoint
            timestamp = ohlc[0] / 1000  # Convert from milliseconds to seconds
            date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')

            # Skip if we already have this date
            if last_date and date <= last_date:
                continue

            formatted_data.append({
                'date': date,
                'open': ohlc[1],
                'high': ohlc[2],
                'low': ohlc[3],
                'close': ohlc[4],
                'volume': ohlc[4] if len(ohlc) > 5 else None  # Some endpoints include volume
            })

        return formatted_data

    def _download_current_metrics(self, crypto_id):
        """Download current market metrics with exchange data"""
        try:
            url = f"{COINGECKO_API_URL}/coins/{crypto_id}"
            params = {
                'localization': 'false',
                'tickers': 'true',  # ← Enable tickers to get exchange information
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            coin_data = response.json()
            return self._format_current_metrics(coin_data)

        except requests.RequestException as e:
            self.logger.error(f"Error downloading current metrics for {crypto_id}: {e}")
            return None

    def _format_current_metrics(self, coin_data):
        """Format current market metrics with exchange information"""
        market_data = coin_data.get('market_data', {})

        # Get top exchanges from tickers
        exchanges = set()
        tickers = coin_data.get('tickers', [])
        for ticker in tickers[:10]:  # Check top 10 tickers
            exchange_name = ticker.get('market', {}).get('name')
            if exchange_name:
                exchanges.add(exchange_name)

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'price': market_data.get('current_price', {}).get('usd'),
            'volume_24h': market_data.get('total_volume', {}).get('usd'),
            'high_24h': market_data.get('high_24h', {}).get('usd'),
            'low_24h': market_data.get('low_24h', {}).get('usd'),
            'market_cap': market_data.get('market_cap', {}).get('usd'),
            'exchanges': list(exchanges)[:5]  # Include top 5 exchanges
        }