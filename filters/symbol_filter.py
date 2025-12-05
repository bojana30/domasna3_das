import requests
import logging
from config import COINGECKO_API_URL, MAX_CRYPTOCURRENCIES


class SymbolFilter:
    def __init__(self, csv_manager):
        self.csv_manager = csv_manager
        self.logger = logging.getLogger(__name__)

    def process(self, input_data=None):
        """Filter 1: Get top active cryptocurrencies with enhanced validation"""
        self.logger.info("Starting symbol filter: Fetching top cryptocurrencies")

        try:
            # Try to load existing symbols first
            existing_symbols = self.csv_manager.load_symbols()
            if existing_symbols and len(existing_symbols) >= MAX_CRYPTOCURRENCIES:
                self.logger.info(f"Using existing symbols: {len(existing_symbols)} cryptocurrencies")
                return existing_symbols[:MAX_CRYPTOCURRENCIES]

            # Fetch new symbols from CoinGecko with multiple pages
            symbols = self._fetch_symbols_with_pagination()
            filtered_symbols = self._filter_valid_symbols(symbols)

            # Save to CSV
            self.csv_manager.save_symbols(filtered_symbols)

            self.logger.info(f"Symbol filter completed: Found {len(filtered_symbols)} valid cryptocurrencies")
            return filtered_symbols

        except Exception as e:
            self.logger.error(f"Error in symbol filter: {e}")
            # Return existing symbols if available
            existing_symbols = self.csv_manager.load_symbols()
            if existing_symbols:
                self.logger.info("Returning existing symbols due to error")
                return existing_symbols[:MAX_CRYPTOCURRENCIES]
            raise

    def _fetch_symbols_with_pagination(self):
        """Fetch symbols from CoinGecko API with pagination for more data"""
        all_symbols = []
        page = 1
        per_page = 250  # Max per page

        while len(all_symbols) < MAX_CRYPTOCURRENCIES:
            url = f"{COINGECKO_API_URL}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': page,
                'sparkline': False
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            page_symbols = response.json()
            if not page_symbols:
                break  # No more data

            all_symbols.extend(page_symbols)
            page += 1

            # Avoid hitting rate limits
            import time
            time.sleep(1)

        return all_symbols[:MAX_CRYPTOCURRENCIES * 2]  # Get extra for filtering

    def _filter_valid_symbols(self, symbols_data):
        """Enhanced filtering based on homework requirements"""
        valid_symbols = []
        seen_symbols = set()

        for symbol_data in symbols_data:
            if len(valid_symbols) >= MAX_CRYPTOCURRENCIES:
                break

            if (self._is_valid_symbol(symbol_data) and
                    symbol_data['symbol'].upper() not in seen_symbols):
                valid_symbols.append({
                    'id': symbol_data['id'],
                    'symbol': symbol_data['symbol'].upper(),
                    'name': symbol_data['name'],
                    'market_cap_rank': symbol_data['market_cap_rank'],
                    'current_price': symbol_data.get('current_price'),
                    'market_cap': symbol_data.get('market_cap'),
                    'total_volume': symbol_data.get('total_volume'),
                    'price_change_percentage_24h': symbol_data.get('price_change_percentage_24h'),
                    'last_updated': symbol_data.get('last_updated')
                })
                seen_symbols.add(symbol_data['symbol'].upper())

        return valid_symbols

    def _is_valid_symbol(self, symbol_data):
        """Enhanced validation based on homework requirements"""
        # Basic validation
        if not symbol_data.get('id') or not symbol_data.get('symbol'):
            return False

        # Exclude delisted or inactive (based on market cap)
        if symbol_data.get('market_cap') is None or symbol_data.get('market_cap', 0) < 500000:
            return False

        # Exclude low liquidity
        if symbol_data.get('total_volume') is None or symbol_data.get('total_volume', 0) < 10000:
            return False

        # Exclude unstable quote currencies and duplicates
        symbol_lower = symbol_data['symbol'].lower()
        unstable_currencies = ['usdt', 'usdc', 'busd', 'dai']  # Stablecoins as quote currencies
        if any(currency in symbol_lower for currency in unstable_currencies):
            return False

        return True