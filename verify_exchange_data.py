import requests
import json
from config import COINGECKO_API_URL


def verify_exchange_coverage():
    """Verify that data comes from international exchanges"""

    # Test with Bitcoin
    url = f"{COINGECKO_API_URL}/coins/bitcoin"
    params = {
        'localization': 'false',
        'tickers': 'true',
        'market_data': 'true'
    }

    response = requests.get(url, params=params)
    data = response.json()

    print(" VERIFYING EXCHANGE DATA COVERAGE")
    print("=" * 50)

    # Check tickers (exchange data)
    tickers = data.get('tickers', [])
    exchanges = set()

    for ticker in tickers:
        exchange = ticker.get('market', {}).get('name')
        if exchange:
            exchanges.add(exchange)

    print(f" Total exchanges for Bitcoin: {len(exchanges)}")
    print(f" Top exchanges:")

    # Check for required international exchanges
    required_exchanges = ['Binance', 'Coinbase', 'Kraken', 'Bitfinex', 'Huobi']
    found_exchanges = []

    for req_exchange in required_exchanges:
        found = any(req_exchange.lower() in ex.lower() for ex in exchanges)
        status = "yes" if found else "no"
        found_exchanges.append((req_exchange, found))
        print(f"   {status} {req_exchange}")

    print(f"\n Market Data Available:")
    print(f"   Current Price: ${data['market_data']['current_price']['usd']:,.2f}")
    print(f"   24h Volume: ${data['market_data']['total_volume']['usd']:,.0f}")
    print(f"   Market Cap: ${data['market_data']['market_cap']['usd']:,.0f}")

    # Check if we meet international exchange requirement
    international_coverage = sum(1 for _, found in found_exchanges if found)
    if international_coverage >= 3:  # At least 3 major exchanges
        print(f"\n INTERNATIONAL EXCHANGE REQUIREMENT: MET ")
    else:
        print(f"\n⚠ INTERNATIONAL EXCHANGE REQUIREMENT: NOT MET ")


if __name__ == "__main__":
    verify_exchange_coverage()