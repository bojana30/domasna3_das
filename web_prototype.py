import time
import logging
import sys
import os
from flask import Flask, render_template, request, jsonify
import math

# Import your existing modules
from utils.csv_manager import CSVManager
from utils.timer import PerformanceTimer
from filters.symbol_filter import SymbolFilter
from filters.date_check_filter import DateCheckFilter
from filters.data_fill_filter import DataFillFilter

# Import Technical Analysis module
try:
    from analysis.technical_analyzer import TechnicalAnalyzer

    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TECHNICAL_ANALYSIS_AVAILABLE = False
    print("⚠️  Technical Analysis module not found. Run: pip install pandas-ta numpy")


class CryptoExchangeProcessor:
    def __init__(self):
        self.csv_manager = CSVManager()
        self.symbol_filter = SymbolFilter(self.csv_manager)
        self.date_check_filter = DateCheckFilter(self.csv_manager)
        self.data_fill_filter = DataFillFilter(self.csv_manager)

        if TECHNICAL_ANALYSIS_AVAILABLE:
            self.technical_analyzer = TechnicalAnalyzer()
        else:
            self.technical_analyzer = None

        self.timer = PerformanceTimer()
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crypto_exchange_analyzer.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

    def run_pipe_and_filter(self):
        """Main Pipe and Filter Architecture Implementation"""
        self.logger.info("Starting Crypto Exchange Analyzer - Pipe and Filter Architecture")

        try:
            with self.timer.measure_time("Complete Crypto Exchange Data Pipeline"):
                # FILTER 1: Get top cryptocurrencies
                self.logger.info(" FILTER 1: Downloading and validating cryptocurrency symbols")
                symbols = self.symbol_filter.process()
                self.logger.info(f" FILTER 1 COMPLETED: {len(symbols)} valid symbols retrieved")

                # FILTER 2: Check last date
                self.logger.info(" FILTER 2: Checking existing data dates and update requirements")
                date_info = self.date_check_filter.process(symbols)
                needs_update = len([c for c in date_info if c['needs_update']])
                self.logger.info(f" FILTER 2 COMPLETED: {needs_update}/{len(symbols)} require updates")

                # FILTER 3: Fill missing data
                self.logger.info(" FILTER 3: Downloading and processing missing exchange data")
                result = self.data_fill_filter.process(date_info)
                self.logger.info(f" FILTER 3 COMPLETED: {result['success_count']} successful downloads")

            return self._create_success_result(result, len(symbols))

        except Exception as e:
            return self._create_error_result(str(e))

    def _create_success_result(self, result, total_symbols):
        elapsed = self.timer.get_elapsed_time()
        return {
            'status': 'success',
            'elapsed_time': elapsed,
            'total_symbols': total_symbols,
            'processed_count': result['processed_count'],
            'success_count': result['success_count'],
            'performance_metrics': self._calculate_performance_metrics(elapsed, total_symbols)
        }

    def _create_error_result(self, error):
        return {'status': 'error', 'error': error}

    def _calculate_performance_metrics(self, elapsed_time, total_symbols):
        """Calculate performance metrics for optimization challenge"""
        return {
            'total_seconds': elapsed_time,
            'cryptocurrencies_per_second': total_symbols / elapsed_time if elapsed_time > 0 else 0,
            'efficiency_score': (total_symbols / elapsed_time) * 1000 if elapsed_time > 0 else 0
        }

    def search_crypto_data(self, search_term):
        """Search for cryptocurrency data by symbol or name"""
        try:
            # Load all symbols
            symbols = self.csv_manager.load_symbols()

            # Search for matching cryptocurrencies (case-insensitive)
            search_term_lower = search_term.lower()
            matches = []

            for crypto in symbols:
                if (search_term_lower in crypto['symbol'].lower() or
                        search_term_lower in crypto['name'].lower() or
                        search_term_lower in crypto['id'].lower()):

                    # Get historical data
                    historical_data = self._get_crypto_historical_data(crypto['id'])

                    # Get metrics data
                    metrics_data = self._get_crypto_metrics_data(crypto['id'])

                    # Clean the crypto info data as well
                    cleaned_crypto = {}
                    for key, value in crypto.items():
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            cleaned_crypto[key] = None
                        else:
                            cleaned_crypto[key] = value

                    matches.append({
                        'symbol_info': cleaned_crypto,
                        'historical_data': historical_data,
                        'metrics_data': metrics_data
                    })

            return matches

        except Exception as e:
            self.logger.error(f"Error searching crypto data: {e}")
            return []

    def _get_crypto_historical_data(self, crypto_id):
        """Get historical data for a specific cryptocurrency"""
        try:
            filename = os.path.join('data', 'historical', f"{crypto_id}_historical.csv")
            if os.path.exists(filename):
                import pandas as pd
                import math
                df = pd.read_csv(filename)

                # Convert DataFrame to list of dictionaries and handle NaN values
                data = []
                for record in df.to_dict('records'):
                    cleaned_record = {}
                    for key, value in record.items():
                        # Convert NaN, None, or pandas NA to None
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            cleaned_record[key] = None
                        else:
                            cleaned_record[key] = value
                    data.append(cleaned_record)

                return data
            return []
        except Exception as e:
            self.logger.error(f"Error reading historical data for {crypto_id}: {e}")
            return []

    def _get_crypto_metrics_data(self, crypto_id):
        """Get metrics data for a specific cryptocurrency"""
        try:
            filename = os.path.join('data', 'metrics', f"{crypto_id}_metrics.csv")
            if os.path.exists(filename):
                import pandas as pd
                import math
                df = pd.read_csv(filename)

                # Convert DataFrame to list of dictionaries and handle NaN values
                data = []
                for record in df.to_dict('records'):
                    cleaned_record = {}
                    for key, value in record.items():
                        # Convert NaN, None, or pandas NA to None
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            cleaned_record[key] = None
                        else:
                            cleaned_record[key] = value
                    data.append(cleaned_record)

                return data
            return []
        except Exception as e:
            self.logger.error(f"Error reading metrics data for {crypto_id}: {e}")
            return []

    def get_all_cryptocurrencies(self):
        """Get list of all available cryptocurrencies"""
        try:
            symbols = self.csv_manager.load_symbols()
            return symbols
        except Exception as e:
            self.logger.error(f"Error getting all cryptocurrencies: {e}")
            return []

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        import numpy as np
        import pandas as pd

        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(v) for v in obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        else:
            return obj

    def perform_technical_analysis(self, crypto_id, time_frame='daily'):
        """Perform technical analysis for a cryptocurrency"""
        if not TECHNICAL_ANALYSIS_AVAILABLE or not self.technical_analyzer:
            return None

        try:
            # Get historical data
            historical_data = self._get_crypto_historical_data(crypto_id)

            if not historical_data or len(historical_data) < 50:
                self.logger.warning(f"Insufficient data for technical analysis of {crypto_id}")
                return None

            # Perform analysis
            import pandas as pd
            df = pd.DataFrame(historical_data)

            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Missing required column {col} for {crypto_id}")
                    return None

            analysis_df = self.technical_analyzer.calculate_indicators(
                df, time_frame
            )

            if analysis_df is None or analysis_df.empty:
                self.logger.error(f"Technical analysis failed for {crypto_id}")
                return None

            # Get summary
            summary = self.technical_analyzer.get_analysis_summary(analysis_df)

            # Convert DataFrame to dict for JSON response
            analysis_dict = analysis_df.to_dict('records')

            # Get indicators summary
            indicators_summary = self._get_indicators_summary(analysis_df)

            analysis_result = {
                'crypto_id': crypto_id,
                'time_frame': time_frame,
                'data_points': len(analysis_dict),
                'analysis_data': analysis_dict[-100:],  # Last 100 points
                'summary': summary,
                'indicators_summary': indicators_summary
            }

            # Convert numpy types to Python types
            return self._convert_numpy_types(analysis_result)

        except Exception as e:
            self.logger.error(f"Error in technical analysis for {crypto_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_indicators_summary(self, df):
        """Get summary of all indicators"""
        if df.empty:
            return {}

        summary = {}
        indicator_columns = [
            'RSI', 'MACD', 'MACD_signal', 'STOCH_K', 'STOCH_D',
            'ADX', 'CCI', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26', 'WMA_20', 'BB_upper', 'BB_middle', 'BB_lower', 'VMA_20'
        ]

        for indicator in indicator_columns:
            if indicator in df.columns:
                # Get last valid value
                last_value = df[indicator].dropna().iloc[-1] if not df[indicator].dropna().empty else None
                if last_value is not None:
                    # Convert to Python type using our converter
                    converted_value = self._convert_numpy_types(last_value)

                    summary[indicator] = {
                        'value': round(converted_value, 4) if isinstance(converted_value,
                                                                         (int, float)) else converted_value,
                        'interpretation': self._interpret_indicator(indicator, converted_value)
                    }

        return summary

    def _interpret_indicator(self, indicator, value):
        """Interpret indicator values"""
        if indicator == 'RSI':
            if value < 30:
                return f'Oversold ({value:.2f})'
            elif value > 70:
                return f'Overbought ({value:.2f})'
            else:
                return f'Neutral ({value:.2f})'
        elif indicator == 'STOCH_K':
            if value < 20:
                return f'Oversold ({value:.2f})'
            elif value > 80:
                return f'Overbought ({value:.2f})'
            else:
                return f'Neutral ({value:.2f})'
        elif indicator == 'ADX':
            if value > 25:
                return f'Strong Trend ({value:.2f})'
            else:
                return f'Weak Trend ({value:.2f})'
        elif indicator == 'CCI':
            if value > 100:
                return f'Overbought ({value:.2f})'
            elif value < -100:
                return f'Oversold ({value:.2f})'
            else:
                return f'Neutral ({value:.2f})'
        elif indicator == 'MACD':
            if value > 0:
                return f'Bullish ({value:.4f})'
            else:
                return f'Bearish ({value:.4f})'
        elif indicator == 'MACD_signal':
            return f'Signal Line ({value:.4f})'
        elif 'SMA' in indicator or 'EMA' in indicator or 'WMA' in indicator:
            return f'Moving Average ({value:.2f})'
        elif 'BB_' in indicator:
            if 'upper' in indicator:
                return f'Upper Band ({value:.2f})'
            elif 'lower' in indicator:
                return f'Lower Band ({value:.2f})'
            else:
                return f'Middle Band ({value:.2f})'

        return f'Value: {value}'


# Flask Web Application
app = Flask(__name__)
processor = CryptoExchangeProcessor()

# Global variable to track if data has been loaded
data_loaded = False


def load_initial_data():
    """Load initial cryptocurrency data"""
    global data_loaded
    if not data_loaded:
        print("Loading initial cryptocurrency data...")
        try:
            # Try to load existing data first
            symbols = processor.csv_manager.load_symbols()
            if symbols:
                print(f"✓ Loaded {len(symbols)} existing cryptocurrencies")
                data_loaded = True
                return True
            else:
                print("No existing data found. Running data collection pipeline...")
                result = processor.run_pipe_and_filter()
                if result['status'] == 'success':
                    print(f"✓ Data collection completed: {result['success_count']} cryptocurrencies")
                    data_loaded = True
                    return True
                else:
                    print(f"✗ Data collection failed: {result['error']}")
                    return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    return True


@app.route('/')
def index():
    """Main page with search interface"""
    # Ensure data is loaded
    load_initial_data()
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Search for cryptocurrencies"""
    search_term = request.form.get('search_term', '').strip()

    if not search_term:
        return jsonify({'error': 'Please enter a search term'})

    try:
        results = processor.search_crypto_data(search_term)

        if not results:
            return jsonify({'error': f'No cryptocurrencies found for "{search_term}"'})

        # Format results for display
        formatted_results = []
        for result in results:
            crypto = result['symbol_info']
            historical_count = len(result['historical_data'])
            metrics_count = len(result['metrics_data'])

            # Format price nicely
            current_price = crypto.get('current_price')
            if current_price and isinstance(current_price, (int, float)):
                formatted_price = f"${current_price:,.2f}"
            else:
                formatted_price = 'N/A'

            formatted_results.append({
                'id': crypto['id'],
                'symbol': crypto['symbol'],
                'name': crypto['name'],
                'market_cap_rank': crypto.get('market_cap_rank', 'N/A'),
                'current_price': formatted_price,
                'historical_records': historical_count,
                'metrics_records': metrics_count,
                'last_updated': crypto.get('last_updated', 'N/A')
            })

        return jsonify({'results': formatted_results})

    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'})


@app.route('/crypto/<crypto_id>')
def crypto_details_page(crypto_id):
    """Page showing detailed cryptocurrency data"""
    return render_template('crypto_details.html', crypto_id=crypto_id)


@app.route('/api/crypto/<crypto_id>')
def get_crypto_details_api(crypto_id):
    """API endpoint to get detailed cryptocurrency data"""
    try:
        historical_data = processor._get_crypto_historical_data(crypto_id)
        metrics_data = processor._get_crypto_metrics_data(crypto_id)

        # Get symbol info for the header
        symbols = processor.get_all_cryptocurrencies()
        crypto_info = next((crypto for crypto in symbols if crypto['id'] == crypto_id), None)

        return jsonify({
            'crypto_info': crypto_info,
            'historical_data': historical_data,
            'metrics_data': metrics_data
        })
    except Exception as e:
        return jsonify({'error': f'Error loading details: {str(e)}'})


@app.route('/analysis/<crypto_id>')
def analysis_page(crypto_id):
    """Technical analysis page for a cryptocurrency"""
    return render_template('analysis.html', crypto_id=crypto_id)


@app.route('/api/analysis/<crypto_id>')
def get_technical_analysis(crypto_id):
    """Get technical analysis data for a cryptocurrency"""
    try:
        # Get analysis for all time frames
        daily_analysis = processor.perform_technical_analysis(crypto_id, 'daily')
        weekly_analysis = processor.perform_technical_analysis(crypto_id, 'weekly')
        monthly_analysis = processor.perform_technical_analysis(crypto_id, 'monthly')

        # Get crypto info
        symbols = processor.get_all_cryptocurrencies()
        crypto_info = next((c for c in symbols if c['id'] == crypto_id), None)

        analysis_data = {
            'crypto_info': processor._convert_numpy_types(crypto_info) if crypto_info else None,
            'daily_analysis': daily_analysis,
            'weekly_analysis': weekly_analysis,
            'monthly_analysis': monthly_analysis,
            'analysis_summary': {
                'total_indicators': 10,
                'oscillators': ['RSI', 'MACD', 'Stochastic', 'ADX', 'CCI'],
                'moving_averages': ['SMA', 'EMA', 'WMA', 'Bollinger Bands', 'Volume MA']
            },
            'technical_analysis_available': TECHNICAL_ANALYSIS_AVAILABLE
        }

        return jsonify(processor._convert_numpy_types(analysis_data))

    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'})


@app.route('/api/analysis/top/<int:top_n>')
def get_top_cryptos_analysis(top_n):
    """Get top cryptocurrencies based on technical analysis"""
    if not TECHNICAL_ANALYSIS_AVAILABLE:
        return jsonify({'error': 'Technical analysis not available. Install pandas-ta.'})

    try:
        symbols = processor.get_all_cryptocurrencies()
        analysis_data = {}

        # Analyze top N cryptocurrencies
        for crypto in symbols[:20]:  # Limit to 20 for performance
            analysis = processor.perform_technical_analysis(crypto['id'], 'daily')
            if analysis:
                analysis_data[crypto['id']] = analysis

        # Get top ranked cryptos using the analyzer's method if available
        top_cryptos = []
        if hasattr(processor.technical_analyzer, 'get_top_cryptocurrencies'):
            top_cryptos = processor.technical_analyzer.get_top_cryptocurrencies(
                symbols, analysis_data, top_n
            )

        # Add crypto info to results
        for crypto in top_cryptos:
            crypto_info = next((c for c in symbols if c['id'] == crypto['crypto_id']), None)
            if crypto_info:
                crypto.update({
                    'symbol': crypto_info['symbol'],
                    'name': crypto_info['name'],
                    'market_cap_rank': crypto_info.get('market_cap_rank')
                })

        return jsonify(processor._convert_numpy_types({
            'top_cryptos': top_cryptos,
            'analysis_criteria': 'Based on technical indicators and buy/sell signals'
        }))

    except Exception as e:
        return jsonify({'error': f'Top analysis error: {str(e)}'})


@app.route('/api/cryptos')
def get_all_cryptos():
    """API endpoint to get all cryptocurrencies (for debugging)"""
    try:
        cryptos = processor.get_all_cryptocurrencies()
        return jsonify({
            'total': len(cryptos),
            'cryptocurrencies': cryptos[:20]  # Limit for performance
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/status')
def status():
    """Check application status"""
    symbols_count = len([f for f in os.listdir('data/symbols') if f.endswith('.csv')]) if os.path.exists(
        'data/symbols') else 0
    historical_count = len([f for f in os.listdir('data/historical') if f.endswith('.csv')]) if os.path.exists(
        'data/historical') else 0
    metrics_count = len([f for f in os.listdir('data/metrics') if f.endswith('.csv')]) if os.path.exists(
        'data/metrics') else 0

    return jsonify({
        'status': 'running',
        'data_loaded': data_loaded,
        'symbols_count': symbols_count,
        'historical_files': historical_count,
        'metrics_files': metrics_count,
        'technical_analysis_available': TECHNICAL_ANALYSIS_AVAILABLE
    })


def display_startup_banner():
    """Display startup information"""
    print("\n" + "=" * 70)
    print(" CRYPTO EXCHANGE ANALYZER - HOMEWORK 3")
    print("=" * 70)
    print(" Web Interface with Technical Analysis")
    print(" Features:")
    print("   • Search cryptocurrencies by symbol, name, or ID")
    print("   • View historical data and metrics")
    print("   • Technical Analysis with 10 indicators")
    print("   • Buy/Sell/Hold signals")
    print("   • 3 Time Frames: Daily, Weekly, Monthly")

    if TECHNICAL_ANALYSIS_AVAILABLE:
        print("    Technical Analysis: ENABLED")
    else:
        print("     Technical Analysis: DISABLED (install pandas-ta)")

    print(" Server will start at: http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    # Display startup banner
    display_startup_banner()

    # Load initial data
    print("\n Initializing data...")
    if load_initial_data():
        print(" Data loaded successfully!")

        # Display data statistics
        symbols = processor.get_all_cryptocurrencies()
        print(f" Available cryptocurrencies: {len(symbols)}")

        historical_files = len([f for f in os.listdir('data/historical') if f.endswith('.csv')]) if os.path.exists(
            'data/historical') else 0
        metrics_files = len([f for f in os.listdir('data/metrics') if f.endswith('.csv')]) if os.path.exists(
            'data/metrics') else 0

        print(f" Historical data files: {historical_files}")
        print(f" Metrics data files: {metrics_files}")

        if TECHNICAL_ANALYSIS_AVAILABLE:
            print(" Technical Analysis: READY")
        else:
            print("  Install for Technical Analysis: pip install pandas-ta numpy")

        print("\n Ready to search! Open http://localhost:5000 in your browser")
        print("   Try: Search 'BTC' → Click 'Analyze' for Technical Analysis")
        print("=" * 70)

        # Start the web server
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(" Failed to load data. Please check your configuration and try again.")
        sys.exit(1)