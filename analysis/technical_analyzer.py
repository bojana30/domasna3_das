import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import logging


class TechnicalAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_indicators(self, historical_data, time_frame='daily'):
        """
        Calculate technical indicators for cryptocurrency data
        Args:
            historical_data: List of dicts or DataFrame with OHLCV data
            time_frame: 'daily', 'weekly', or 'monthly'
        Returns:
            DataFrame with indicators
        """
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
        else:
            df = historical_data.copy()

        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove NaN values
        df = df.dropna(subset=['close'])

        if len(df) < 50:  # Need enough data for indicators
            self.logger.warning(f"Insufficient data points: {len(df)}")
            return df

        # Calculate all indicators
        df = self._calculate_all_indicators(df)

        # Generate trading signals
        df = self._generate_signals(df)

        # Time frame resampling if needed
        if time_frame != 'daily':
            df = self._resample_time_frame(df, time_frame)

        return df

    def _calculate_all_indicators(self, df):
        """Calculate all 10 technical indicators"""

        # OSCILLATORS (5 indicators)

        # 1. RSI (Relative Strength Index)
        df['RSI'] = ta.rsi(df['close'], length=14)

        # 2. MACD (Moving Average Convergence Divergence)
        macd = ta.macd(df['close'])
        if macd is not None and not macd.empty:
            # Check column names that pandas_ta returns
            macd_columns = macd.columns.tolist()
            if 'MACD_12_26_9' in macd_columns:
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_signal'] = macd['MACDs_12_26_9']
                df['MACD_histogram'] = macd['MACDh_12_26_9']
            elif 'MACD' in macd_columns:
                df['MACD'] = macd['MACD']
                df['MACD_signal'] = macd['MACDs']
                df['MACD_histogram'] = macd['MACDh']

        # 3. Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None and not stoch.empty:
            stoch_columns = stoch.columns.tolist()
            if 'STOCHk_14_3_3' in stoch_columns:
                df['STOCH_K'] = stoch['STOCHk_14_3_3']
                df['STOCH_D'] = stoch['STOCHd_14_3_3']
            elif 'STOCHk' in stoch_columns:
                df['STOCH_K'] = stoch['STOCHk']
                df['STOCH_D'] = stoch['STOCHd']

        # 4. ADX (Average Directional Index)
        adx = ta.adx(df['high'], df['low'], df['close'])
        if adx is not None and not adx.empty:
            adx_columns = adx.columns.tolist()
            if 'ADX_14' in adx_columns:
                df['ADX'] = adx['ADX_14']
                df['ADX_POS'] = adx['DMP_14']
                df['ADX_NEG'] = adx['DMN_14']
            elif 'ADX' in adx_columns:
                df['ADX'] = adx['ADX']
                df['ADX_POS'] = adx['DMP']
                df['ADX_NEG'] = adx['DMN']

        # 5. CCI (Commodity Channel Index)
        cci_result = ta.cci(df['high'], df['low'], df['close'], length=20)
        if cci_result is not None:
            df['CCI'] = cci_result

        # MOVING AVERAGES (5 indicators)

        # 6. SMA (Simple Moving Average)
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['SMA_200'] = ta.sma(df['close'], length=200)

        # 7. EMA (Exponential Moving Average)
        df['EMA_12'] = ta.ema(df['close'], length=12)
        df['EMA_26'] = ta.ema(df['close'], length=26)

        # 8. WMA (Weighted Moving Average)
        wma_result = ta.wma(df['close'], length=20)
        if wma_result is not None:
            df['WMA_20'] = wma_result

        # 9. Bollinger Bands - FIXED VERSION
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and not bb.empty:
            # Check for different column naming patterns
            bb_columns = bb.columns.tolist()

            # Try different possible column names
            upper_col = None
            middle_col = None
            lower_col = None

            # Common patterns in pandas_ta
            possible_upper = ['BBU_20_2.0', 'BBU_20_2', 'BBU', 'BB_UPPER']
            possible_middle = ['BBM_20_2.0', 'BBM_20_2', 'BBM', 'BB_MIDDLE']
            possible_lower = ['BBL_20_2.0', 'BBL_20_2', 'BBL', 'BB_LOWER']

            for col in possible_upper:
                if col in bb_columns:
                    upper_col = col
                    break

            for col in possible_middle:
                if col in bb_columns:
                    middle_col = col
                    break

            for col in possible_lower:
                if col in bb_columns:
                    lower_col = col
                    break

            # If still not found, use first 3 columns
            if not upper_col and len(bb_columns) >= 3:
                upper_col = bb_columns[0]
                middle_col = bb_columns[1]
                lower_col = bb_columns[2]

            if upper_col:
                df['BB_upper'] = bb[upper_col]
                df['BB_middle'] = bb[middle_col] if middle_col else bb[bb_columns[1]]
                df['BB_lower'] = bb[lower_col] if lower_col else bb[bb_columns[2]]

        # 10. Volume Moving Average
        if 'volume' in df.columns:
            vma_result = ta.sma(df['volume'], length=20)
            if vma_result is not None:
                df['VMA_20'] = vma_result

        return df

    def _generate_signals(self, df):
        """Generate buy/sell/hold signals based on indicators"""

        # Initialize signals as 'HOLD'
        df['signal'] = 'HOLD'
        df['signal_strength'] = 0

        if len(df) < 2:
            return df

        signals = []

        for i in range(1, len(df)):
            buy_signals = 0
            sell_signals = 0

            # RSI signals (30/70 levels)
            if 'RSI' in df.columns and not pd.isna(df.iloc[i]['RSI']):
                if df.iloc[i]['RSI'] < 30:
                    buy_signals += 1
                elif df.iloc[i]['RSI'] > 70:
                    sell_signals += 1

            # MACD signals
            if all(col in df.columns for col in ['MACD', 'MACD_signal']):
                if (not pd.isna(df.iloc[i]['MACD']) and
                        not pd.isna(df.iloc[i]['MACD_signal'])):
                    if (df.iloc[i]['MACD'] > df.iloc[i]['MACD_signal'] and
                            df.iloc[i - 1]['MACD'] <= df.iloc[i - 1]['MACD_signal']):
                        buy_signals += 1
                    elif (df.iloc[i]['MACD'] < df.iloc[i]['MACD_signal'] and
                          df.iloc[i - 1]['MACD'] >= df.iloc[i - 1]['MACD_signal']):
                        sell_signals += 1

            # Stochastic signals (20/80 levels)
            if 'STOCH_K' in df.columns and not pd.isna(df.iloc[i]['STOCH_K']):
                if df.iloc[i]['STOCH_K'] < 20:
                    buy_signals += 1
                elif df.iloc[i]['STOCH_K'] > 80:
                    sell_signals += 1

            # Bollinger Bands signals
            if all(col in df.columns for col in ['close', 'BB_lower', 'BB_upper']):
                if (not pd.isna(df.iloc[i]['close']) and
                        not pd.isna(df.iloc[i]['BB_lower']) and
                        not pd.isna(df.iloc[i]['BB_upper'])):
                    if df.iloc[i]['close'] < df.iloc[i]['BB_lower']:
                        buy_signals += 1
                    elif df.iloc[i]['close'] > df.iloc[i]['BB_upper']:
                        sell_signals += 1

            # Moving Average Crossover (EMA12/EMA26)
            if all(col in df.columns for col in ['EMA_12', 'EMA_26']):
                if (not pd.isna(df.iloc[i]['EMA_12']) and
                        not pd.isna(df.iloc[i]['EMA_26'])):
                    if (df.iloc[i]['EMA_12'] > df.iloc[i]['EMA_26'] and
                            df.iloc[i - 1]['EMA_12'] <= df.iloc[i - 1]['EMA_26']):
                        buy_signals += 1
                    elif (df.iloc[i]['EMA_12'] < df.iloc[i]['EMA_26'] and
                          df.iloc[i - 1]['EMA_12'] >= df.iloc[i - 1]['EMA_26']):
                        sell_signals += 1

            # Determine final signal
            if buy_signals > sell_signals:
                signal = 'BUY'
                strength = buy_signals - sell_signals
            elif sell_signals > buy_signals:
                signal = 'SELL'
                strength = sell_signals - buy_signals
            else:
                signal = 'HOLD'
                strength = 0

            signals.append(signal)
            df.at[df.index[i], 'signal_strength'] = strength

        # Add signals to dataframe (skip first row)
        if signals:
            df.loc[df.index[1:], 'signal'] = signals
            df.loc[df.index[0], 'signal'] = 'HOLD'

        return df

    def _resample_time_frame(self, df, time_frame):
        """Resample data to weekly or monthly timeframe"""
        if time_frame == 'weekly':
            # Resample to weekly data (last observation of the week)
            resampled = df.set_index('date').resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Recalculate indicators on weekly data
            resampled.reset_index(inplace=True)
            return self._calculate_all_indicators(resampled)

        elif time_frame == 'monthly':
            # Resample to monthly data
            resampled = df.set_index('date').resample('M').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Recalculate indicators on monthly data
            resampled.reset_index(inplace=True)
            return self._calculate_all_indicators(resampled)

        return df

    def get_analysis_summary(self, df):
        """Get summary statistics of technical analysis"""
        if df.empty:
            return {}

        summary = {
            'total_signals': len(df),
            'buy_signals': (df['signal'] == 'BUY').sum(),
            'sell_signals': (df['signal'] == 'SELL').sum(),
            'hold_signals': (df['signal'] == 'HOLD').sum(),
            'latest_signal': df.iloc[-1]['signal'] if len(df) > 0 else 'HOLD',
            'latest_signal_strength': df.iloc[-1]['signal_strength'] if len(df) > 0 else 0,
            'current_price': df.iloc[-1]['close'] if 'close' in df.columns and len(df) > 0 else None,
            'indicators': {}
        }

        # Add latest indicator values
        indicator_columns = ['RSI', 'MACD', 'STOCH_K', 'ADX', 'CCI',
                             'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'BB_upper', 'BB_lower']

        for indicator in indicator_columns:
            if indicator in df.columns and not pd.isna(df.iloc[-1][indicator]):
                summary['indicators'][indicator] = round(float(df.iloc[-1][indicator]), 4)

        return summary

    def get_top_cryptocurrencies(self, symbols_data, analysis_data, top_n=10):
        """Get top cryptocurrencies based on technical analysis signals"""
        rankings = []

        for crypto_id, data in analysis_data.items():
            if 'summary' in data and data['summary']:
                signal_score = 0
                summary = data['summary']

                # Score based on signals
                if summary['latest_signal'] == 'BUY':
                    signal_score = 3
                elif summary['latest_signal'] == 'SELL':
                    signal_score = -3
                else:
                    signal_score = 1

                # Adjust by signal strength
                signal_score += summary['latest_signal_strength'] * 0.5

                # Add RSI consideration
                if 'RSI' in summary['indicators']:
                    rsi = summary['indicators']['RSI']
                    if rsi < 30:  # Oversold
                        signal_score += 2
                    elif rsi > 70:  # Overbought
                        signal_score -= 2

                rankings.append({
                    'crypto_id': crypto_id,
                    'signal_score': signal_score,
                    'signal': summary['latest_signal'],
                    'current_price': summary['current_price'],
                    'summary': summary
                })

        # Sort by signal score (descending)
        rankings.sort(key=lambda x: x['signal_score'], reverse=True)

        return rankings[:top_n]