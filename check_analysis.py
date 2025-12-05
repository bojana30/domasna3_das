import pandas as pd
import numpy as np
import sys
import os

sys.path.append('.')

print("🔍 TESTING TECHNICAL ANALYSIS SETUP")
print("=" * 50)

try:
    from analysis.technical_analyzer import TechnicalAnalyzer

    print("✅ Technical Analyzer imported successfully!")

    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    sample_data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    })

    print(f"📊 Sample data created: {len(sample_data)} rows")

    # Test the analyzer
    analyzer = TechnicalAnalyzer()

    print("\n🧪 Testing indicator calculation...")
    result = analyzer.calculate_indicators(sample_data, 'daily')

    print(f"✅ Result shape: {result.shape}")
    print(f"✅ Columns: {len(result.columns)}")

    # Check for key indicators
    print("\n🔍 Checking for required indicators:")

    oscillators = ['RSI', 'MACD', 'STOCH_K', 'ADX', 'CCI']
    moving_averages = ['SMA_20', 'EMA_12', 'WMA_20', 'BB_upper', 'VMA_20']

    all_indicators = oscillators + moving_averages

    missing = []
    present = []

    for indicator in all_indicators:
        if indicator in result.columns:
            present.append(indicator)
        else:
            missing.append(indicator)

    print(f"✅ Present ({len(present)}): {', '.join(present[:5])}...")
    if missing:
        print(f"⚠️  Missing ({len(missing)}): {', '.join(missing)}")

    # Check signals
    if 'signal' in result.columns:
        signal_counts = result['signal'].value_counts()
        print(f"\n📈 Signal generation:")
        print(f"   Buy: {signal_counts.get('BUY', 0)}")
        print(f"   Sell: {signal_counts.get('SELL', 0)}")
        print(f"   Hold: {signal_counts.get('HOLD', 0)}")
    else:
        print("\n❌ No signals generated")

    # Test summary
    print("\n📊 Testing summary generation...")
    summary = analyzer.get_analysis_summary(result)

    if summary:
        print(f"✅ Summary generated:")
        print(f"   Latest signal: {summary.get('latest_signal', 'N/A')}")
        print(f"   Signal strength: {summary.get('latest_signal_strength', 'N/A')}")
        print(f"   Buy signals: {summary.get('buy_signals', 'N/A')}")
        print(f"   Indicators calculated: {len(summary.get('indicators', {}))}")
    else:
        print("❌ No summary generated")

    print("\n" + "=" * 50)
    print("🎯 TECHNICAL ANALYSIS READY FOR HOMEWORK 3!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n💡 Make sure to:")
    print("   1. Create analysis/technical_analyzer.py")
    print("   2. Install: pip install pandas-ta numpy")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()