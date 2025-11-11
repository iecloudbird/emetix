"""
Test RF Ensemble predictions on real stocks
Compare with LSTM predictions and actual prices
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from config.logging_config import get_logger
from config.settings import MODELS_DIR
from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.data.fetchers.yfinance_fetcher import YFinanceFetcher

logger = get_logger(__name__)


def test_rf_predictions():
    """Test RF Ensemble on sample stocks"""
    
    # Load trained model
    model_path = MODELS_DIR / "rf_ensemble.pkl"
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return
    
    model = RFEnsembleModel()
    model.load(str(model_path))
    logger.info(f"✅ Loaded model from {model_path}\n")
    
    # Test stocks (same as LSTM tests)
    test_tickers = [
        'AAPL',   # Tech leader
        'MSFT',   # Software giant
        'GOOGL',  # High growth
        'JPM',    # Financial
        'JNJ',    # Healthcare
        'PG',     # Consumer defensive
        'TSLA',   # High volatility
        'AMD',    # Semiconductor
        'BA',     # Industrial (negative FCF)
        'NVDA'    # AI leader
    ]
    
    fetcher = YFinanceFetcher()
    results = []
    
    print("="*90)
    print(f"{'TICKER':<10} {'PRICE':<10} {'PRED RET%':<12} {'PROB':<10} {'SIGNAL':<15}")
    print("="*90)
    
    for ticker in test_tickers:
        try:
            # Fetch data
            stock_data = fetcher.fetch_stock_data(ticker)
            
            if stock_data is None or (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
                print(f"⚠️  {ticker}: No data available")
                continue
            
            # Get current price from yfinance directly
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                if current_price == 0:
                    # Fallback to recent history
                    hist = ticker_obj.history(period='5d')
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
            except:
                current_price = 0
            
            # Get enhanced technical/sentiment features
            from src.data.fetchers.technical_sentiment_fetcher import TechnicalSentimentFetcher
            tech_fetcher = TechnicalSentimentFetcher()
            enhanced_features = tech_fetcher.fetch_enhanced_features(ticker)
            
            # Prepare features for RF with enhanced features
            features_df = model.prepare_features(stock_data, lstm_predictions=None, enhanced_features=enhanced_features)
            
            # Get prediction
            prediction = model.predict_score(features_df)
            
            # Interpret results
            reg_score = prediction['regression_score']  # Predicted % return
            clf_prob = prediction['classification_prob']  # Probability 0-1
            
            # Generate signal based on predicted return and probability
            signal = "🚀 STRONG BUY" if reg_score > 20 and clf_prob > 0.7 else \
                    "✅ BUY" if reg_score > 10 and clf_prob > 0.6 else \
                    "⚖️  HOLD" if reg_score > -5 else \
                    "⚠️  SELL" if reg_score > -15 else \
                    "🔴 STRONG SELL"
            
            # Print results
            print(
                f"{ticker:<10} ${current_price:<9.2f} {reg_score:<11.2f}% "
                f"{clf_prob:<9.2f} {signal:<15}"
            )
            
            results.append({
                'ticker': ticker,
                'price': current_price,
                'predicted_return': reg_score,
                'outperform_prob': clf_prob,
                'signal': signal
            })
            
        except Exception as e:
            print(f"❌ {ticker}: {e}")
            continue
    
    print("="*90)
    
    # Summary statistics
    if results:
        df = pd.DataFrame(results)
        print(f"\n📊 SUMMARY:")
        print(f"   Tested: {len(results)} stocks")
        print(f"   Avg Predicted Return: {df['predicted_return'].mean():.2f}%")
        print(f"   Avg Outperform Prob: {df['outperform_prob'].mean():.2f}")
        print(f"   Buy signals: {len([s for s in df['signal'] if 'BUY' in s])} / {len(df)}")
        
        # Show top picks
        df_sorted = df.sort_values('predicted_return', ascending=False)
        print(f"\n🎯 TOP 3 PICKS BY PREDICTED RETURN:")
        for i, row in df_sorted.head(3).iterrows():
            print(f"   {row['ticker']}: {row['predicted_return']:.1f}% (Prob: {row['outperform_prob']:.2f})")


if __name__ == "__main__":
    test_rf_predictions()
