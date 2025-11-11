"""
Train Random Forest Ensemble Model for Stock Valuation
======================================================

Complements LSTM-DCF with fundamental valuation multiples approach.
Uses P/E, P/B, PEG, ROE, Beta, Debt/Equity and other metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.models.ensemble.rf_ensemble import RFEnsembleModel
from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.fetchers.technical_sentiment_fetcher import TechnicalSentimentFetcher
from config.settings import MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class RFEnsembleTrainer:
    """Train RF Ensemble on fundamental metrics"""
    
    def __init__(self):
        """Initialize trainer with enhanced features"""
        self.model = RFEnsembleModel(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4
        )
        self.fetcher = YFinanceFetcher()
        self.tech_sentiment_fetcher = TechnicalSentimentFetcher()
    
    def load_tickers_from_training_data(self) -> List[str]:
        """Load tickers from LSTM training data"""
        data_path = PROCESSED_DATA_DIR / "training" / "lstm_dcf_training_cleaned.csv"
        if not data_path.exists():
            logger.warning(f"Training data not found at {data_path}")
            return []
        
        df = pd.read_csv(data_path)
        tickers = df['ticker'].unique().tolist()
        logger.info(f"✅ Loaded {len(tickers)} tickers from LSTM training data")
        return tickers
    
    def get_expanded_ticker_list(self) -> List[str]:
        """
        Get comprehensive ticker list for larger training dataset
        Combines multiple sources to reach 300+ samples
        """
        # Start with LSTM training tickers (104)
        lstm_tickers = self.load_tickers_from_training_data()
        
        # S&P 500 major companies (top 100 by market cap)
        sp500_major = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'BRK-B', 'TSLA', 'META', 'UNH',
            'XOM', 'LLY', 'JNJ', 'JPM', 'V', 'PG', 'AVGO', 'HD', 'CVX', 'MA',
            'ABBV', 'PFE', 'BAC', 'KO', 'COST', 'MRK', 'TMO', 'WMT', 'CRM', 'ACN',
            'CSCO', 'ABT', 'ADBE', 'DHR', 'TXN', 'DIS', 'VZ', 'ORCL', 'NKE', 'COP',
            'NFLX', 'INTC', 'QCOM', 'CMCSA', 'PM', 'PEP', 'T', 'AMD', 'WFC', 'UNP',
            'NOW', 'IBM', 'LOW', 'CAT', 'RTX', 'HON', 'SPGI', 'NEE', 'GS', 'LMT',
            'UPS', 'MS', 'BLK', 'AMGN', 'ELV', 'SYK', 'BKNG', 'GILD', 'BA', 'MDT',
            'AXP', 'TJX', 'ADP', 'VRTX', 'SBUX', 'ISRG', 'CVS', 'TMUS', 'CI', 'ZTS',
            'LRCX', 'DE', 'MO', 'CB', 'SCHW', 'SO', 'FI', 'BDX', 'C', 'REGN',
            'MMM', 'BSX', 'EOG', 'DUK', 'ITW', 'PLD', 'NOC', 'APD', 'ICE', 'FCX'
        ]
        
        # NASDAQ 100 tech/growth stocks
        nasdaq100_tech = [
            'AVGO', 'BROADCOM', 'MCHP', 'KLAC', 'MRVL', 'SWKS', 'QRVO', 'NXPI', 'MPWR', 'WOLF',
            'TEAM', 'DXCM', 'ILMN', 'BIIB', 'GILEAD', 'ALGN', 'INCY', 'SIRI', 'BMRN', 'VRTX',
            'MDB', 'DOCU', 'CRWD', 'ZM', 'OKTA', 'DDOG', 'NET', 'FSLY', 'ESTC', 'SNOW'
        ]
        
        # Russell 2000 representative small-caps
        russell2k_sample = [
            'SAIA', 'ODFL', 'CHRW', 'JBHT', 'KNX', 'EXPD', 'LSTR', 'MATX', 'ARCB', 'WERN',
            'CVLT', 'NSIT', 'NEOG', 'PRGS', 'PLUS', 'CDNS', 'SNPS', 'ANSS', 'KEYS', 'TDY',
            'CPRT', 'ROL', 'VRSK', 'BR', 'FDS', 'IEX', 'ROP', 'WST', 'A', 'ALLE'
        ]
        
        # Financial sector diversity
        financials = [
            'SCHW', 'USB', 'TFC', 'PNC', 'COF', 'AIG', 'MET', 'PRU', 'ALL', 'TRV',
            'CB', 'PGR', 'AFL', 'HIG', 'CMA', 'ZION', 'FITB', 'HBAN', 'RF', 'KEY'
        ]
        
        # Healthcare and biotech
        healthcare = [
            'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN',
            'GILD', 'VRTX', 'SYK', 'BSX', 'MDT', 'CI', 'HUM', 'ELV', 'CVS', 'ANTM'
        ]
        
        # Energy and materials
        energy_materials = [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'HES', 'BKR',
            'FCX', 'NEM', 'AA', 'X', 'CLF', 'MT', 'STLD', 'NUE', 'CMC', 'RS'
        ]
        
        # Consumer and retail
        consumer = [
            'WMT', 'HD', 'COST', 'TGT', 'LOW', 'TJX', 'NKE', 'SBUX', 'MCD', 'KO',
            'PEP', 'PG', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'MDLZ', 'MNST', 'KHC'
        ]
        
        # Combine all lists
        all_tickers = (
            lstm_tickers + sp500_major + nasdaq100_tech + russell2k_sample + 
            financials + healthcare + energy_materials + consumer
        )
        
        # Remove duplicates and filter out problematic tickers
        unique_tickers = []
        seen = set()
        
        for ticker in all_tickers:
            if ticker not in seen and ticker not in ['BRK-A', 'BF-B', 'GOOG']:  # Avoid duplicate share classes
                unique_tickers.append(ticker)
                seen.add(ticker)
        
        logger.info(f"✅ Generated expanded ticker list: {len(unique_tickers)} total tickers")
        logger.info(f"   LSTM: {len(lstm_tickers)}, S&P500: {len(sp500_major)}")
        logger.info(f"   NASDAQ100: {len(nasdaq100_tech)}, Russell2K: {len(russell2k_sample)}")
        logger.info(f"   Sectors: Financials({len(financials)}), Health({len(healthcare)}), Energy({len(energy_materials)}), Consumer({len(consumer)})")
        
        return unique_tickers
    
    def fetch_current_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch current fundamental metrics for ticker
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with fundamental metrics
        """
        try:
            stock_data = self.fetcher.fetch_stock_data(ticker)
            
            if stock_data is None or (isinstance(stock_data, pd.DataFrame) and stock_data.empty):
                return None
            
            # Helper to extract scalar values from DataFrame/dict/Series
            def extract_value(data, key, default=np.nan):
                val = data.get(key, default)
                if isinstance(val, pd.Series):
                    return val.iloc[0] if not val.empty else default
                return val
            
            # Extract key metrics
            metrics = {
                'ticker': ticker,
                'pe_ratio': extract_value(stock_data, 'pe_ratio'),
                'price_to_book': extract_value(stock_data, 'price_to_book'),
                'peg_ratio': extract_value(stock_data, 'peg_ratio'),
                'enterprise_to_ebitda': extract_value(stock_data, 'enterprise_to_ebitda'),
                'beta': extract_value(stock_data, 'beta'),
                'debt_to_equity': extract_value(stock_data, 'debt_to_equity'),
                'current_ratio': extract_value(stock_data, 'current_ratio'),
                'return_on_equity': extract_value(stock_data, 'return_on_equity'),
                'revenue_growth': extract_value(stock_data, 'revenue_growth'),
                'free_cash_flow': extract_value(stock_data, 'free_cash_flow'),
                'revenue': extract_value(stock_data, 'revenue'),
                'market_cap': extract_value(stock_data, 'market_cap'),
                'current_price': extract_value(stock_data, 'current_price')
            }
            
            return metrics
            
        except Exception as e:
            logger.debug(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_enhanced_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch enhanced 14-feature metrics for ticker (Risk + Sentiment Pipeline)
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dict with 14 enhanced features
        """
        try:
            # Get enhanced technical and sentiment features
            enhanced_features = self.tech_sentiment_fetcher.fetch_enhanced_features(ticker)
            
            if enhanced_features is None:
                logger.debug(f"No enhanced features for {ticker}")
                return None
            
            # Add ticker info
            enhanced_features['ticker'] = ticker
            
            return enhanced_features
            
        except Exception as e:
            logger.debug(f"Error fetching enhanced features for {ticker}: {e}")
            return None
    
    def calculate_forward_returns(self, ticker: str, months_forward: int = 12) -> float:
        """
        Calculate actual forward returns for training target
        
        Args:
            ticker: Stock ticker
            months_forward: How many months forward to measure
            
        Returns:
            Forward return percentage or None
        """
        try:
            # Get historical prices
            stock = yf.Ticker(ticker)
            
            # Get price from 12 months ago to today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months_forward * 30 + 30)  # Extra buffer
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty or len(hist) < 2:
                return None
            
            # Calculate return from 12 months ago to now
            # (This simulates what we would have predicted 12 months ago)
            price_12m_ago = hist['Close'].iloc[0]
            price_now = hist['Close'].iloc[-1]
            
            forward_return = ((price_now - price_12m_ago) / price_12m_ago) * 100
            
            return forward_return
            
        except Exception as e:
            logger.debug(f"Error calculating returns for {ticker}: {e}")
            return None
    
    def create_training_dataset(self, tickers: List[str], max_tickers: int = None) -> pd.DataFrame:
        """
        Create training dataset with features and targets
        
        Args:
            tickers: List of stock tickers
            max_tickers: Maximum tickers to process (None = all)
            
        Returns:
            DataFrame with features and targets
        """
        if max_tickers:
            tickers = tickers[:max_tickers]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 CREATING RF TRAINING DATASET")
        logger.info(f"{'='*80}\n")
        logger.info(f"Processing {len(tickers)} tickers...")
        
        training_data = []
        failed_tickers = []
        
        # Progress tracking
        total_tickers = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            # Progress reporting
            if i % 25 == 0 or i == total_tickers:
                logger.info(f"   Progress: {i}/{total_tickers} ({i/total_tickers*100:.1f}%) - {len(training_data)} valid samples so far")
            
            try:
                # Get enhanced features (14-feature Risk + Sentiment Pipeline)
                metrics = self.fetch_enhanced_fundamentals(ticker)
                if metrics is None:
                    failed_tickers.append((ticker, "No enhanced features"))
                    continue
                
                # Get forward returns (target)
                forward_return = self.calculate_forward_returns(ticker, months_forward=12)
                if forward_return is None:
                    failed_tickers.append((ticker, "No price history"))
                    continue
            
            except Exception as e:
                failed_tickers.append((ticker, f"Error: {str(e)}"))
                continue
            
            # Add target
            metrics['forward_return_12m'] = forward_return
            metrics['is_outperformer'] = 1 if forward_return > 0 else 0  # Binary classification
            
            training_data.append(metrics)
        
        df = pd.DataFrame(training_data)
        
        logger.info(f"\n✅ Dataset Creation Summary:")
        logger.info(f"   📊 Total samples: {len(df)}")
        logger.info(f"   ✅ Success rate: {len(df)}/{total_tickers} ({len(df)/total_tickers*100:.1f}%)")
        logger.info(f"   ❌ Failed tickers: {len(failed_tickers)}")
        logger.info(f"   📈 Positive returns: {df['is_outperformer'].sum()}/{len(df)} ({df['is_outperformer'].mean()*100:.1f}%)")
        logger.info(f"   💰 Mean return: {df['forward_return_12m'].mean():.2f}% (±{df['forward_return_12m'].std():.2f}%)")
        
        # Show a few failed examples
        if failed_tickers:
            logger.info(f"\n⚠️  Sample failed tickers:")
            for ticker, reason in failed_tickers[:5]:
                logger.info(f"     {ticker}: {reason}")
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> Dict:
        """
        Train RF model on dataset
        
        Args:
            df: Training dataframe
            
        Returns:
            Training metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🌲 TRAINING RANDOM FOREST ENSEMBLE")
        logger.info(f"{'='*80}\n")
        
        # Prepare ENHANCED 14-feature set (Risk + Sentiment Pipeline)
        feature_df = df.copy()
        
        # Define 14-feature set in exact order
        X_cols = [
            # 1-3: Core Risk Metrics
            'beta', 'debt_to_equity', '30d_volatility',
            # 4-5: Volume and Short Interest
            'volume_zscore', 'short_percent',
            # 6: Technical Indicator
            'rsi_14',
            # 7-10: News Sentiment Features
            'sentiment_mean', 'sentiment_std', 'news_volume', 'relevance_mean',
            # 11-14: Fundamental Features
            'pe_ratio', 'revenue_growth', 'current_ratio', 'return_on_equity'
        ]
        
        # Filter to available features
        X_cols = [col for col in X_cols if col in feature_df.columns]
        X = feature_df[X_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())  # Fill with median
        X = X.replace([np.inf, -np.inf], 0)  # Handle infinities
        
        # Targets
        y_regression = df['forward_return_12m'].values
        y_classification = df['is_outperformer'].values
        
        logger.info(f"Training with {len(X_cols)} features:")
        for col in X_cols:
            logger.info(f"  • {col}")
        
        # Train
        metrics = self.model.train(X, y_regression, y_classification)
        
        logger.info(f"\n📊 Training Results:")
        logger.info(f"   R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"   CV Mean: {metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})")
        if 'classification_accuracy' in metrics:
            logger.info(f"   Classification Accuracy: {metrics['classification_accuracy']*100:.2f}%")
        
        # Feature importance
        importance = self.model.get_feature_importance()
        logger.info(f"\n🎯 Feature Importance:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"   {row['feature']:30s} {row['importance']*100:6.2f}%")
        
        # Save feature importance
        importance_path = MODELS_DIR / "rf_feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        logger.info(f"\n✅ Feature importance saved: {importance_path}")
        
        return metrics
    
    def save_model(self):
        """Save trained model"""
        model_path = MODELS_DIR / "rf_ensemble.pkl"
        self.model.save(str(model_path))
        logger.info(f"✅ Model saved: {model_path}")
    
    def save_training_data(self, df: pd.DataFrame):
        """Save training dataset for reference"""
        output_path = PROCESSED_DATA_DIR / "rf_training_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Training data saved: {output_path}")


def main():
    """Main training execution"""
    logger.info("\n" + "="*80)
    logger.info("🚀 RF ENSEMBLE TRAINING PIPELINE - EXPANDED DATASET")
    logger.info("="*80 + "\n")
    
    trainer = RFEnsembleTrainer()
    
    # Get expanded ticker list for larger training dataset
    tickers = trainer.get_expanded_ticker_list()
    
    if not tickers:
        logger.error("❌ No tickers found.")
        return
    
    # Create training dataset with 300+ samples target
    logger.info(f"🎯 Target: 300+ training samples from {len(tickers)} tickers")
    df = trainer.create_training_dataset(tickers, max_tickers=350)  # Fetch 350 to ensure 300+ valid samples
    
    if df.empty:
        logger.error("❌ No training data collected.")
        return
    
    # Save dataset
    trainer.save_training_data(df)
    
    # Train model
    metrics = trainer.train_model(df)
    
    # Save model
    trainer.save_model()
    
    logger.info("\n" + "="*80)
    logger.info("✅ RF ENSEMBLE TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nModel Performance:")
    logger.info(f"  • R² Score: {metrics['r2_score']:.4f}")
    logger.info(f"  • CV Score: {metrics['cv_mean']:.4f}")
    if 'classification_accuracy' in metrics:
        logger.info(f"  • Classification: {metrics['classification_accuracy']*100:.1f}%")
    
    logger.info(f"\nFiles Created:")
    logger.info(f"  • Model: models/rf_ensemble.pkl")
    logger.info(f"  • Feature Importance: models/rf_feature_importance.csv")
    logger.info(f"  • Training Data: data/processed/rf_training_data.csv")
    
    logger.info(f"\nNext Steps:")
    logger.info(f"  1. Test RF model: python scripts/test_rf_ensemble.py")
    logger.info(f"  2. Build consensus scorer with LSTM + RF")
    logger.info(f"  3. Integrate into analyze_stock.py")


if __name__ == "__main__":
    main()
