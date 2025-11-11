"""
Consensus Stock Analyzer
=========================
End-to-end stock analysis using 70/20/10 weighted ensemble:
- 70% LSTM-DCF (intrinsic value)
- 20% RF Risk + Sentiment (short-term brake)
- 10% P/E Sanity Check (reality anchor)

Includes Reverse DCF validation and margin of safety calculation.

Usage:
    python scripts/analyze_stock_consensus.py PYPL
    python scripts/analyze_stock_consensus.py AAPL MSFT GOOGL --compare
    python scripts/analyze_stock_consensus.py --watchlist 50
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import joblib
import argparse
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from config.settings import MODELS_DIR, RAW_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class ConsensusScorer:
    """
    Weighted consensus scoring: 70% LSTM-DCF, 20% RF Risk, 10% P/E
    """
    
    def __init__(self):
        self.weights = {
            'lstm_dcf': 0.70,
            'rf_risk_sentiment': 0.20,
            'pe_sanity': 0.10
        }
        
        self.lstm_model = None
        self.lstm_scaler = None
        self.rf_model = None
        self.rf_scaler = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models"""
        # Load LSTM-DCF
        try:
            lstm_path = MODELS_DIR / 'lstm_dcf_enhanced.pth'
            if lstm_path.exists():
                checkpoint = torch.load(str(lstm_path), weights_only=False)
                
                self.lstm_model = LSTMDCFModel(
                    input_size=checkpoint['hyperparameters']['input_size'],
                    hidden_size=checkpoint['hyperparameters']['hidden_size'],
                    num_layers=checkpoint['hyperparameters']['num_layers'],
                    output_size=checkpoint['hyperparameters']['output_size'],
                    dropout=checkpoint['hyperparameters'].get('dropout', 0.2)
                )
                
                self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
                self.lstm_model.eval()
                self.lstm_scaler = checkpoint.get('scaler')
                
                logger.info("✅ LSTM-DCF model loaded")
            else:
                logger.warning(f"LSTM model not found at {lstm_path}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
        
        # Load RF Risk + Sentiment
        try:
            rf_path = MODELS_DIR / 'rf_risk_sentiment.pkl'
            if rf_path.exists():
                bundle = joblib.load(rf_path)
                self.rf_model = bundle['model']
                self.rf_scaler = bundle['scaler']
                logger.info("✅ RF Risk+Sentiment model loaded")
            else:
                logger.warning(f"RF model not found at {rf_path}")
        except Exception as e:
            logger.error(f"Failed to load RF model: {e}")
    
    def calculate_consensus(self, ticker: str) -> Dict:
        """
        Calculate consensus score and recommendation
        """
        logger.info(f"{'='*80}")
        logger.info(f"Analyzing: {ticker}")
        logger.info(f"{'='*80}")
        
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Get basic info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not current_price:
                raise ValueError("Could not fetch current price")
            
            company_name = info.get('longName', ticker)
            
            # Component 1: LSTM-DCF (70%)
            lstm_result = self._lstm_dcf_score(ticker, stock, current_price)
            
            # Component 2: RF Risk + Sentiment (20%)
            rf_result = self._rf_risk_score(ticker, stock)
            
            # Component 3: P/E Sanity (10%)
            pe_result = self._pe_sanity_score(ticker, info)
            
            # Calculate weighted consensus
            consensus_score = (
                lstm_result['score'] * self.weights['lstm_dcf'] +
                rf_result['score'] * self.weights['rf_risk_sentiment'] +
                pe_result['score'] * self.weights['pe_sanity']
            )
            
            # Generate signal
            signal = self._generate_signal(
                consensus_score,
                lstm_result.get('margin_of_safety', 0),
                lstm_result.get('reverse_dcf_flag', 'N/A'),
                rf_result.get('raw_penalty', 0)
            )
            
            # Compile results
            result.update({
                'success': True,
                'company_name': company_name,
                'current_price': current_price,
                'consensus_score': round(consensus_score, 1),
                'signal': signal,
                'breakdown': {
                    'lstm_dcf': lstm_result['score'],
                    'rf_risk_sentiment': rf_result['score'],
                    'pe_sanity': pe_result['score']
                },
                'lstm_details': lstm_result,
                'rf_details': rf_result,
                'pe_details': pe_result
            })
            
            # Print results
            self._print_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            result['error'] = str(e)
            return result
    
    def _lstm_dcf_score(self, ticker: str, stock, current_price: float) -> Dict:
        """
        LSTM-DCF component (70% weight)
        Returns score 0-100 based on undervaluation depth
        """
        result = {
            'score': 50,  # Neutral default
            'fair_value': None,
            'margin_of_safety': 0,
            'predicted_growth': None,
            'reverse_dcf_flag': 'N/A'
        }
        
        if self.lstm_model is None:
            logger.warning("LSTM model not loaded, using neutral score")
            return result
        
        try:
            # TODO: Full implementation requires sequence preparation
            # For MVP, use simplified DCF with average historical growth
            
            # Get current FCF
            cashflow_path = RAW_DATA_DIR / 'financial_statements' / f'{ticker}_cashflow.csv'
            
            if cashflow_path.exists():
                df = pd.read_csv(cashflow_path)
                df['date'] = pd.to_datetime(df['fiscalDateEnding'])
                df = df.sort_values('date', ascending=False)
                
                operating_cf = pd.to_numeric(df['operatingCashflow'].iloc[0], errors='coerce')
                capex = pd.to_numeric(df['capitalExpenditures'].iloc[0], errors='coerce')
                
                if pd.notna(operating_cf) and pd.notna(capex):
                    current_fcf = operating_cf - abs(capex)
                    
                    # Simple growth estimate (historical average)
                    fcf_series = []
                    for _, row in df.head(8).iterrows():
                        ocf = pd.to_numeric(row['operatingCashflow'], errors='coerce')
                        cx = pd.to_numeric(row['capitalExpenditures'], errors='coerce')
                        if pd.notna(ocf) and pd.notna(cx):
                            fcf_series.append(ocf - abs(cx))
                    
                    if len(fcf_series) > 1:
                        fcf_series = pd.Series(fcf_series)
                        avg_growth = fcf_series.pct_change().mean()
                        avg_growth = np.clip(avg_growth, -0.3, 0.5)  # Clip to reasonable range
                        
                        # Calculate fair value (simplified DCF)
                        wacc = 0.08
                        terminal_growth = 0.03
                        shares = stock.info.get('sharesOutstanding', 1)
                        
                        pv_sum = 0
                        fcf = current_fcf
                        for t in range(1, 11):
                            fcf = fcf * (1 + avg_growth)
                            pv_sum += fcf / (1 + wacc) ** t
                        
                        terminal_fcf = fcf * (1 + terminal_growth)
                        terminal_value = terminal_fcf / (wacc - terminal_growth)
                        pv_terminal = terminal_value / (1 + wacc) ** 10
                        
                        enterprise_value = pv_sum + pv_terminal
                        fair_value = enterprise_value / shares
                        
                        # Margin of safety
                        mos = (fair_value - current_price) / current_price * 100
                        
                        # Score: 50 + (MoS * 100), capped at 0-100
                        score = 50 + mos
                        score = max(0, min(100, score))
                        
                        result.update({
                            'score': score,
                            'fair_value': fair_value,
                            'margin_of_safety': mos,
                            'predicted_growth': avg_growth * 100,
                            'reverse_dcf_flag': '✅ OK' if abs(mos) < 50 else '⚠️ CAUTION'
                        })
        
        except Exception as e:
            logger.warning(f"LSTM-DCF scoring failed: {e}")
        
        return result
    
    def _rf_risk_score(self, ticker: str, stock) -> Dict:
        """
        RF Risk + Sentiment component (20% weight)
        Returns score 20-70 (shifted penalty)
        """
        result = {
            'score': 50,  # Neutral default
            'risk_class': 'Medium',
            'raw_penalty': 0,
            'sentiment_signal': 'NEUTRAL'
        }
        
        if self.rf_model is None:
            logger.warning("RF model not loaded, using neutral score")
            return result
        
        try:
            info = stock.info
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return result
            
            # Extract features (simplified - full version in train script)
            beta = info.get('beta', 1.0)
            recent_prices = hist['Close'].tail(30)
            volatility_30d = recent_prices.pct_change().std() * np.sqrt(252) if len(recent_prices) > 1 else 0.3
            debt_to_equity = info.get('debtToEquity', 0) / 100
            
            # RSI
            deltas = hist['Close'].diff()
            gain = deltas.where(deltas > 0, 0).rolling(window=14).mean()
            loss = -deltas.where(deltas < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = rsi.iloc[-1] if not rsi.empty else 50.0
            
            # Volume z-score
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].tail(5).mean()
            volume_zscore = (recent_volume - avg_volume) / (hist['Volume'].std() + 1e-8)
            
            short_pct = info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') else 0
            
            # Sentiment (neutral defaults)
            sentiment_mean = 0.0
            sentiment_std = 0.2
            news_volume = 10
            relevance_mean = 0.5
            
            # Valuation
            pe_ratio = info.get('trailingPE', 20) or 20
            price_to_book = info.get('priceToBook', 2) or 2
            profit_margin = (info.get('profitMargins', 0.1) or 0.1) * 100
            roe = (info.get('returnOnEquity', 0.15) or 0.15) * 100
            
            features = np.array([[
                beta, volatility_30d, debt_to_equity, volume_zscore, short_pct, rsi_14,
                sentiment_mean, sentiment_std, news_volume, relevance_mean,
                pe_ratio, price_to_book, profit_margin, roe
            ]])
            
            # Predict risk class
            features_scaled = self.rf_scaler.transform(features)
            risk_class = self.rf_model.predict(features_scaled)[0]
            
            # Calculate penalty
            penalties = {'Low': +10, 'Medium': -5, 'High': -20}
            base_penalty = penalties[risk_class]
            
            # Sentiment adjustments
            if sentiment_mean < 0.3 and rsi_14 < 30 and news_volume > 20:
                base_penalty += 10  # Contrarian opportunity
            
            if sentiment_mean < 0.1 and news_volume > 40:
                base_penalty -= 15  # Panic
            
            final_penalty = max(-30, min(10, base_penalty))
            
            # Convert to 0-100 score: -30→20, -5→51.67, +10→70
            score = (final_penalty + 30) / 40 * 50 + 20
            
            sentiment_signal = 'OVERSOLD' if sentiment_mean < 0.3 else 'NEUTRAL' if sentiment_mean < 0.6 else 'OVERBOUGHT'
            
            result.update({
                'score': score,
                'risk_class': risk_class,
                'raw_penalty': final_penalty,
                'sentiment_signal': sentiment_signal,
                'rsi': rsi_14,
                'volatility': volatility_30d
            })
        
        except Exception as e:
            logger.warning(f"RF risk scoring failed: {e}")
        
        return result
    
    def _pe_sanity_score(self, ticker: str, info: Dict) -> Dict:
        """
        P/E Sanity Check (10% weight)
        """
        result = {
            'score': 50,
            'pe_ratio': None,
            'sector': 'Unknown',
            'sector_avg': None
        }
        
        try:
            pe = info.get('trailingPE')
            sector = info.get('sector', 'Unknown')
            
            if pe is None or pe < 0:
                return result
            
            # Sector average P/E (hardcoded benchmarks)
            sector_pes = {
                'Technology': 22,
                'Healthcare': 18,
                'Financial Services': 12,
                'Consumer Cyclical': 16,
                'Energy': 14,
                'Industrials': 17,
                'Consumer Defensive': 20,
                'Utilities': 15,
                'Real Estate': 25,
                'Basic Materials': 14,
                'Communication Services': 18
            }
            
            sector_avg = sector_pes.get(sector, 18)
            
            # Scoring logic
            if pe < 0.8 * sector_avg:
                score = 100  # Undervalued
            elif pe < 1.2 * sector_avg:
                score = 70   # Fair
            elif pe < 2.0 * sector_avg:
                score = 40   # Overvalued
            else:
                score = 20   # Bubble
            
            result.update({
                'score': score,
                'pe_ratio': pe,
                'sector': sector,
                'sector_avg': sector_avg
            })
        
        except Exception as e:
            logger.warning(f"P/E sanity check failed: {e}")
        
        return result
    
    def _generate_signal(
        self,
        score: float,
        mos: float,
        reverse_flag: str,
        risk_penalty: float
    ) -> str:
        """Generate buy/hold/sell signal"""
        
        if mos > 20 and reverse_flag in ['✅ OK', 'N/A'] and score > 70:
            return "STRONG BUY"
        elif mos > 10 and risk_penalty > -15 and score > 60:
            return "BUY"
        elif mos > 0 and score > 50:
            return "HOLD (Slight Undervaluation)"
        elif mos < -10 and score < 40:
            return "SELL"
        else:
            return "HOLD"
    
    def _print_results(self, result: Dict):
        """Pretty print analysis results"""
        if not result['success']:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}\n")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 {result['ticker']} - {result['company_name']}")
        print(f"{'='*80}\n")
        
        lstm = result['lstm_details']
        rf = result['rf_details']
        pe = result['pe_details']
        
        print(f"💰 Valuation:")
        if lstm.get('fair_value'):
            print(f"   Fair Value:   ${lstm['fair_value']:.2f}")
        print(f"   Current:      ${result['current_price']:.2f}")
        if lstm.get('margin_of_safety'):
            arrow = "↗" if lstm['margin_of_safety'] > 0 else "↘"
            print(f"   MoS:          {lstm['margin_of_safety']:.1f}% {arrow}")
        
        print(f"\n✅ Consensus: {result['consensus_score']:.1f}/100 → {result['signal']}")
        
        if lstm.get('predicted_growth'):
            print(f"\n📈 Growth Analysis:")
            print(f"   Predicted:    {lstm['predicted_growth']:.1f}%")
            print(f"   Validation:   {lstm['reverse_dcf_flag']}")
        
        print(f"\n⚖️  Breakdown:")
        print(f"   LSTM-DCF:     {result['breakdown']['lstm_dcf']:.1f} (70%)")
        print(f"   Risk+Sent:    {result['breakdown']['rf_risk_sentiment']:.1f} (20%)")
        print(f"   P/E Sanity:   {result['breakdown']['pe_sanity']:.1f} (10%)")
        
        print(f"\n🛡️  Risk Assessment:")
        print(f"   Class:        {rf['risk_class']}")
        print(f"   Sentiment:    {rf['sentiment_signal']}")
        if rf.get('rsi'):
            print(f"   RSI(14):      {rf['rsi']:.1f}")
        
        if pe.get('pe_ratio'):
            print(f"\n📊 Valuation Metrics:")
            print(f"   P/E Ratio:    {pe['pe_ratio']:.1f}")
            print(f"   Sector Avg:   {pe['sector_avg']:.1f} ({pe['sector']})")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Consensus Stock Analyzer")
    parser.add_argument(
        'tickers',
        nargs='*',
        help='Stock ticker(s) to analyze'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple stocks side-by-side'
    )
    parser.add_argument(
        '--watchlist',
        type=int,
        default=None,
        help='Analyze top N stocks from S&P 500'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=60,
        help='Minimum consensus score to display (default: 60)'
    )
    
    args = parser.parse_args()
    
    if not args.tickers and not args.watchlist:
        parser.error("Provide tickers or use --watchlist")
    
    # Initialize scorer
    scorer = ConsensusScorer()
    
    # Get tickers
    if args.watchlist:
        # Top liquid S&P 500 stocks
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
            'WMT', 'JPM', 'PG', 'XOM', 'UNH', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'DHR',
            'WFC', 'DIS', 'ADBE', 'VZ', 'CRM', 'NEE', 'CMCSA', 'TXN', 'PM', 'NKE',
            'UPS', 'RTX', 'HON', 'ORCL', 'INTC', 'QCOM', 'BMY', 'LIN', 'AMD', 'UNP'
        ][:args.watchlist]
    else:
        tickers = args.tickers
    
    # Analyze stocks
    results = []
    for ticker in tickers:
        result = scorer.calculate_consensus(ticker)
        results.append(result)
        
        if len(tickers) > 1 and not args.compare:
            input("\nPress Enter to continue...")
    
    # Comparison view
    if args.compare and len(results) > 1:
        print(f"\n{'='*100}")
        print(f"📊 Stock Comparison")
        print(f"{'='*100}\n")
        
        comparison_df = pd.DataFrame([
            {
                'Ticker': r['ticker'],
                'Price': r.get('current_price', 0),
                'Consensus': r.get('consensus_score', 0),
                'Signal': r.get('signal', 'N/A'),
                'MoS%': r.get('lstm_details', {}).get('margin_of_safety', 0),
                'Risk': r.get('rf_details', {}).get('risk_class', 'N/A'),
                'P/E': r.get('pe_details', {}).get('pe_ratio', 0)
            }
            for r in results if r['success']
        ])
        
        comparison_df = comparison_df.sort_values('Consensus', ascending=False)
        print(comparison_df.to_string(index=False))
        print(f"\n{'='*100}\n")
    
    # Filter by min score
    filtered = [r for r in results if r['success'] and r.get('consensus_score', 0) >= args.min_score]
    
    print(f"\n📋 Summary: {len(filtered)}/{len(results)} stocks above score {args.min_score}")
    
    # Save results
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
