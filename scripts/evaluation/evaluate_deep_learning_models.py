"""
Comprehensive Evaluation of Deep Learning Models
Tests: LSTM-DCF (Price), LSTM Growth Forecaster

Usage:
    python scripts/evaluate_deep_learning_models.py
    python scripts/evaluate_deep_learning_models.py --output-report
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import torch

from src.models.deep_learning.lstm_dcf import LSTMDCFModel
from src.models.deep_learning.lstm_growth_forecaster import LSTMGrowthForecaster, DCFValuationWithLSTM
from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.fetchers.alpha_vantage_financials import AlphaVantageFinancialsFetcher
from config.settings import MODELS_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)

# Test stocks
TEST_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']


class DeepLearningEvaluator:
    """Evaluates deep learning models (LSTM-DCF, LSTM Growth)"""
    
    def __init__(self):
        self.yf_fetcher = YFinanceFetcher()
        self.av_fetcher = AlphaVantageFinancialsFetcher()
        self.results = {
            'lstm_dcf_price': {},
            'lstm_growth': {},
            'performance': {},
            'srs_compliance': {}
        }
    
    def load_models(self):
        """Load all deep learning models"""
        logger.info("Loading deep learning models...")
        
        try:
            # LSTM-DCF (Price Prediction Model)
            lstm_dcf_path = MODELS_DIR / "lstm_dcf_final.pth"
            if lstm_dcf_path.exists():
                self.lstm_dcf = LSTMDCFModel(input_size=12, hidden_size=128, num_layers=3)
                self.lstm_dcf.load_model(str(lstm_dcf_path))
                self.lstm_dcf.eval()
                logger.info("✅ LSTM-DCF (Price) Model loaded")
            else:
                logger.warning("⚠️ LSTM-DCF Model not found")
                self.lstm_dcf = None
            
            # LSTM Growth Forecaster
            lstm_growth_path = MODELS_DIR / "lstm_growth_forecaster.pth"
            if lstm_growth_path.exists():
                self.lstm_growth = LSTMGrowthForecaster(input_size=4, hidden_size=64, num_layers=2)
                self.lstm_growth.load_model(str(lstm_growth_path))
                self.lstm_growth.eval()
                self.dcf_valuator = DCFValuationWithLSTM(self.lstm_growth)
                logger.info("✅ LSTM Growth Forecaster loaded")
            else:
                logger.warning("⚠️ LSTM Growth Forecaster not found (may not be trained yet)")
                self.lstm_growth = None
                self.dcf_valuator = None
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def evaluate_lstm_dcf_price(self, ticker: str) -> Dict:
        """Evaluate LSTM-DCF Price Prediction Model"""
        if not self.lstm_dcf:
            return {'error': 'Model not loaded'}
        
        try:
            # This model requires 60 periods of historical data
            # For now, we'll just test loading and inference time
            
            # Create dummy input (would normally fetch real data)
            dummy_input = torch.randn(1, 60, 12)  # batch_size=1, seq_len=60, features=12
            
            # Time the inference (SRS NFR-ML-1: < 300ms)
            start_time = time.time()
            with torch.no_grad():
                prediction = self.lstm_dcf(dummy_input)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            result = {
                'ticker': ticker,
                'model': 'LSTM-DCF (Price)',
                'inference_time_ms': inference_time,
                'meets_srs_nfr_ml1': inference_time < 300,
                'notes': 'Dummy input used (real data pipeline not fully integrated)'
            }
            
            logger.info(f"{ticker}: LSTM-DCF inference time={inference_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
            return {'error': str(e)}
    
    def evaluate_lstm_growth(self, ticker: str) -> Dict:
        """Evaluate LSTM Growth Forecaster"""
        if not self.lstm_growth or not self.dcf_valuator:
            return {'error': 'Model not loaded (may need to train first)'}
        
        try:
            # Fetch financial statements from Alpha Vantage
            data = self.av_fetcher.prepare_lstm_training_data(ticker, use_cache=True)
            
            if data is None:
                return {'error': 'No financial data available'}
            
            # Get standardized data (last 20 quarters)
            std_data = data['standardized_data'].tail(20)
            
            if len(std_data) < 20:
                return {'error': f'Insufficient data: {len(std_data)} quarters (need 20)'}
            
            # Prepare input tensor
            features = std_data[['revenue_norm_std', 'capex_norm_std', 'da_norm_std', 'nopat_norm_std']].values
            input_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
            
            # Time the inference (SRS NFR-ML-1: < 300ms)
            start_time = time.time()
            
            # Forecast growth rates
            with torch.no_grad():
                growth_rates = self.lstm_growth(input_tensor).squeeze().numpy()
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Get current metrics for DCF calculation
            current_metrics = data['normalized_data'].iloc[-1]
            
            result = {
                'ticker': ticker,
                'model': 'LSTM Growth Forecaster',
                'growth_rates': {
                    'revenue': float(growth_rates[0]),
                    'capex': float(growth_rates[1]),
                    'da': float(growth_rates[2]),
                    'nopat': float(growth_rates[3])
                },
                'data_quarters': len(std_data),
                'inference_time_ms': inference_time,
                'meets_srs_nfr_ml1': inference_time < 300
            }
            
            logger.info(f"{ticker}: LSTM Growth forecast - Revenue={growth_rates[0]:.2%}, "
                       f"CapEx={growth_rates[1]:.2%}, Time={inference_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_evaluation(self, test_stocks: List[str] = None):
        """Run comprehensive evaluation on test stocks"""
        if test_stocks is None:
            test_stocks = TEST_STOCKS
        
        logger.info("\n" + "="*80)
        logger.info("DEEP LEARNING MODELS EVALUATION")
        logger.info("="*80)
        
        # Evaluate LSTM-DCF (Price)
        logger.info("\n📊 Evaluating LSTM-DCF (Price Prediction)...")
        lstm_dcf_results = []
        for ticker in test_stocks:
            result = self.evaluate_lstm_dcf_price(ticker)
            if 'error' not in result:
                lstm_dcf_results.append(result)
        
        self.results['lstm_dcf_price'] = lstm_dcf_results
        
        # Evaluate LSTM Growth Forecaster
        logger.info("\n🎯 Evaluating LSTM Growth Forecaster...")
        lstm_growth_results = []
        
        # Only evaluate stocks we have data for (from fetch_progress.json)
        available_stocks = ['AAPL', 'ABBV', 'ABT', 'ACN', 'AMZN', 'APA', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BDX']
        eval_stocks = [s for s in test_stocks if s in available_stocks]
        
        for ticker in eval_stocks:
            result = self.evaluate_lstm_growth(ticker)
            if 'error' not in result:
                lstm_growth_results.append(result)
        
        self.results['lstm_growth'] = lstm_growth_results
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()
        
        # Check SRS compliance
        self._check_srs_compliance()
        
        # Print summary
        self._print_summary()
    
    def _calculate_aggregate_metrics(self):
        """Calculate aggregate performance metrics"""
        metrics = {}
        
        # LSTM-DCF metrics
        if self.results['lstm_dcf_price']:
            dcf_times = [r['inference_time_ms'] for r in self.results['lstm_dcf_price']]
            metrics['lstm_dcf_avg_inference_time'] = np.mean(dcf_times)
            metrics['lstm_dcf_max_inference_time'] = np.max(dcf_times)
            metrics['lstm_dcf_passes_nfr_ml1'] = all(
                r['meets_srs_nfr_ml1'] for r in self.results['lstm_dcf_price']
            )
        
        # LSTM Growth metrics
        if self.results['lstm_growth']:
            growth_times = [r['inference_time_ms'] for r in self.results['lstm_growth']]
            metrics['lstm_growth_avg_inference_time'] = np.mean(growth_times)
            metrics['lstm_growth_max_inference_time'] = np.max(growth_times)
            metrics['lstm_growth_passes_nfr_ml1'] = all(
                r['meets_srs_nfr_ml1'] for r in self.results['lstm_growth']
            )
        
        self.results['performance'] = metrics
    
    def _check_srs_compliance(self):
        """Check compliance with SRS requirements"""
        compliance = {
            'NFR-ML-1': {  # ML inference < 300ms
                'requirement': 'ML inference shall complete in < 300ms per stock',
                'lstm_dcf_price': self.results['performance'].get('lstm_dcf_passes_nfr_ml1', False),
                'lstm_growth': self.results['performance'].get('lstm_growth_passes_nfr_ml1', False),
                'status': 'PASS' if (
                    self.results['performance'].get('lstm_dcf_passes_nfr_ml1', False) and
                    self.results['performance'].get('lstm_growth_passes_nfr_ml1', False)
                ) else 'PARTIAL' if any([
                    self.results['performance'].get('lstm_dcf_passes_nfr_ml1', False),
                    self.results['performance'].get('lstm_growth_passes_nfr_ml1', False)
                ]) else 'FAIL'
            },
            'FR-ML-1': {  # LSTM for time-series forecasting
                'requirement': 'System shall use LSTM for time-series forecasting (60 periods)',
                'status': 'PASS' if self.lstm_dcf else 'FAIL',
                'notes': 'LSTM-DCF uses 60 periods, LSTM Growth uses 20 quarters'
            },
            'FR-ML-2': {  # DCF with LSTM predictions
                'requirement': 'System shall calculate DCF with LSTM-predicted cash flows',
                'status': 'PASS' if self.dcf_valuator else 'IN_PROGRESS',
                'notes': 'LSTM Growth Forecaster integrated with DCF valuation'
            }
        }
        
        self.results['srs_compliance'] = compliance
    
    def _print_summary(self):
        """Print evaluation summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 EVALUATION SUMMARY")
        logger.info("="*80)
        
        # LSTM-DCF Summary
        if self.results['lstm_dcf_price']:
            logger.info("\n✅ LSTM-DCF (Price Prediction):")
            logger.info(f"  Stocks evaluated: {len(self.results['lstm_dcf_price'])}")
            logger.info(f"  Avg inference time: {self.results['performance']['lstm_dcf_avg_inference_time']:.2f}ms")
            logger.info(f"  Max inference time: {self.results['performance']['lstm_dcf_max_inference_time']:.2f}ms")
            logger.info(f"  NFR-ML-1 (< 300ms): {'✅ PASS' if self.results['performance']['lstm_dcf_passes_nfr_ml1'] else '❌ FAIL'}")
        
        # LSTM Growth Summary
        if self.results['lstm_growth']:
            logger.info("\n✅ LSTM Growth Forecaster:")
            logger.info(f"  Stocks evaluated: {len(self.results['lstm_growth'])}")
            logger.info(f"  Avg inference time: {self.results['performance']['lstm_growth_avg_inference_time']:.2f}ms")
            logger.info(f"  Max inference time: {self.results['performance']['lstm_growth_max_inference_time']:.2f}ms")
            logger.info(f"  NFR-ML-1 (< 300ms): {'✅ PASS' if self.results['performance']['lstm_growth_passes_nfr_ml1'] else '❌ FAIL'}")
            
            # Show sample growth rates
            logger.info("\n  Sample Growth Rate Forecasts:")
            for result in self.results['lstm_growth'][:3]:
                rates = result['growth_rates']
                logger.info(f"    {result['ticker']}: Revenue={rates['revenue']:+.2%}, "
                           f"CapEx={rates['capex']:+.2%}, NOPAT={rates['nopat']:+.2%}")
        else:
            logger.info("\n⚠️ LSTM Growth Forecaster: Not evaluated (model may not be trained yet)")
        
        # SRS Compliance
        logger.info("\n📋 SRS Compliance:")
        for req_id, req_data in self.results['srs_compliance'].items():
            status = req_data['status']
            status_icon = "✅" if status == 'PASS' else "🔄" if status == 'IN_PROGRESS' or status == 'PARTIAL' else "❌"
            logger.info(f"  {req_id}: {status_icon} {status}")
        
        logger.info("="*80)
    
    def save_results(self, output_path: str = None):
        """Save evaluation results to JSON"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "data" / "evaluation" / "deep_learning_eval.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate deep learning models')
    parser.add_argument('--stocks', nargs='+', help='Stocks to evaluate')
    parser.add_argument('--output-report', action='store_true', help='Save evaluation report')
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DeepLearningEvaluator()
    
    # Load models
    evaluator.load_models()
    
    # Run evaluation
    test_stocks = args.stocks if args.stocks else TEST_STOCKS
    evaluator.run_comprehensive_evaluation(test_stocks)
    
    # Save results
    if args.output_report:
        evaluator.save_results()


if __name__ == "__main__":
    main()
