"""
Consensus Strategy Backtester
==============================
Backtests the 70/20/10 consensus strategy from 2015-2025.

Strategy:
- Buy: Consensus score > 70, MoS > 10%
- Sell: Consensus score < 40, MoS < -10%
- Hold: Otherwise

Metrics:
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Average Holding Period

Usage:
    python scripts/evaluation/backtest_consensus_strategy.py --start 2015-01-01 --end 2025-11-01
    python scripts/evaluation/backtest_consensus_strategy.py --tickers AAPL MSFT GOOGL --quick
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config.settings import PROCESSED_DATA_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)


class ConsensusBacktester:
    """
    Backtests consensus strategy with rebalancing
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        rebalance_freq: str = 'quarterly'
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        
        self.portfolio = {}
        self.cash = initial_capital
        self.portfolio_value_history = []
        self.trades = []
        
    def calculate_simple_dcf_score(
        self,
        ticker: str,
        date: pd.Timestamp
    ) -> Dict:
        """
        Simplified DCF scoring for backtest (no ML)
        Uses historical fundamentals at given date
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical price at date
            hist = stock.history(start=date - timedelta(days=7), end=date + timedelta(days=1))
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Get fundamentals (NOTE: yfinance gives latest, not historical)
            # In production, use point-in-time data from database
            info = stock.info
            
            # Simple valuation based on multiples
            pe = info.get('trailingPE', 20)
            pb = info.get('priceToBook', 2)
            
            # Sector average P/E (simplified)
            sector_pe = 18
            
            # Score based on relative valuation
            if pe < 0.8 * sector_pe:
                dcf_score = 80
                mos = 25
            elif pe < 1.2 * sector_pe:
                dcf_score = 60
                mos = 5
            elif pe < 2.0 * sector_pe:
                dcf_score = 40
                mos = -10
            else:
                dcf_score = 20
                mos = -30
            
            # Risk score (simplified - based on beta)
            beta = info.get('beta', 1.0)
            if beta < 0.8:
                risk_score = 70
                risk_class = 'Low'
            elif beta < 1.2:
                risk_score = 50
                risk_class = 'Medium'
            else:
                risk_score = 30
                risk_class = 'High'
            
            # P/E sanity score
            if pe < 0.8 * sector_pe:
                pe_score = 100
            elif pe < 1.2 * sector_pe:
                pe_score = 70
            elif pe < 2.0 * sector_pe:
                pe_score = 40
            else:
                pe_score = 20
            
            # Consensus (70/20/10)
            consensus_score = (
                dcf_score * 0.70 +
                risk_score * 0.20 +
                pe_score * 0.10
            )
            
            return {
                'ticker': ticker,
                'date': date,
                'price': current_price,
                'consensus_score': consensus_score,
                'margin_of_safety': mos,
                'risk_class': risk_class,
                'pe_ratio': pe,
                'beta': beta
            }
            
        except Exception as e:
            logger.warning(f"Scoring failed for {ticker} at {date}: {e}")
            return None
    
    def generate_signal(self, score_data: Dict) -> str:
        """
        Generate buy/hold/sell signal
        """
        score = score_data['consensus_score']
        mos = score_data['margin_of_safety']
        
        if score > 70 and mos > 10:
            return 'BUY'
        elif score < 40 and mos < -10:
            return 'SELL'
        else:
            return 'HOLD'
    
    def rebalance_portfolio(
        self,
        tickers: List[str],
        date: pd.Timestamp
    ):
        """
        Rebalance portfolio at given date
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Rebalancing Portfolio: {date.date()}")
        logger.info(f"{'='*80}")
        
        # Score all tickers
        scores = []
        for ticker in tickers:
            score_data = self.calculate_simple_dcf_score(ticker, date)
            if score_data:
                signal = self.generate_signal(score_data)
                score_data['signal'] = signal
                scores.append(score_data)
        
        if not scores:
            logger.warning("No valid scores, skipping rebalance")
            return
        
        scores_df = pd.DataFrame(scores)
        
        # Filter buy candidates
        buy_candidates = scores_df[scores_df['signal'] == 'BUY'].sort_values(
            'consensus_score', ascending=False
        )
        
        # Sell existing positions if needed
        sells = []
        for ticker in list(self.portfolio.keys()):
            ticker_scores = scores_df[scores_df['ticker'] == ticker]
            if not ticker_scores.empty:
                signal = ticker_scores.iloc[0]['signal']
                if signal == 'SELL':
                    # Sell position
                    shares = self.portfolio[ticker]['shares']
                    price = ticker_scores.iloc[0]['price']
                    proceeds = shares * price
                    self.cash += proceeds
                    
                    cost_basis = self.portfolio[ticker]['cost_basis']
                    pnl = proceeds - cost_basis
                    pnl_pct = (pnl / cost_basis) * 100
                    
                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'value': proceeds,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    
                    sells.append(ticker)
                    logger.info(f"   SELL {ticker}: {shares} shares @ ${price:.2f} | P&L: {pnl_pct:+.1f}%")
                    del self.portfolio[ticker]
        
        # Buy new positions (top 10 candidates)
        max_positions = 10
        current_positions = len(self.portfolio)
        slots_available = max_positions - current_positions
        
        if slots_available > 0 and not buy_candidates.empty:
            allocation_per_stock = self.cash / slots_available
            
            for idx, row in buy_candidates.head(slots_available).iterrows():
                ticker = row['ticker']
                price = row['price']
                shares = int(allocation_per_stock / price)
                
                if shares > 0:
                    cost = shares * price
                    self.cash -= cost
                    
                    self.portfolio[ticker] = {
                        'shares': shares,
                        'cost_basis': cost,
                        'entry_price': price,
                        'entry_date': date
                    }
                    
                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': cost,
                        'pnl': 0,
                        'pnl_pct': 0
                    })
                    
                    logger.info(f"   BUY  {ticker}: {shares} shares @ ${price:.2f} | Score: {row['consensus_score']:.1f}")
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for ticker, position in self.portfolio.items():
            ticker_data = scores_df[scores_df['ticker'] == ticker]
            if not ticker_data.empty:
                current_price = ticker_data.iloc[0]['price']
                portfolio_value += position['shares'] * current_price
        
        self.portfolio_value_history.append({
            'date': date,
            'value': portfolio_value,
            'cash': self.cash,
            'positions': len(self.portfolio)
        })
        
        logger.info(f"\n   Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"   Cash: ${self.cash:,.2f}")
        logger.info(f"   Positions: {len(self.portfolio)}")
    
    def run_backtest(self, tickers: List[str]):
        """
        Run full backtest
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Backtest: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"{'='*80}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Rebalance Frequency: {self.rebalance_freq}")
        logger.info(f"Universe: {len(tickers)} stocks")
        
        # Generate rebalance dates
        if self.rebalance_freq == 'quarterly':
            freq = 'Q'
        elif self.rebalance_freq == 'monthly':
            freq = 'M'
        else:
            freq = 'Y'
        
        rebalance_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=freq
        )
        
        # Run rebalances
        for date in rebalance_dates:
            self.rebalance_portfolio(tickers, date)
        
        # Calculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """
        Calculate backtest performance metrics
        """
        if not self.portfolio_value_history:
            logger.error("No portfolio history to analyze")
            return
        
        pv_df = pd.DataFrame(self.portfolio_value_history)
        pv_df['returns'] = pv_df['value'].pct_change()
        
        # Total return
        final_value = pv_df['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        years = (self.end_date - self.start_date).days / 365.25
        annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100
        
        # Sharpe ratio (assume 2% risk-free rate)
        excess_returns = pv_df['returns'].dropna() - (0.02 / 252)  # Daily risk-free
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        cummax = pv_df['value'].cummax()
        drawdown = (pv_df['value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        trades_df = pd.DataFrame(self.trades)
        closed_trades = trades_df[trades_df['action'] == 'SELL']
        if not closed_trades.empty:
            wins = (closed_trades['pnl'] > 0).sum()
            win_rate = wins / len(closed_trades) * 100
        else:
            win_rate = 0
        
        # Average holding period
        if not closed_trades.empty:
            avg_holding_days = 90  # Placeholder (would need entry/exit matching)
        else:
            avg_holding_days = 0
        
        # Print results
        print(f"\n{'='*80}")
        print(f"📊 Backtest Results")
        print(f"{'='*80}\n")
        
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Duration: {years:.1f} years")
        
        print(f"\n💰 Returns:")
        print(f"   Initial Capital:     ${self.initial_capital:,.2f}")
        print(f"   Final Value:         ${final_value:,.2f}")
        print(f"   Total Return:        {total_return:+.2f}%")
        print(f"   Annualized Return:   {annualized_return:+.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:        {sharpe_ratio:.2f}")
        print(f"   Max Drawdown:        {max_drawdown:.2f}%")
        
        print(f"\n📊 Trading Stats:")
        print(f"   Total Trades:        {len(trades_df)}")
        print(f"   Closed Trades:       {len(closed_trades)}")
        print(f"   Win Rate:            {win_rate:.1f}%")
        if not closed_trades.empty:
            avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl_pct'].mean()
            avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl_pct'].mean()
            print(f"   Avg Win:             {avg_win:+.1f}%")
            print(f"   Avg Loss:            {avg_loss:+.1f}%")
        
        print(f"\n{'='*80}\n")
        
        # Save results
        results_dir = PROCESSED_DATA_DIR / 'backtest'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Portfolio value history
        pv_path = results_dir / f'portfolio_value_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        pv_df.to_csv(pv_path, index=False)
        
        # Trades
        trades_path = results_dir / f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(trades_path, index=False)
        
        logger.info(f"✅ Results saved:")
        logger.info(f"   Portfolio: {pv_path}")
        logger.info(f"   Trades: {trades_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Consensus Strategy")
    parser.add_argument(
        '--start',
        type=str,
        default='2015-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2025-11-01',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital (default: $100,000)'
    )
    parser.add_argument(
        '--rebalance',
        type=str,
        default='quarterly',
        choices=['monthly', 'quarterly', 'yearly'],
        help='Rebalancing frequency'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Specific tickers to backtest (default: S&P 500 sample)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with limited tickers'
    )
    
    args = parser.parse_args()
    
    # Get tickers
    if args.tickers:
        tickers = args.tickers
    elif args.quick:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'V', 'JNJ', 'WMT']
    else:
        # Full S&P 500 sample
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ',
            'WMT', 'JPM', 'PG', 'XOM', 'UNH', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'CSCO', 'ACN', 'MCD', 'ABT', 'DHR',
            'WFC', 'DIS', 'ADBE', 'VZ', 'CRM', 'NEE', 'CMCSA', 'TXN', 'PM', 'NKE',
            'UPS', 'RTX', 'HON', 'ORCL', 'INTC', 'QCOM', 'BMY', 'LIN', 'AMD', 'UNP'
        ]
    
    # Initialize backtester
    backtester = ConsensusBacktester(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_freq=args.rebalance
    )
    
    # Run backtest
    backtester.run_backtest(tickers)


if __name__ == "__main__":
    main()
