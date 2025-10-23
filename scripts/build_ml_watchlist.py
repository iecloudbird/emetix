"""
Build ML-Powered Watchlist - Batch Processing Script
Generates ranked watchlists using ML-enhanced scoring (LSTM-DCF + RF Ensemble)
"""
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import yfinance as yf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.watchlist_manager_agent import WatchlistManagerAgent
from config.logging_config import get_logger
from config.settings import CACHE_DIR

logger = get_logger(__name__)


def fetch_stock_scores(ticker: str) -> Dict:
    """
    Fetch quick heuristic scores for a stock
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dict with growth, sentiment, valuation, risk, macro scores
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Growth score (revenue growth)
        revenue_growth = info.get('revenueGrowth', 0)
        growth_score = min(max(revenue_growth * 5, 0), 1) if revenue_growth else 0.5
        
        # Valuation score (P/E based)
        pe_ratio = info.get('trailingPE', 20)
        valuation_score = min(max((25 - pe_ratio) / 25, 0), 1) if pe_ratio else 0.5
        
        # Sentiment score (1-month momentum)
        hist = stock.history(period="1mo")
        if not hist.empty:
            month_return = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
            sentiment_score = min(max((month_return + 0.1) * 2.5, 0), 1)
        else:
            sentiment_score = 0.5
        
        # Risk score (beta-based)
        beta = info.get('beta', 1.0)
        risk_score = min(max((2 - beta) / 2, 0), 1) if beta else 0.5
        
        # Macro score (default)
        macro_score = 0.6
        
        return {
            'ticker': ticker,
            'growth': round(growth_score, 2),
            'sentiment': round(sentiment_score, 2),
            'valuation': round(valuation_score, 2),
            'risk': round(risk_score, 2),
            'macro': round(macro_score, 2)
        }
        
    except Exception as e:
        logger.error(f"Error fetching scores for {ticker}: {e}")
        # Return default scores on error
        return {
            'ticker': ticker,
            'growth': 0.5,
            'sentiment': 0.5,
            'valuation': 0.5,
            'risk': 0.5,
            'macro': 0.5
        }


def build_watchlist(
    tickers: List[str],
    output_format: str = 'console',
    output_path: str = None,
    verbose: bool = True
) -> Dict:
    """
    Build ML-powered watchlist for multiple stocks
    
    Args:
        tickers: List of stock ticker symbols
        output_format: Output format ('console', 'json', 'csv')
        output_path: Path to save output file
        verbose: Print progress messages
        
    Returns:
        Dict with watchlist results
    """
    if verbose:
        print("\n" + "="*80)
        print("  ML-POWERED WATCHLIST BUILDER - EMETIX")
        print("="*80 + "\n")
    
    # Initialize WatchlistManagerAgent
    if verbose:
        print("[*] Initializing ML-enhanced WatchlistManagerAgent...")
    
    try:
        agent = WatchlistManagerAgent()
        if not agent.ml_models_available:
            print("[!] WARNING: ML models not available, using traditional scoring")
        else:
            if verbose:
                print("[OK] ML models loaded (LSTM-DCF + RF Ensemble)\n")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return {'error': str(e)}
    
    # Fetch scores for all tickers
    if verbose:
        print(f"[*] Fetching data for {len(tickers)} stocks...")
    
    stocks_data = []
    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i}/{len(tickers)}] Processing {ticker}...", end='\r')
        
        scores = fetch_stock_scores(ticker)
        stocks_data.append(scores)
    
    if verbose:
        print(f"\n[OK] Data fetched for {len(stocks_data)} stocks\n")
    
    # Build watchlist using agent
    if verbose:
        print("[*] Building ML-enhanced watchlist...")
    
    try:
        watchlist_result = agent.build_watchlist(stocks_data)
        
        if verbose:
            print("[OK] Watchlist generated!\n")
        
    except Exception as e:
        logger.error(f"Error building watchlist: {e}")
        return {'error': str(e)}
    
    # Format output
    result = {
        'timestamp': datetime.now().isoformat(),
        'tickers_analyzed': len(tickers),
        'ml_models_used': agent.ml_models_available,
        'watchlist': watchlist_result
    }
    
    # Save output if requested
    if output_path:
        save_output(result, output_format, output_path, verbose)
    
    # Display results
    if output_format == 'console':
        display_watchlist_console(watchlist_result, verbose)
    
    return result


def display_watchlist_console(watchlist_result: Dict, verbose: bool = True):
    """Display watchlist in formatted console output"""
    if not verbose:
        return
    
    print("="*80)
    print("  WATCHLIST RESULTS")
    print("="*80 + "\n")
    
    # Extract ranked stocks from result string
    result_str = str(watchlist_result)
    
    print(result_str)
    print("\n" + "="*80 + "\n")


def save_output(result: Dict, output_format: str, output_path: str, verbose: bool):
    """Save watchlist output to file"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if verbose:
                print(f"[OK] Saved JSON to {output_path}")
        
        elif output_format == 'csv':
            # Extract watchlist data and convert to DataFrame
            # This is a simplified version - customize based on actual output structure
            df = pd.DataFrame([result])
            df.to_csv(output_path, index=False)
            if verbose:
                print(f"[OK] Saved CSV to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        if verbose:
            print(f"[X] Failed to save output: {e}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Build ML-powered watchlist with LSTM-DCF and RF Ensemble scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build watchlist for tech stocks (console output)
  python scripts/build_ml_watchlist.py AAPL MSFT GOOGL TSLA NVDA

  # Save to JSON file
  python scripts/build_ml_watchlist.py AAPL MSFT GOOGL --output watchlist.json --format json

  # Save to CSV with custom path
  python scripts/build_ml_watchlist.py AAPL MSFT GOOGL --output data/cache/watchlist.csv --format csv

  # Build watchlist from file
  python scripts/build_ml_watchlist.py --file tickers.txt

  # Quiet mode (minimal output)
  python scripts/build_ml_watchlist.py AAPL MSFT GOOGL --quiet
        """
    )
    
    parser.add_argument(
        'tickers',
        nargs='*',
        help='Stock ticker symbols (space-separated)'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Read tickers from file (one per line)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path'
    )
    
    parser.add_argument(
        '--format',
        choices=['console', 'json', 'csv'],
        default='console',
        help='Output format (default: console)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    args = parser.parse_args()
    
    # Get tickers from args or file
    tickers = []
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        parser.print_help()
        return
    
    if not tickers:
        print("Error: No tickers provided")
        return
    
    # Build watchlist
    verbose = not args.quiet
    
    result = build_watchlist(
        tickers=tickers,
        output_format=args.format,
        output_path=args.output,
        verbose=verbose
    )
    
    if 'error' in result:
        print(f"[X] Error: {result['error']}")
        sys.exit(1)
    
    if verbose:
        print("[*] Watchlist building complete!")


if __name__ == "__main__":
    main()
