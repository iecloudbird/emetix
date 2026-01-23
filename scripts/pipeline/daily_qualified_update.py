"""
Daily Qualified Update - Stage 2 of Quality Screening Pipeline

Runs pillar scoring on attention list stocks and updates qualified_stocks.
Only processes stocks in attention_stocks (not the full universe).

Features:
- 4-Pillar Scoring (Value, Quality, Growth, Safety)
- Classification (Buy/Hold/Watch)
- Momentum check (200MA/50MA)
- Composite score threshold (>=60 to qualify)

Usage:
    python scripts/pipeline/daily_qualified_update.py
    python scripts/pipeline/daily_qualified_update.py --ticker AAPL  # Single stock
    python scripts/pipeline/daily_qualified_update.py --force-all    # Rescore all
"""
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from src.analysis.stock_screener import StockScreener
from src.analysis.pillar_scorer import PillarScorer
from src.analysis.quality_growth_gate import QualityGrowthGate
from src.data.pipeline_db import pipeline_db, is_pipeline_available

logger = get_logger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for MongoDB serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class DailyQualifiedUpdater:
    """
    Stage 2 Updater: Scores attention stocks using 4-pillar system.
    
    Processes attention_stocks and:
    1. Fetches current data
    2. Calculates 4-pillar scores
    3. Determines classification (Buy/Hold/Watch)
    4. Stores qualified stocks if:
       - Composite score >= 50, OR
       - "Pillar Excellence": 2+ pillars score >= 65
    """
    
    # Qualification paths
    MIN_QUALIFIED_SCORE = 50  # Lowered from 60 for better coverage
    PILLAR_EXCELLENCE_THRESHOLD = 65  # Minimum pillar score for excellence
    PILLAR_EXCELLENCE_COUNT = 2  # Number of excellent pillars needed
    
    # Classification thresholds
    BUY_MOS_THRESHOLD = 20
    BUY_SCORE_THRESHOLD = 65  # Lowered from 70
    HOLD_MOS_MIN = -10
    HOLD_MOS_MAX = 20
    
    def __init__(self):
        self.screener = StockScreener()
        self.scorer = PillarScorer()
        self.gate = QualityGrowthGate()
        self.stats = {
            "processed": 0,
            "qualified": 0,
            "disqualified": 0,
            "errors": 0
        }
        self.errors = []
    
    def get_attention_tickers(self, force_all: bool = False) -> List[Dict]:
        """Get tickers from attention list to process"""
        if not is_pipeline_available():
            logger.error("Pipeline DB not available")
            return []
        
        if force_all:
            # Get all attention stocks regardless of status
            return pipeline_db.get_attention_stocks(status=None, limit=1000)
        else:
            # Only active attention stocks
            return pipeline_db.get_attention_stocks(status="active", limit=500)
    
    def calculate_classification(
        self,
        composite_score: float,
        margin_of_safety: float
    ) -> str:
        """
        Determine Buy/Hold/Watch classification.
        
        Buy: MoS >= 20% AND Score >= 70
        Hold: MoS -10% to +20% AND Score >= 70
        Watch: Everything else
        """
        if (margin_of_safety >= self.BUY_MOS_THRESHOLD and 
            composite_score >= self.BUY_SCORE_THRESHOLD):
            return "buy"
        
        if (self.HOLD_MOS_MIN <= margin_of_safety <= self.HOLD_MOS_MAX and
            composite_score >= self.BUY_SCORE_THRESHOLD):
            return "hold"
        
        return "watch"
    
    def check_momentum(self, data: Dict) -> Dict:
        """
        Check momentum status (200MA/50MA).
        
        Returns:
            Momentum indicators dict
        """
        price_vs_200ma = data.get("price_vs_200ma")
        price_vs_50ma = data.get("price_vs_50ma")
        
        below_200ma = price_vs_200ma < 0 if price_vs_200ma is not None else None
        above_50ma = price_vs_50ma > 0 if price_vs_50ma is not None else None
        
        # Ideal entry: below 200MA (accumulation) but above 50MA (stabilizing)
        ideal_entry = None
        if below_200ma is not None and above_50ma is not None:
            ideal_entry = below_200ma and above_50ma
        
        return {
            "price_vs_200ma": round(price_vs_200ma, 2) if price_vs_200ma else None,
            "price_vs_50ma": round(price_vs_50ma, 2) if price_vs_50ma else None,
            "below_200ma": below_200ma,
            "above_50ma": above_50ma,
            "accumulation_zone": below_200ma,
            "stabilizing": above_50ma,
            "ideal_entry": ideal_entry
        }
    
    def check_pillar_excellence(self, pillar_scores: Dict) -> Dict:
        """
        Check if stock qualifies via Pillar Excellence path.
        
        Pillar Excellence: 2+ pillars with score >= 65
        This catches quality stocks like NVO that have excellent Quality + Safety
        but lower composite due to one weak pillar.
        
        Returns:
            {
                "qualified": bool,
                "excellent_pillars": ["quality", "safety"],
                "excellent_count": 2
            }
        """
        excellent_pillars = []
        for pillar, score in pillar_scores.items():
            if score >= self.PILLAR_EXCELLENCE_THRESHOLD:
                excellent_pillars.append(pillar)
        
        return {
            "qualified": len(excellent_pillars) >= self.PILLAR_EXCELLENCE_COUNT,
            "excellent_pillars": excellent_pillars,
            "excellent_count": len(excellent_pillars)
        }

    def process_stock(
        self,
        ticker: str,
        attention_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Process a single stock: fetch data, score, classify.
        
        Returns:
            Qualified stock document or None if doesn't qualify
        """
        try:
            # Fetch current data
            data = self.screener._fetch_stock_data(ticker)
            
            if not data:
                self.stats["errors"] += 1
                return None
            
            # Calculate pillar scores
            scoring_result = self.scorer.calculate_composite(data)
            composite_score = scoring_result["composite_score"]
            pillar_scores = scoring_result["pillars"]
            
            # Check qualification via two paths:
            # Path 1: Composite score >= 50
            # Path 2: Pillar Excellence (2+ pillars >= 65)
            path1_qualified = composite_score >= self.MIN_QUALIFIED_SCORE
            pillar_excellence = self.check_pillar_excellence(pillar_scores)
            path2_qualified = pillar_excellence["qualified"]
            
            # Determine qualification method
            if path1_qualified or path2_qualified:
                self.stats["qualified"] += 1
                qualification_path = []
                if path1_qualified:
                    qualification_path.append(f"composite>={self.MIN_QUALIFIED_SCORE}")
                if path2_qualified:
                    qualification_path.append(f"pillar_excellence:{pillar_excellence['excellent_pillars']}")
            else:
                self.stats["disqualified"] += 1
                return None
            
            # Get margin of safety
            margin_of_safety = data.get("margin_of_safety", 0) or 0
            
            # Determine classification
            classification = self.calculate_classification(
                composite_score, margin_of_safety
            )
            
            # Check momentum
            momentum = self.check_momentum(data)
            
            # Get trigger info from attention data
            triggers = []
            if attention_data:
                triggers = [
                    t.get("type") + (f":path{t.get('path')}" if t.get("path") else "")
                    for t in attention_data.get("triggers", [])
                ]
            
            # Build qualified stock document
            qualified_doc = {
                "ticker": ticker,
                "company_name": data.get("company_name"),
                "sector": data.get("sector"),
                "industry": data.get("industry"),
                
                # Scores
                "pillar_scores": scoring_result["pillars"],
                "composite_score": composite_score,
                "classification": classification,
                
                # Qualification info
                "qualification_path": qualification_path,
                "pillar_excellence": pillar_excellence if path2_qualified else None,
                
                # Valuation
                "current_price": data.get("current_price"),
                "fair_value": data.get("fair_value"),
                "lstm_fair_value": data.get("lstm_fair_value"),
                "fair_value_method": data.get("fair_value_method"),
                "margin_of_safety": margin_of_safety,
                "lstm_predicted_growth": data.get("lstm_predicted_growth"),
                
                # Key Metrics - Value
                "pe_ratio": data.get("pe_ratio"),
                "forward_pe": data.get("forward_pe"),
                "pb_ratio": data.get("pb_ratio"),
                "peg_ratio": data.get("peg_ratio"),
                "ev_ebitda": data.get("ev_ebitda"),
                "fcf_yield": data.get("fcf_yield"),
                
                # Key Metrics - Quality
                "fcf_roic": data.get("fcf_roic"),
                "roe": data.get("roe"),
                "roa": data.get("roa"),
                "profit_margin": data.get("profit_margin"),
                "gross_margin": data.get("gross_margin"),
                "debt_equity": data.get("debt_equity"),
                "current_ratio": data.get("current_ratio"),
                
                # Key Metrics - Growth
                "revenue_growth": data.get("revenue_growth"),
                "earnings_growth": data.get("earnings_growth"),
                
                # Key Metrics - Safety
                "beta": data.get("beta"),
                "volatility": data.get("volatility"),
                "pct_from_52w_high": data.get("pct_from_52w_high"),
                
                # Market Info
                "market_cap": data.get("market_cap"),
                "dividend_yield": data.get("dividend_yield"),
                
                # Momentum
                "momentum": momentum,
                
                # Trigger info
                "triggers": triggers,
                
                # Strength/Weakness
                "analysis": self.scorer.get_strength_weakness_analysis(scoring_result)
            }
            
            return qualified_doc
            
        except Exception as e:
            self.stats["errors"] += 1
            self.errors.append(f"{ticker}: {str(e)}")
            logger.debug(f"Error processing {ticker}: {e}")
            return None
    
    def run_update(
        self,
        ticker: Optional[str] = None,
        force_all: bool = False,
        save_to_db: bool = True
    ) -> Dict:
        """
        Run the daily qualified update.
        
        Args:
            ticker: Optional single ticker to process
            force_all: Process all attention stocks (not just active)
            save_to_db: Whether to save to MongoDB
            
        Returns:
            Update results summary
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("DAILY QUALIFIED UPDATE - Stage 2")
        logger.info("=" * 60)
        
        # Get tickers to process
        if ticker:
            attention_list = [{"ticker": ticker.upper()}]
            logger.info(f"Processing single ticker: {ticker.upper()}")
        else:
            attention_list = self.get_attention_tickers(force_all)
            logger.info(f"Processing {len(attention_list)} attention stocks")
        
        if not attention_list:
            logger.warning("No attention stocks to process")
            return {"status": "no_data", "message": "No attention stocks found"}
        
        # Log scan start
        scan_id = None
        if save_to_db and is_pipeline_available():
            scan_id = pipeline_db.log_scan_start("daily_qualified", len(attention_list))
        
        # Process each stock
        qualified_stocks = []
        classifications = {"buy": 0, "hold": 0, "watch": 0}
        
        for i, attention_doc in enumerate(attention_list):
            stock_ticker = attention_doc.get("ticker")
            if not stock_ticker:
                continue
                
            self.stats["processed"] += 1
            
            if (i + 1) % 25 == 0:
                logger.info(f"Progress: {i + 1}/{len(attention_list)}")
            
            result = self.process_stock(stock_ticker, attention_doc)
            
            if result:
                # Convert numpy types before saving
                result = convert_numpy_types(result)
                
                qualified_stocks.append(result)
                classifications[result["classification"]] += 1
                
                # Save to DB
                if save_to_db and is_pipeline_available():
                    pipeline_db.upsert_qualified_stock(result)
                    
                    # Graduate the attention stock
                    pipeline_db.update_attention_status(stock_ticker, "graduated")
            
            # Rate limiting
            time.sleep(0.3)
        
        # Remove stocks that no longer qualify
        removed = 0
        if save_to_db and is_pipeline_available():
            removed = pipeline_db.remove_unqualified(self.MIN_QUALIFIED_SCORE)
        
        # Log completion
        if scan_id and is_pipeline_available():
            pipeline_db.log_scan_complete(
                scan_id=scan_id,
                graduated=self.stats["qualified"],
                errors=self.errors[:10]
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Build summary
        summary = {
            "scan_type": "daily_qualified",
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": round(duration, 1),
            "attention_count": len(attention_list),
            "stats": self.stats,
            "classifications": classifications,
            "removed_unqualified": removed,
            "qualified_stocks": qualified_stocks,
            "error_sample": self.errors[:5]
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Qualified: {self.stats['qualified']}")
        logger.info(f"Disqualified: {self.stats['disqualified']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"\nClassifications:")
        logger.info(f"  Buy: {classifications['buy']}")
        logger.info(f"  Hold: {classifications['hold']}")
        logger.info(f"  Watch: {classifications['watch']}")
        
        if qualified_stocks:
            logger.info(f"\nTop Qualified Stocks (by score):")
            sorted_stocks = sorted(
                qualified_stocks, 
                key=lambda x: x["composite_score"],
                reverse=True
            )[:10]
            
            for stock in sorted_stocks:
                pillars = stock["pillar_scores"]
                logger.info(
                    f"  {stock['ticker']}: {stock['composite_score']:.1f} "
                    f"({stock['classification'].upper()}) - "
                    f"V:{pillars['value']['score']:.0f} "
                    f"Q:{pillars['quality']['score']:.0f} "
                    f"G:{pillars['growth']['score']:.0f} "
                    f"S:{pillars['safety']['score']:.0f}"
                )
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Daily Qualified Update - Stage 2 Pipeline"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Process single ticker only"
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Process all attention stocks (not just active)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to MongoDB (dry run)"
    )
    
    args = parser.parse_args()
    
    # Run update
    updater = DailyQualifiedUpdater()
    
    try:
        results = updater.run_update(
            ticker=args.ticker,
            force_all=args.force_all,
            save_to_db=not args.no_save
        )
        
        # Print detailed results for single ticker
        if args.ticker and results.get("qualified_stocks"):
            stock = results["qualified_stocks"][0]
            print("\n" + "=" * 60)
            print(f"DETAILED RESULTS: {stock['ticker']}")
            print("=" * 60)
            print(f"Company: {stock['company_name']}")
            print(f"Sector: {stock['sector']}")
            print(f"Classification: {stock['classification'].upper()}")
            print(f"Composite Score: {stock['composite_score']:.1f}")
            print(f"\nPillar Scores:")
            for pillar, data in stock["pillar_scores"].items():
                print(f"  {pillar.upper()}: {data['score']:.1f}")
            print(f"\nValuation:")
            print(f"  Current Price: ${stock['current_price']:.2f}")
            print(f"  Fair Value: ${stock.get('fair_value', 0):.2f}")
            print(f"  Margin of Safety: {stock['margin_of_safety']:.1f}%")
            print(f"\nMomentum:")
            momentum = stock.get("momentum", {})
            print(f"  Price vs 200MA: {momentum.get('price_vs_200ma', 'N/A')}%")
            print(f"  Price vs 50MA: {momentum.get('price_vs_50ma', 'N/A')}%")
            print(f"  Accumulation Zone: {momentum.get('accumulation_zone', 'N/A')}")
            print(f"\nStrengths: {', '.join(stock['analysis']['strengths']) or 'None'}")
            print(f"Weaknesses: {', '.join(stock['analysis']['weaknesses']) or 'None'}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nUpdate interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Update failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
