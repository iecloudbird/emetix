"""
Pipeline MongoDB Client for Emetix Phase 3

Handles connection to MongoDB Atlas emetix_pipeline database for:
- attention_stocks: Stage 1 trigger matches
- qualified_stocks: Stage 2 validated stocks with pillar scores
- scan_history: Scan job logs
- us_stocks: Universe of tradeable stocks

Uses the emetix_pipeline database, separate from the main emetix database.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from bson.objectid import ObjectId
import certifi

from config.logging_config import get_logger
from config.settings import get_env

logger = get_logger(__name__)


# MongoDB connection settings
MONGODB_URI = get_env("MONGODB_URI", "")
PIPELINE_DATABASE = "emetix_pipeline"  # Separate database for pipeline data


class PipelineDBClient:
    """
    MongoDB client for the Quality Screening Pipeline.
    
    Collections:
    - attention_stocks: Stocks that triggered attention (Stage 1)
    - qualified_stocks: Validated stocks with pillar scores (Stage 2)
    - scan_history: Scan job metadata and logs
    - us_stocks: Universe of tradeable US stocks
    """
    
    _instance: Optional['PipelineDBClient'] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    
    def __new__(cls):
        """Singleton pattern for pipeline client"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._connected = False
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas pipeline database.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self._connected and self._client is not None:
            return True
            
        if not MONGODB_URI:
            logger.warning("MONGODB_URI not configured - Pipeline features disabled")
            return False
        
        try:
            self._client = MongoClient(
                MONGODB_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                retryWrites=True,
                w='majority'
            )
            
            # Verify connection
            self._client.admin.command('ping')
            
            self._db = self._client[PIPELINE_DATABASE]
            self._connected = True
            logger.info(f"Connected to MongoDB Atlas: {PIPELINE_DATABASE}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to pipeline DB: {e}")
            self._connected = False
            return False
    
    @property
    def db(self) -> Optional[Database]:
        """Get database, connecting if needed"""
        if not self._connected:
            self.connect()
        return self._db
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None
    
    # =========================================================================
    # ATTENTION STOCKS (Stage 1)
    # =========================================================================
    
    def upsert_attention_stock(
        self,
        ticker: str,
        triggers: List[Dict],
        status: str = "active"
    ) -> bool:
        """
        Insert or update a stock in the attention list.
        
        Args:
            ticker: Stock ticker symbol
            triggers: List of trigger objects that fired
            status: active, graduated, or expired
            
        Returns:
            True if successful
        """
        if not self.connect():
            return False
        
        try:
            now = datetime.now(timezone.utc)
            
            # Check if stock already exists
            existing = self._db.attention_stocks.find_one({"ticker": ticker})
            
            if existing:
                # Merge new triggers with existing
                existing_triggers = existing.get("triggers", [])
                existing_types = {t.get("type") for t in existing_triggers}
                
                for new_trigger in triggers:
                    if new_trigger.get("type") not in existing_types:
                        existing_triggers.append({
                            **new_trigger,
                            "triggered_at": now
                        })
                
                self._db.attention_stocks.update_one(
                    {"ticker": ticker},
                    {
                        "$set": {
                            "triggers": existing_triggers,
                            "last_updated": now,
                            "status": status
                        }
                    }
                )
            else:
                # Insert new
                for trigger in triggers:
                    trigger["triggered_at"] = now
                    
                self._db.attention_stocks.insert_one({
                    "ticker": ticker,
                    "triggers": triggers,
                    "first_triggered": now,
                    "last_updated": now,
                    "status": status
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert attention stock {ticker}: {e}")
            return False
    
    def get_attention_stocks(
        self,
        status: str = "active",
        limit: int = 500
    ) -> List[Dict]:
        """
        Get stocks in the attention list.
        
        Args:
            status: Filter by status (active, graduated, expired, or None for all)
            limit: Maximum number to return
            
        Returns:
            List of attention stock documents
        """
        if not self.connect():
            return []
        
        try:
            query = {}
            if status:
                query["status"] = status
            
            cursor = self._db.attention_stocks.find(query).sort(
                "last_updated", DESCENDING
            ).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Failed to get attention stocks: {e}")
            return []
    
    def update_attention_status(self, ticker: str, status: str) -> bool:
        """Update status of an attention stock"""
        if not self.connect():
            return False
        
        try:
            result = self._db.attention_stocks.update_one(
                {"ticker": ticker},
                {"$set": {"status": status, "last_updated": datetime.now(timezone.utc)}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update attention status: {e}")
            return False
    
    def bulk_expire_old_attention(self, days_old: int = 30) -> int:
        """
        Expire attention stocks not updated in N days.
        
        Returns:
            Number of stocks expired
        """
        if not self.connect():
            return 0
        
        try:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            result = self._db.attention_stocks.update_many(
                {
                    "status": "active",
                    "last_updated": {"$lt": cutoff}
                },
                {"$set": {"status": "expired"}}
            )
            return result.modified_count
        except Exception as e:
            logger.error(f"Failed to expire old attention stocks: {e}")
            return 0
    
    def clear_attention_stocks(self) -> int:
        """
        Clear all attention stocks (for fresh pipeline re-run).
        
        Returns:
            Number of documents deleted
        """
        if not self.connect():
            return 0
        
        try:
            result = self._db.attention_stocks.delete_many({})
            logger.info(f"Cleared {result.deleted_count} attention stocks")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear attention stocks: {e}")
            return 0
    
    def clear_qualified_stocks(self) -> int:
        """
        Clear all qualified stocks (for fresh Stage 2 re-run).
        
        Returns:
            Number of documents deleted
        """
        if not self.connect():
            return 0
        
        try:
            result = self._db.qualified_stocks.delete_many({})
            logger.info(f"Cleared {result.deleted_count} qualified stocks")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to clear qualified stocks: {e}")
            return 0
    
    def clear_all_pipeline_data(self) -> Dict[str, int]:
        """
        Clear all pipeline data (attention + qualified) for fresh start.
        Does NOT clear universe collection.
        
        Returns:
            Dict with deleted counts per collection
        """
        return {
            "attention_stocks": self.clear_attention_stocks(),
            "qualified_stocks": self.clear_qualified_stocks()
        }
    
    # =========================================================================
    # QUALIFIED STOCKS (Stage 2)
    # =========================================================================
    
    def upsert_qualified_stock(self, stock_data: Dict) -> bool:
        """
        Insert or update a qualified stock with full scoring data.
        
        Args:
            stock_data: Dictionary with all scoring fields
            
        Returns:
            True if successful
        """
        if not self.connect():
            return False
        
        try:
            ticker = stock_data.get("ticker")
            if not ticker:
                return False
            
            stock_data["last_updated"] = datetime.now(timezone.utc)
            
            self._db.qualified_stocks.update_one(
                {"ticker": ticker},
                {"$set": stock_data},
                upsert=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert qualified stock: {e}")
            return False
    
    def bulk_upsert_qualified(self, stocks: List[Dict]) -> int:
        """
        Bulk upsert multiple qualified stocks.
        
        Returns:
            Number of stocks upserted
        """
        if not self.connect() or not stocks:
            return 0
        
        try:
            now = datetime.now(timezone.utc)
            operations = []
            
            for stock in stocks:
                ticker = stock.get("ticker")
                if not ticker:
                    continue
                    
                stock["last_updated"] = now
                operations.append(
                    UpdateOne(
                        {"ticker": ticker},
                        {"$set": stock},
                        upsert=True
                    )
                )
            
            if operations:
                result = self._db.qualified_stocks.bulk_write(operations)
                return result.upserted_count + result.modified_count
            return 0
            
        except Exception as e:
            logger.error(f"Failed to bulk upsert qualified stocks: {e}")
            return 0
    
    def get_qualified_stocks(
        self,
        classification: Optional[str] = None,
        min_score: float = 0,
        sector: Optional[str] = None,
        limit: int = 100,
        sort_by: str = "composite_score"
    ) -> List[Dict]:
        """
        Get qualified stocks with optional filters.
        
        Args:
            classification: buy, hold, watch, or None for all (case-insensitive)
            min_score: Minimum composite score
            sector: Filter by sector
            limit: Maximum results
            sort_by: Field to sort by (descending)
            
        Returns:
            List of qualified stock documents
        """
        if not self.connect():
            return []
        
        try:
            query = {}
            
            if classification:
                # Case-insensitive match (DB stores "Buy", API may send "buy")
                query["classification"] = {"$regex": f"^{classification}$", "$options": "i"}
            
            if min_score > 0:
                query["composite_score"] = {"$gte": min_score}
            
            if sector:
                query["sector"] = sector
            
            cursor = self._db.qualified_stocks.find(query).sort(
                sort_by, DESCENDING
            ).limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Failed to get qualified stocks: {e}")
            return []
    
    def get_qualified_by_ticker(self, ticker: str) -> Optional[Dict]:
        """Get a single qualified stock by ticker"""
        if not self.connect():
            return None
        
        try:
            return self._db.qualified_stocks.find_one({"ticker": ticker})
        except Exception as e:
            logger.error(f"Failed to get qualified stock {ticker}: {e}")
            return None
    
    def get_classified_counts(self) -> Dict[str, int]:
        """Get count of stocks in each classification"""
        if not self.connect():
            return {}
        
        try:
            pipeline = [
                {"$group": {"_id": "$classification", "count": {"$sum": 1}}}
            ]
            result = list(self._db.qualified_stocks.aggregate(pipeline))
            return {doc["_id"]: doc["count"] for doc in result if doc["_id"]}
        except Exception as e:
            logger.error(f"Failed to get classification counts: {e}")
            return {}
    
    def remove_unqualified(self, min_score: float = 60) -> int:
        """
        Remove stocks that no longer meet qualification threshold.
        
        Returns:
            Number removed
        """
        if not self.connect():
            return 0
        
        try:
            result = self._db.qualified_stocks.delete_many({
                "composite_score": {"$lt": min_score}
            })
            return result.deleted_count
        except Exception as e:
            logger.error(f"Failed to remove unqualified stocks: {e}")
            return 0
    
    # =========================================================================
    # SCAN HISTORY
    # =========================================================================
    
    def log_scan_start(
        self,
        scan_type: str,
        universe_size: int = 0
    ) -> Optional[str]:
        """
        Log the start of a scan job.
        
        Returns:
            Scan job ID
        """
        if not self.connect():
            return None
        
        try:
            doc = {
                "scan_type": scan_type,
                "started_at": datetime.now(timezone.utc),
                "universe_size": universe_size,
                "status": "running",
                "new_attention": 0,
                "graduated": 0,
                "expired": 0,
                "errors": []
            }
            result = self._db.scan_history.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to log scan start: {e}")
            return None
    
    def log_scan_complete(
        self,
        scan_id: str,
        new_attention: int = 0,
        graduated: int = 0,
        expired: int = 0,
        errors: List[str] = None
    ) -> bool:
        """Log completion of a scan job"""
        if not self.connect():
            return False
        
        try:
            self._db.scan_history.update_one(
                {"_id": ObjectId(scan_id)},
                {
                    "$set": {
                        "completed_at": datetime.now(timezone.utc),
                        "status": "completed",
                        "new_attention": new_attention,
                        "graduated": graduated,
                        "expired": expired,
                        "errors": errors or []
                    }
                }
            )
            return True
        except Exception as e:
            logger.error(f"Failed to log scan complete: {e}")
            return False
    
    def get_recent_scans(self, limit: int = 10) -> List[Dict]:
        """Get recent scan history"""
        if not self.connect():
            return []
        
        try:
            cursor = self._db.scan_history.find().sort(
                "started_at", DESCENDING
            ).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to get scan history: {e}")
            return []
    
    # =========================================================================
    # UNIVERSE (US_STOCKS)
    # =========================================================================
    
    def get_universe_tickers(
        self,
        sector: Optional[str] = None,
        min_market_cap: float = 0
    ) -> List[str]:
        """
        Get list of tickers from the universe.
        
        Args:
            sector: Filter by sector (optional)
            min_market_cap: Minimum market cap filter
            
        Returns:
            List of ticker symbols
        """
        if not self.connect():
            return []
        
        try:
            query = {}
            if sector:
                query["sector"] = sector
            if min_market_cap > 0:
                query["market_cap"] = {"$gte": min_market_cap}
            
            cursor = self._db.us_stocks.find(
                query,
                {"ticker": 1, "_id": 0}
            )
            return [doc["ticker"] for doc in cursor]
            
        except Exception as e:
            logger.error(f"Failed to get universe tickers: {e}")
            return []
    
    def get_universe_stats(self) -> Dict:
        """Get universe statistics"""
        if not self.connect():
            return {}
        
        try:
            total = self._db.us_stocks.count_documents({})
            
            # Count by sector
            sector_pipeline = [
                {"$group": {"_id": "$sector", "count": {"$sum": 1}}}
            ]
            sector_counts = {
                doc["_id"]: doc["count"] 
                for doc in self._db.us_stocks.aggregate(sector_pipeline)
                if doc["_id"]
            }
            
            return {
                "total_stocks": total,
                "by_sector": sector_counts
            }
        except Exception as e:
            logger.error(f"Failed to get universe stats: {e}")
            return {}
    
    def upsert_universe_stock(self, stock_data: Dict) -> bool:
        """Add or update a stock in the universe"""
        if not self.connect():
            return False
        
        try:
            ticker = stock_data.get("ticker")
            if not ticker:
                return False
            
            stock_data["last_updated"] = datetime.now(timezone.utc)
            
            self._db.us_stocks.update_one(
                {"ticker": ticker},
                {"$set": stock_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert universe stock: {e}")
            return False
    
    # =========================================================================
    # CURATED WATCHLIST (Stage 3)
    # =========================================================================
    
    def save_curated_watchlist(self, curated_data: Dict) -> bool:
        """
        Save the curated watchlist from Stage 3.
        
        Replaces the entire curated_watchlist collection with fresh data.
        
        Args:
            curated_data: Output from Stage 3 curation containing:
                - summary: counts by category
                - watchlists: strong_buy, moderate_buy, hold, watch
                - generated_at: timestamp
                
        Returns:
            True if saved successfully
        """
        if not self.connect():
            return False
        
        try:
            # Clear existing curated data
            self._db.curated_watchlist.delete_many({})
            
            # Insert metadata document
            metadata = {
                "_id": "metadata",
                "generated_at": curated_data.get("generated_at"),
                "curation_version": curated_data.get("curation_version", "3.1"),
                "source_qualified_count": curated_data.get("source_qualified_count", 0),
                "curated_total": curated_data.get("curated_total", 0),
                "summary": curated_data.get("summary", {})
            }
            self._db.curated_watchlist.insert_one(metadata)
            
            # Insert stocks by category
            watchlists = curated_data.get("watchlists", {})
            
            # Strong Buy
            for stock in watchlists.get("strong_buy", []):
                stock["_category"] = "strong_buy"
                stock["_list_type"] = "buy"
                self._db.curated_watchlist.insert_one(stock)
            
            # Moderate Buy
            for stock in watchlists.get("moderate_buy", []):
                stock["_category"] = "moderate_buy"
                stock["_list_type"] = "buy"
                self._db.curated_watchlist.insert_one(stock)
            
            # Hold
            for stock in watchlists.get("hold", []):
                stock["_category"] = "hold"
                stock["_list_type"] = "hold"
                self._db.curated_watchlist.insert_one(stock)
            
            # Watch (nested by sub-category)
            watch_lists = watchlists.get("watch", {})
            for sub_category, stocks in watch_lists.items():
                for stock in stocks:
                    stock["_category"] = sub_category
                    stock["_list_type"] = "watch"
                    self._db.curated_watchlist.insert_one(stock)
            
            logger.info(f"Saved curated watchlist: {curated_data.get('curated_total', 0)} stocks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save curated watchlist: {e}")
            return False
    
    def get_curated_watchlist(self) -> Optional[Dict]:
        """
        Get the full curated watchlist for frontend display.
        
        Returns:
            {
                "metadata": {...},
                "strong_buy": [...],
                "moderate_buy": [...],
                "hold": [...],
                "watch": {"growth_watch": [...], ...}
            }
        """
        if not self.connect():
            return None
        
        try:
            # Get metadata
            metadata = self._db.curated_watchlist.find_one({"_id": "metadata"})
            if not metadata:
                return None
            
            # Get stocks by category
            strong_buy = list(self._db.curated_watchlist.find(
                {"_category": "strong_buy"}
            ).sort("composite_score", DESCENDING))
            
            moderate_buy = list(self._db.curated_watchlist.find(
                {"_category": "moderate_buy"}
            ).sort("composite_score", DESCENDING))
            
            hold = list(self._db.curated_watchlist.find(
                {"_category": "hold"}
            ).sort("composite_score", DESCENDING))
            
            # Get watch stocks grouped by sub-category
            watch_cursor = self._db.curated_watchlist.find(
                {"_list_type": "watch"}
            ).sort("composite_score", DESCENDING)
            
            watch_by_category = {}
            for stock in watch_cursor:
                cat = stock.get("_category", "other")
                if cat not in watch_by_category:
                    watch_by_category[cat] = []
                watch_by_category[cat].append(stock)
            
            # Clean up internal fields for response
            def clean_stock(s):
                s["_id"] = str(s["_id"]) if "_id" in s else None
                s.pop("_category", None)
                s.pop("_list_type", None)
                return s
            
            return {
                "metadata": {
                    "generated_at": metadata.get("generated_at"),
                    "curation_version": metadata.get("curation_version"),
                    "source_qualified_count": metadata.get("source_qualified_count"),
                    "curated_total": metadata.get("curated_total"),
                    "summary": metadata.get("summary", {})
                },
                "strong_buy": [clean_stock(s) for s in strong_buy],
                "moderate_buy": [clean_stock(s) for s in moderate_buy],
                "hold": [clean_stock(s) for s in hold],
                "watch": {k: [clean_stock(s) for s in v] for k, v in watch_by_category.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to get curated watchlist: {e}")
            return None
    
    def get_curated_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """
        Get curated stocks by category.
        
        Args:
            category: strong_buy, moderate_buy, hold, or watch sub-category
            limit: Max stocks to return
            
        Returns:
            List of curated stock documents
        """
        if not self.connect():
            return []
        
        try:
            stocks = list(self._db.curated_watchlist.find(
                {"_category": category}
            ).sort("composite_score", DESCENDING).limit(limit))
            
            # Clean up
            for s in stocks:
                s["_id"] = str(s["_id"]) if "_id" in s else None
                s.pop("_category", None)
                s.pop("_list_type", None)
            
            return stocks
            
        except Exception as e:
            logger.error(f"Failed to get curated stocks for {category}: {e}")
            return []
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("Pipeline DB connection closed")
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline state"""
        if not self.connect():
            return {}
        
        try:
            # Get curated metadata if available
            curated_meta = self._db.curated_watchlist.find_one({"_id": "metadata"})
            curated_info = None
            if curated_meta:
                curated_info = {
                    "generated_at": curated_meta.get("generated_at"),
                    "total": curated_meta.get("curated_total", 0),
                    "summary": curated_meta.get("summary", {})
                }
            
            return {
                "attention_active": self._db.attention_stocks.count_documents({"status": "active"}),
                "attention_graduated": self._db.attention_stocks.count_documents({"status": "graduated"}),
                "qualified_total": self._db.qualified_stocks.count_documents({}),
                "classifications": self.get_classified_counts(),
                "curated": curated_info,
                "universe_size": self._db.us_stocks.count_documents({}),
                "last_scan": self.get_recent_scans(1)[0] if self.get_recent_scans(1) else None
            }
        except Exception as e:
            logger.error(f"Failed to get pipeline summary: {e}")
            return {}


# Singleton instance
pipeline_db = PipelineDBClient()


def get_pipeline_db() -> Optional[Database]:
    """Get pipeline database instance"""
    return pipeline_db.db


def is_pipeline_available() -> bool:
    """Check if pipeline DB is available"""
    return pipeline_db.connect()
