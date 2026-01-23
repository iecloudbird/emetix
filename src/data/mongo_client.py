"""
MongoDB Atlas Client for Emetix

Handles connection to MongoDB Atlas for storing:
- Watchlists (public/shared)
- Investment strategies
- Educational content
- Session-based data (no user auth required)

NOTE: Personal risk capacity is stored client-side in localStorage,
not in MongoDB, to avoid requiring user authentication.
"""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from bson.objectid import ObjectId
import certifi

from config.logging_config import get_logger
from config.settings import get_env

logger = get_logger(__name__)


# MongoDB connection settings from environment
MONGODB_URI = get_env("MONGODB_URI", "")
MONGODB_DATABASE = get_env("MONGODB_DATABASE", "emetix")


class MongoDBClient:
    """
    MongoDB Atlas client for Emetix.
    
    Uses connection pooling for efficiency.
    Implements lazy connection - only connects when first operation is performed.
    
    Collections:
    - watchlists: User-created or system watchlists
    - strategies: Investment strategy templates
    - education: Educational content and tutorials
    - sessions: Temporary session data (24h TTL)
    """
    
    _instance: Optional['MongoDBClient'] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None
    
    def __new__(cls):
        """Singleton pattern for MongoDB client"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._connected = False
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self._connected and self._client is not None:
            return True
            
        if not MONGODB_URI:
            logger.warning("MONGODB_URI not configured - MongoDB features disabled")
            return False
        
        try:
            # Connect with SSL certificate verification
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
            
            self._db = self._client[MONGODB_DATABASE]
            self._connected = True
            logger.info(f"Connected to MongoDB Atlas: {MONGODB_DATABASE}")
            
            # Ensure indexes
            self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self._connected = False
            return False
    
    def _create_indexes(self):
        """Create necessary indexes for performance"""
        try:
            # Watchlists - index by session and created_at
            self._db.watchlists.create_index("session_id")
            self._db.watchlists.create_index("created_at")
            self._db.watchlists.create_index("is_public")
            
            # Education - index by category and order
            self._db.education.create_index("category")
            self._db.education.create_index("order")
            
            # Sessions - TTL index (auto-delete after 24 hours)
            self._db.sessions.create_index(
                "created_at",
                expireAfterSeconds=86400  # 24 hours
            )
            
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
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
    # WATCHLIST OPERATIONS
    # =========================================================================
    
    def save_watchlist(
        self,
        name: str,
        tickers: List[str],
        session_id: Optional[str] = None,
        is_public: bool = False,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save a watchlist to MongoDB.
        
        Args:
            name: Watchlist name
            tickers: List of stock tickers
            session_id: Optional session ID for private watchlists
            is_public: Whether watchlist is publicly accessible
            metadata: Additional metadata (risk level, strategy, etc.)
            
        Returns:
            Watchlist ID if saved successfully, None otherwise
        """
        if not self.connect():
            return None
        
        try:
            doc = {
                "name": name,
                "tickers": tickers,
                "session_id": session_id,
                "is_public": is_public,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "ticker_count": len(tickers)
            }
            
            result = self._db.watchlists.insert_one(doc)
            watchlist_id = str(result.inserted_id)
            logger.info(f"Saved watchlist '{name}' with {len(tickers)} tickers")
            return watchlist_id
            
        except Exception as e:
            logger.error(f"Failed to save watchlist: {e}")
            return None
    
    def get_watchlist(self, watchlist_id: str) -> Optional[Dict]:
        """Get a watchlist by ID"""
        if not self.connect():
            return None
        
        try:
            doc = self._db.watchlists.find_one({"_id": ObjectId(watchlist_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as e:
            logger.error(f"Failed to get watchlist: {e}")
            return None
    
    def get_public_watchlists(self, limit: int = 20) -> List[Dict]:
        """Get public watchlists"""
        if not self.connect():
            return []
        
        try:
            cursor = self._db.watchlists.find(
                {"is_public": True}
            ).sort("created_at", -1).limit(limit)
            
            watchlists = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                watchlists.append(doc)
            return watchlists
            
        except Exception as e:
            logger.error(f"Failed to get public watchlists: {e}")
            return []
    
    def get_session_watchlists(self, session_id: str) -> List[Dict]:
        """Get watchlists for a session"""
        if not self.connect():
            return []
        
        try:
            cursor = self._db.watchlists.find(
                {"session_id": session_id}
            ).sort("created_at", -1)
            
            watchlists = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                watchlists.append(doc)
            return watchlists
            
        except Exception as e:
            logger.error(f"Failed to get session watchlists: {e}")
            return []
    
    def delete_watchlist(self, watchlist_id: str, session_id: str) -> bool:
        """Delete a watchlist (only if owned by session)"""
        if not self.connect():
            return False
        
        try:
            result = self._db.watchlists.delete_one({
                "_id": ObjectId(watchlist_id),
                "session_id": session_id
            })
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete watchlist: {e}")
            return False
    
    # =========================================================================
    # EDUCATION CONTENT OPERATIONS
    # =========================================================================
    
    def get_education_content(
        self,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get educational content.
        
        Categories: 'fundamentals', 'valuation', 'risk', 'technical', 'strategies'
        """
        if not self.connect():
            return []
        
        try:
            query = {}
            if category:
                query["category"] = category
            
            cursor = self._db.education.find(query).sort("order", 1).limit(limit)
            
            content = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                content.append(doc)
            return content
            
        except Exception as e:
            logger.error(f"Failed to get education content: {e}")
            return []
    
    def add_education_content(
        self,
        title: str,
        category: str,
        content: str,
        order: int = 0,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Add educational content (admin only in production)"""
        if not self.connect():
            return None
        
        try:
            doc = {
                "title": title,
                "category": category,
                "content": content,
                "order": order,
                "metadata": metadata or {},
                "created_at": datetime.utcnow()
            }
            
            result = self._db.education.insert_one(doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to add education content: {e}")
            return None
    
    # =========================================================================
    # STRATEGY TEMPLATES
    # =========================================================================
    
    def get_strategies(self) -> List[Dict]:
        """Get investment strategy templates"""
        if not self.connect():
            return []
        
        try:
            cursor = self._db.strategies.find().sort("name", 1)
            
            strategies = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                strategies.append(doc)
            return strategies
            
        except Exception as e:
            logger.error(f"Failed to get strategies: {e}")
            return []
    
    def save_strategy(
        self,
        name: str,
        description: str,
        criteria: Dict,
        risk_level: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Save a strategy template"""
        if not self.connect():
            return None
        
        try:
            doc = {
                "name": name,
                "description": description,
                "criteria": criteria,
                "risk_level": risk_level,
                "metadata": metadata or {},
                "created_at": datetime.utcnow()
            }
            
            result = self._db.strategies.insert_one(doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to save strategy: {e}")
            return None
    
    # =========================================================================
    # SESSION MANAGEMENT (Ephemeral, no auth required)
    # =========================================================================
    
    def create_session(self) -> Optional[str]:
        """Create a new anonymous session"""
        if not self.connect():
            return None
        
        try:
            doc = {
                "created_at": datetime.utcnow(),
                "data": {}
            }
            result = self._db.sessions.insert_one(doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        if not self.connect():
            return None
        
        try:
            doc = self._db.sessions.find_one({"_id": ObjectId(session_id)})
            if doc:
                return doc.get("data", {})
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def update_session_data(self, session_id: str, data: Dict) -> bool:
        """Update session data"""
        if not self.connect():
            return False
        
        try:
            result = self._db.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"data": data}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("MongoDB connection closed")


# Singleton instance
mongo_client = MongoDBClient()


def get_mongo_db() -> Optional[Database]:
    """Get MongoDB database instance"""
    return mongo_client.db


def is_mongo_available() -> bool:
    """Check if MongoDB is available"""
    return mongo_client.connect()
