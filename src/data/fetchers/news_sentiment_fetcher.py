"""
Enhanced News Sentiment Fetcher with Smart Fallback System
Supports multiple data sources with tiered priority and deduplication:
- Tier 1: Yahoo Finance (FREE, unlimited) + NewsAPI (100/day)
- Tier 2: Finnhub (60/min fallback when NewsAPI exhausted)
- Tier 3: Google News RSS (FREE, supplementary coverage)
"""
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from difflib import SequenceMatcher
from config.settings import NEWS_API_KEY, FINNHUB_API_KEY
from config.logging_config import get_logger

logger = get_logger(__name__)


class NewsSentimentFetcher:
    """
    Fetch and analyze news sentiment from multiple sources with smart fallback
    
    Data Source Strategy (Tiered):
    - Tier 1 (Primary): Yahoo Finance (FREE) + NewsAPI (100/day)
    - Tier 2 (Fallback): Finnhub (60/min) when NewsAPI hits rate limit
    - Tier 3 (Supplementary): Google News RSS (FREE) for additional coverage
    
    Features:
    - Automatic fallback when APIs hit rate limits
    - Deduplication to avoid redundant articles
    - Clean, reliable sentiment analysis
    """
    
    def __init__(self, news_api_key: Optional[str] = None, finnhub_api_key: Optional[str] = None):
        """
        Initialize news sentiment fetcher
        
        Args:
            news_api_key: Optional NewsAPI key (uses env var if not provided)
            finnhub_api_key: Optional Finnhub key (uses env var if not provided)
        """
        self.news_api_key = news_api_key or NEWS_API_KEY
        self.finnhub_api_key = finnhub_api_key or FINNHUB_API_KEY
        self.logger = logger
        
        # Track API status for smart fallback
        self.api_status = {
            'newsapi_available': bool(self.news_api_key and self.news_api_key != 'your_news_api_key_here'),
            'finnhub_available': bool(self.finnhub_api_key and self.finnhub_api_key != 'your_finnhub_key_here'),
            'newsapi_exhausted': False,
            'finnhub_exhausted': False
        }
        
        # Sentiment keywords for simple analysis
        self.keywords_positive = [
            'beat', 'surge', 'rally', 'upgrade', 'strong', 'growth', 'profit', 
            'gain', 'bullish', 'outperform', 'record', 'soar', 'boom', 'accelerate',
            'innovation', 'breakthrough', 'success', 'buy', 'target raised'
        ]
        
        self.keywords_negative = [
            'miss', 'drop', 'fall', 'downgrade', 'weak', 'loss', 'decline', 
            'bearish', 'underperform', 'concern', 'plunge', 'crash', 'warning',
            'lawsuit', 'investigation', 'fraud', 'sell', 'target cut', 'disappoint'
        ]
    
    def fetch_yahoo_finance_news(self, ticker: str, limit: int = 20) -> List[Dict]:
        """
        Fetch news from Yahoo Finance (FREE, no API key needed)
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles
            
        Returns:
            List of news articles with sentiment
        """
        try:
            from datetime import timezone
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                self.logger.warning(f"No Yahoo Finance news found for {ticker}")
                return []
            
            articles = []
            for item in news[:limit]:
                title = item.get('title', '')
                
                # Calculate sentiment score
                sentiment = self._calculate_sentiment(title)
                
                # Handle timestamp with timezone awareness
                timestamp = item.get('providerPublishTime', 0)
                publish_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                
                articles.append({
                    'source': 'Yahoo Finance',
                    'title': title,
                    'link': item.get('link', ''),
                    'publisher': item.get('publisher', 'Unknown'),
                    'publish_time': publish_time,
                    'sentiment': sentiment['label'],
                    'sentiment_score': sentiment['score']
                })
            
            self.logger.info(f"Fetched {len(articles)} articles from Yahoo Finance for {ticker}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance news for {ticker}: {str(e)}")
            return []
    
    def fetch_newsapi_news(self, ticker: str, company_name: str, days: int = 7) -> Tuple[List[Dict], bool]:
        """
        Fetch news from NewsAPI (Tier 1 - Primary premium source)
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name for better search
            days: Number of days to look back
            
        Returns:
            Tuple of (articles list, rate_limit_hit boolean)
        """
        if not self.api_status['newsapi_available'] or self.api_status['newsapi_exhausted']:
            self.logger.info("NewsAPI not available or exhausted, skipping")
            return [], False
        
        try:
            url = 'https://newsapi.org/v2/everything'
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': f'{ticker} OR {company_name}',
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # Check for rate limiting
            if response.status_code == 429:
                self.logger.warning("NewsAPI rate limit hit, marking as exhausted")
                self.api_status['newsapi_exhausted'] = True
                return [], True
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                error_msg = data.get('message', 'Unknown error')
                if 'rate limit' in error_msg.lower() or 'quota' in error_msg.lower():
                    self.api_status['newsapi_exhausted'] = True
                    self.logger.warning(f"NewsAPI quota exhausted: {error_msg}")
                    return [], True
                self.logger.error(f"NewsAPI error: {error_msg}")
                return [], False
            
            articles = []
            for item in data.get('articles', [])[:20]:
                title = item.get('title', '')
                description = item.get('description', '')
                
                # Calculate sentiment from title and description
                sentiment = self._calculate_sentiment(f"{title} {description}")
                
                articles.append({
                    'source': 'NewsAPI',
                    'title': title,
                    'description': description,
                    'link': item.get('url', ''),
                    'publisher': item.get('source', {}).get('name', 'Unknown'),
                    'publish_time': datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00')),
                    'sentiment': sentiment['label'],
                    'sentiment_score': sentiment['score']
                })
            
            self.logger.info(f"✅ Fetched {len(articles)} articles from NewsAPI for {ticker}")
            return articles, False
            
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI news for {ticker}: {str(e)}")
            return [], False
    
    def fetch_google_news_rss(self, ticker: str, company_name: str) -> List[Dict]:
        """
        Fetch news from Google News RSS feed (Tier 3 - Supplementary, FREE)
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name for better search
            
        Returns:
            List of news articles with sentiment
        """
        try:
            import feedparser
            from urllib.parse import quote_plus
            
            # Google News RSS feed URL with proper encoding
            query = f"{ticker} {company_name} stock"
            encoded_query = quote_plus(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries[:15]:
                title = entry.get('title', '')
                
                # Calculate sentiment
                sentiment = self._calculate_sentiment(title)
                
                # Handle publish time with timezone awareness
                try:
                    publish_time = datetime(*entry.get('published_parsed', datetime.now().timetuple())[:6])
                    # Make timezone-aware (UTC)
                    from datetime import timezone
                    publish_time = publish_time.replace(tzinfo=timezone.utc)
                except:
                    publish_time = datetime.now(timezone.utc)
                
                articles.append({
                    'source': 'Google News',
                    'title': title,
                    'link': entry.get('link', ''),
                    'publisher': entry.get('source', {}).get('title', 'Unknown'),
                    'publish_time': publish_time,
                    'sentiment': sentiment['label'],
                    'sentiment_score': sentiment['score']
                })
            
            self.logger.info(f"✅ Fetched {len(articles)} articles from Google News for {ticker}")
            return articles
            
        except ImportError:
            self.logger.warning("feedparser not installed. Install with: pip install feedparser")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching Google News for {ticker}: {str(e)}")
            return []
    
    def fetch_finnhub_news(self, ticker: str, days: int = 7) -> Tuple[List[Dict], bool]:
        """
        Fetch news from Finnhub (Tier 2 - Fallback when NewsAPI exhausted)
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            Tuple of (articles list, rate_limit_hit boolean)
        """
        if not self.api_status['finnhub_available'] or self.api_status['finnhub_exhausted']:
            self.logger.info("Finnhub not available or exhausted, skipping")
            return [], False
        
        try:
            url = f'https://finnhub.io/api/v1/company-news'
            
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                'symbol': ticker,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # Check for rate limiting
            if response.status_code == 429:
                self.logger.warning("Finnhub rate limit hit, marking as exhausted")
                self.api_status['finnhub_exhausted'] = True
                return [], True
            
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and data.get('error'):
                error_msg = data.get('error', 'Unknown error')
                if 'limit' in error_msg.lower() or 'quota' in error_msg.lower():
                    self.api_status['finnhub_exhausted'] = True
                    self.logger.warning(f"Finnhub quota exhausted: {error_msg}")
                    return [], True
                self.logger.error(f"Finnhub error: {error_msg}")
                return [], False
            
            articles = []
            for item in data[:20]:
                headline = item.get('headline', '')
                summary = item.get('summary', '')
                
                # Calculate sentiment from headline and summary
                sentiment = self._calculate_sentiment(f"{headline} {summary}")
                
                # Handle timestamp with timezone awareness
                from datetime import timezone
                timestamp = item.get('datetime', 0)
                publish_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                
                articles.append({
                    'source': 'Finnhub',
                    'title': headline,
                    'description': summary,
                    'link': item.get('url', ''),
                    'publisher': item.get('source', 'Unknown'),
                    'publish_time': publish_time,
                    'sentiment': sentiment['label'],
                    'sentiment_score': sentiment['score']
                })
            
            self.logger.info(f"✅ Fetched {len(articles)} articles from Finnhub for {ticker}")
            return articles, False
            
        except Exception as e:
            self.logger.error(f"Error fetching Finnhub news for {ticker}: {str(e)}")
            return [], False
    
    def fetch_all_news(self, ticker: str, company_name: Optional[str] = None) -> Dict:
        """
        Fetch news from ALL available sources with smart fallback and deduplication
        
        Strategy:
        - Tier 1: Yahoo Finance (always) + NewsAPI (if available)
        - Tier 2: Finnhub (fallback if NewsAPI hits rate limit)
        - Tier 3: Google News (supplementary for broader coverage)
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name (optional, fetched from yfinance if not provided)
            
        Returns:
            Aggregated news sentiment data with deduplication
        """
        # Get company name if not provided
        if not company_name:
            try:
                stock = yf.Ticker(ticker)
                company_name = stock.info.get('longName', ticker)
            except:
                company_name = ticker
        
        all_articles = []
        sources_used = []
        fallback_triggered = False
        
        # Tier 1: Yahoo Finance (always fetch, FREE and unlimited)
        self.logger.info(f"📰 Fetching news for {ticker} ({company_name})...")
        yahoo_news = self.fetch_yahoo_finance_news(ticker, limit=20)
        if yahoo_news:
            all_articles.extend(yahoo_news)
            sources_used.append('Yahoo Finance')
        
        # Tier 1: NewsAPI (primary premium source, 100/day)
        newsapi_articles, newsapi_rate_limited = self.fetch_newsapi_news(ticker, company_name, days=7)
        if newsapi_articles:
            all_articles.extend(newsapi_articles)
            sources_used.append('NewsAPI')
        
        # Tier 2: Finnhub fallback (if NewsAPI hit rate limit)
        if newsapi_rate_limited or (self.api_status['newsapi_exhausted'] and not newsapi_articles):
            fallback_triggered = True
            self.logger.warning("⚠️  NewsAPI exhausted, using Finnhub as fallback...")
            finnhub_articles, _ = self.fetch_finnhub_news(ticker, days=7)
            if finnhub_articles:
                all_articles.extend(finnhub_articles)
                sources_used.append('Finnhub (fallback)')
        
        # Tier 3: Google News RSS (supplementary, FREE)
        google_news = self.fetch_google_news_rss(ticker, company_name)
        if google_news:
            all_articles.extend(google_news)
            sources_used.append('Google News')
        
        # Deduplicate articles by title similarity
        unique_articles = self._deduplicate_articles(all_articles)
        
        if not unique_articles:
            return {
                'ticker': ticker,
                'company_name': company_name,
                'total_articles': 0,
                'unique_articles': 0,
                'duplicates_removed': 0,
                'sentiment_score': 0.5,
                'sentiment_label': 'NEUTRAL',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sources_used': [],
                'fallback_triggered': fallback_triggered,
                'articles': []
            }
        
        # Calculate aggregate sentiment
        positive_count = sum(1 for a in unique_articles if a['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for a in unique_articles if a['sentiment'] == 'NEGATIVE')
        neutral_count = sum(1 for a in unique_articles if a['sentiment'] == 'NEUTRAL')
        
        # Weighted average sentiment score
        avg_sentiment = sum(a['sentiment_score'] for a in unique_articles) / len(unique_articles)
        
        # Determine overall label
        if avg_sentiment > 0.6:
            label = 'BULLISH'
        elif avg_sentiment < 0.4:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'total_articles': len(all_articles),
            'unique_articles': len(unique_articles),
            'duplicates_removed': len(all_articles) - len(unique_articles),
            'sentiment_score': round(avg_sentiment, 2),
            'sentiment_label': label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sources_used': sources_used,
            'fallback_triggered': fallback_triggered,
            'articles': sorted(unique_articles, key=lambda x: x['publish_time'], reverse=True)[:10],  # Top 10 most recent
            'confidence': self._calculate_confidence(len(unique_articles), positive_count, negative_count, neutral_count)
        }
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Remove duplicate articles based on title similarity
        
        Args:
            articles: List of articles to deduplicate
            
        Returns:
            List of unique articles
        """
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = []
        
        for article in articles:
            title = article['title'].lower().strip()
            
            # Check if similar title already exists
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = SequenceMatcher(None, title, seen_title).ratio()
                if similarity > 0.85:  # 85% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.append(title)
        
        duplicates_removed = len(articles) - len(unique_articles)
        if duplicates_removed > 0:
            self.logger.info(f"🔄 Removed {duplicates_removed} duplicate articles")
        
        return unique_articles
    
    def _calculate_sentiment(self, text: str) -> Dict[str, any]:
        """
        Calculate sentiment from text using keyword matching
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label and score
        """
        text_lower = text.lower()
        
        # Count positive and negative keywords
        pos_score = sum(1 for kw in self.keywords_positive if kw in text_lower)
        neg_score = sum(1 for kw in self.keywords_negative if kw in text_lower)
        
        # Calculate normalized score (0-1 scale)
        total_keywords = pos_score + neg_score
        if total_keywords == 0:
            score = 0.5  # Neutral
            label = 'NEUTRAL'
        else:
            raw_score = (pos_score - neg_score) / total_keywords
            score = (raw_score + 1) / 2  # Normalize to 0-1
            
            if score > 0.6:
                label = 'POSITIVE'
            elif score < 0.4:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
        
        return {'score': round(score, 2), 'label': label}
    
    def _calculate_confidence(self, total: int, pos: int, neg: int, neu: int) -> str:
        """
        Calculate confidence level based on article count and distribution
        
        Args:
            total: Total number of articles
            pos: Positive article count
            neg: Negative article count
            neu: Neutral article count
            
        Returns:
            Confidence level (HIGH, MEDIUM, LOW)
        """
        if total >= 20:
            # Check if sentiment is clear (not too mixed)
            max_count = max(pos, neg, neu)
            if max_count / total > 0.6:
                return 'HIGH'
            return 'MEDIUM'
        elif total >= 10:
            return 'MEDIUM'
        else:
            return 'LOW'


# Convenience function for quick sentiment check
def get_news_sentiment(ticker: str, company_name: Optional[str] = None) -> Dict:
    """
    Quick function to get aggregated news sentiment
    
    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name
        
    Returns:
        Aggregated sentiment data
    """
    fetcher = NewsSentimentFetcher()
    return fetcher.fetch_all_news(ticker, company_name)


if __name__ == "__main__":
    # Test the news sentiment fetcher
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "AAPL"
    
    print(f"\n{'='*80}")
    print(f"TESTING SMART NEWS SENTIMENT FETCHER - {ticker}")
    print(f"{'='*80}\n")
    
    result = get_news_sentiment(ticker)
    
    print(f"Company: {result['company_name']}")
    print(f"Total Articles: {result['total_articles']}")
    print(f"Unique Articles: {result['unique_articles']}")
    print(f"Duplicates Removed: {result['duplicates_removed']}")
    print(f"Sentiment Score: {result['sentiment_score']}/1.0")
    print(f"Sentiment Label: {result['sentiment_label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources Used: {', '.join(result['sources_used'])}")
    if result['fallback_triggered']:
        print(f"⚠️  Fallback Triggered: NewsAPI exhausted, used Finnhub")
    
    print(f"\n📊 Breakdown:")
    print(f"  Positive: {result['positive_count']} ({result['positive_count']/max(result['unique_articles'], 1)*100:.1f}%)")
    print(f"  Negative: {result['negative_count']} ({result['negative_count']/max(result['unique_articles'], 1)*100:.1f}%)")
    print(f"  Neutral: {result['neutral_count']} ({result['neutral_count']/max(result['unique_articles'], 1)*100:.1f}%)")
    
    print(f"\n📰 Top 5 Recent Headlines:")
    for i, article in enumerate(result['articles'][:5], 1):
        print(f"{i}. [{article['sentiment']}] {article['title'][:80]}")
        print(f"   Source: {article['source']} | Publisher: {article['publisher']}")
        print(f"   Time: {article['publish_time'].strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\n{'='*80}")
    print(f"✅ Test Complete - System Status:")
    print(f"   Yahoo Finance: ✅ Always available (FREE)")
    print(f"   NewsAPI: {'✅ Configured' if result.get('sources_used') and 'NewsAPI' in result['sources_used'] else '⚠️  Not configured or exhausted'}")
    print(f"   Finnhub: {'✅ Configured' if result.get('sources_used') and 'Finnhub' in result['sources_used'] else '⚠️  Not configured or not needed'}")
    print(f"   Google News: {'✅ Available' if result.get('sources_used') and 'Google News' in result['sources_used'] else '⚠️  feedparser not installed'}")
    print(f"{'='*80}\n")
