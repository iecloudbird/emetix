"""
Test News Sentiment Fetcher
Quick test to verify multi-source news sentiment analysis
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.fetchers.news_sentiment_fetcher import NewsSentimentFetcher, get_news_sentiment


def test_news_sources(ticker: str = "AAPL"):
    """Test individual news sources"""
    print(f"\n{'='*80}")
    print(f"TESTING NEWS SOURCES FOR {ticker}")
    print(f"{'='*80}\n")
    
    fetcher = NewsSentimentFetcher()
    
    # Test Yahoo Finance
    print("1. Testing Yahoo Finance News (FREE)...")
    yahoo_news = fetcher.fetch_yahoo_finance_news(ticker, limit=10)
    print(f"   ✅ Found {len(yahoo_news)} articles from Yahoo Finance")
    
    # Test Google News
    print("\n2. Testing Google News RSS (FREE)...")
    try:
        google_news = fetcher.fetch_google_news_rss(ticker, "Apple Inc")
        print(f"   ✅ Found {len(google_news)} articles from Google News")
    except Exception as e:
        print(f"   ⚠️  Google News failed: {str(e)}")
        print(f"   💡 Install feedparser: pip install feedparser")
    
    # Test NewsAPI
    print("\n3. Testing NewsAPI (requires API key)...")
    newsapi_news = fetcher.fetch_newsapi_news(ticker, "Apple Inc")
    if newsapi_news:
        print(f"   ✅ Found {len(newsapi_news)} articles from NewsAPI")
    else:
        print(f"   ⚠️  NewsAPI not configured or rate limited")
        print(f"   💡 Get free API key from https://newsapi.org/register")


def test_aggregated_sentiment(ticker: str = "AAPL"):
    """Test aggregated multi-source sentiment"""
    print(f"\n{'='*80}")
    print(f"TESTING AGGREGATED SENTIMENT FOR {ticker}")
    print(f"{'='*80}\n")
    
    result = get_news_sentiment(ticker)
    
    print(f"📊 SENTIMENT ANALYSIS RESULTS")
    print(f"{'─'*80}")
    print(f"Company:           {result['company_name']}")
    print(f"Total Articles:    {result['total_articles']}")
    print(f"Sentiment Score:   {result['sentiment_score']}/1.0")
    print(f"Sentiment Label:   {result['sentiment_label']}")
    print(f"Confidence:        {result['confidence']}")
    print(f"Sources Used:      {', '.join(result['sources_used'])}")
    print(f"\n📈 Breakdown:")
    print(f"   Positive:  {result['positive_count']} ({result['positive_count']/result['total_articles']*100:.1f}%)")
    print(f"   Negative:  {result['negative_count']} ({result['negative_count']/result['total_articles']*100:.1f}%)")
    print(f"   Neutral:   {result['neutral_count']} ({result['neutral_count']/result['total_articles']*100:.1f}%)")
    
    if result['articles']:
        print(f"\n📰 Top 5 Recent Headlines:")
        print(f"{'─'*80}")
        for i, article in enumerate(result['articles'][:5], 1):
            sentiment_emoji = {'POSITIVE': '🟢', 'NEGATIVE': '🔴', 'NEUTRAL': '⚪'}
            emoji = sentiment_emoji.get(article['sentiment'], '⚪')
            print(f"\n{i}. {emoji} [{article['sentiment']}] {article['title']}")
            print(f"   📅 {article['publish_time'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   📰 {article['source']} | {article['publisher']}")


def main():
    print("\n" + "="*80)
    print("NEWS SENTIMENT FETCHER TEST SUITE")
    print("="*80)
    
    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    # Test individual sources
    test_news_sources(ticker)
    
    # Test aggregated sentiment
    test_aggregated_sentiment(ticker)
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    print("\n💡 Tips:")
    print("  - Install feedparser for Google News: pip install feedparser")
    print("  - Add NEWS_API_KEY to .env for NewsAPI access")
    print("  - Test other stocks: python scripts/test_news_sentiment.py MSFT")
    print()


if __name__ == "__main__":
    main()
