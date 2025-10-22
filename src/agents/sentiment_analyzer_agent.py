"""
Sentiment Analyzer Agent - Specialized for market sentiment analysis
Part of Multi-Agent Stock Analysis System
"""
from langchain.agents import create_react_agent, AgentExecutor
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain import hub
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config.settings import GROQ_API_KEY
from config.logging_config import get_logger

logger = get_logger(__name__)


class SentimentAnalyzerAgent:
    """
    Specialized agent for scanning and scoring market sentiment
    Uses Groq Mixtral-8x7B for nuanced text analysis
    
    Sentiment sources:
    - Financial news (NewsAPI, Yahoo Finance)
    - Social media signals (Twitter/X)
    - Analyst ratings and upgrades/downgrades
    """
    
    def __init__(self, api_key: str = GROQ_API_KEY):
        """
        Initialize Sentiment Analyzer Agent
        
        Args:
            api_key: Groq API key
        """
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        self.logger = logger
        # Use Llama 3.1 for better sentiment analysis
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup sentiment analysis tools"""
        
        def analyze_news_sentiment_tool(ticker: str) -> str:
            """Analyze sentiment from financial news (Yahoo Finance + optional NewsAPI)"""
            try:
                # Use enhanced news sentiment fetcher for multi-source analysis
                from src.data.fetchers.news_sentiment_fetcher import NewsSentimentFetcher
                
                fetcher = NewsSentimentFetcher()
                result = fetcher.fetch_all_news(ticker)
                
                # Format result for agent consumption
                summary = f"""
NEWS SENTIMENT ANALYSIS FOR {ticker}

Company: {result['company_name']}
Total Articles Analyzed: {result['total_articles']}
Sources: {', '.join(result['sources_used'])}

SENTIMENT SCORE: {result['sentiment_score']}/1.0
SENTIMENT LABEL: {result['sentiment_label']}
CONFIDENCE: {result['confidence']}

Distribution:
  - Positive: {result['positive_count']} articles
  - Negative: {result['negative_count']} articles
  - Neutral: {result['neutral_count']} articles

Top Recent Headlines:
"""
                for i, article in enumerate(result['articles'][:5], 1):
                    summary += f"\n{i}. [{article['sentiment']}] {article['title']}"
                    summary += f"\n   Source: {article['source']}"
                
                return summary
                
            except Exception as e:
                self.logger.error(f"Error analyzing news sentiment: {str(e)}")
                # Fallback to simple Yahoo Finance method
                return self._fallback_yahoo_news_sentiment(ticker)
        
        def analyze_social_sentiment_tool(ticker: str) -> str:
            """Analyze social media sentiment (simulated)"""
            try:
                # In production, this would integrate with Twitter API, Reddit API, etc.
                # For now, we'll use a placeholder that analyzes based on market data
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1mo")
                
                if hist.empty:
                    return f"No data available for social sentiment analysis of {ticker}"
                
                # Use price momentum as proxy for social sentiment
                month_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                volume_trend = hist['Volume'].tail(5).mean() / hist['Volume'].mean()
                
                # Heuristic sentiment scoring
                if month_return > 10 and volume_trend > 1.2:
                    sentiment_score = 0.8
                    label = "VERY BULLISH"
                    buzz = "HIGH"
                elif month_return > 5:
                    sentiment_score = 0.65
                    label = "BULLISH"
                    buzz = "MODERATE"
                elif month_return > -5:
                    sentiment_score = 0.5
                    label = "NEUTRAL"
                    buzz = "LOW"
                elif month_return > -10:
                    sentiment_score = 0.35
                    label = "BEARISH"
                    buzz = "MODERATE"
                else:
                    sentiment_score = 0.2
                    label = "VERY BEARISH"
                    buzz = "HIGH"
                
                data = {
                    'ticker': ticker,
                    'social_sentiment_score': sentiment_score,
                    'sentiment_label': label,
                    'buzz_level': buzz,
                    'month_return_pct': round(month_return, 2),
                    'volume_trend': round(volume_trend, 2),
                    'note': 'Based on price momentum and volume indicators (production version would use real social data)'
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error analyzing social sentiment for {ticker}: {str(e)}"
        
        def analyze_analyst_sentiment_tool(ticker: str) -> str:
            """Analyze analyst ratings and recommendations"""
            try:
                import yfinance as yf
                
                stock = yf.Ticker(ticker)
                recommendations = stock.recommendations
                
                if recommendations is None or recommendations.empty:
                    return f"No analyst recommendations available for {ticker}"
                
                # Get recent recommendations (last 3 months)
                recent_recs = recommendations.tail(20)
                
                # Count recommendation types
                rec_counts = recent_recs['To Grade'].value_counts()
                
                # Calculate sentiment score
                buy_keywords = ['buy', 'outperform', 'overweight', 'positive']
                hold_keywords = ['hold', 'neutral', 'equal']
                sell_keywords = ['sell', 'underperform', 'underweight', 'negative']
                
                buy_count = sum(rec_counts.get(grade, 0) for grade in rec_counts.index 
                              if any(kw in grade.lower() for kw in buy_keywords))
                hold_count = sum(rec_counts.get(grade, 0) for grade in rec_counts.index 
                               if any(kw in grade.lower() for kw in hold_keywords))
                sell_count = sum(rec_counts.get(grade, 0) for grade in rec_counts.index 
                               if any(kw in grade.lower() for kw in sell_keywords))
                
                total = buy_count + hold_count + sell_count
                
                if total > 0:
                    # Weighted score: Buy=1, Hold=0.5, Sell=0
                    sentiment_score = (buy_count + 0.5 * hold_count) / total
                else:
                    sentiment_score = 0.5
                
                data = {
                    'ticker': ticker,
                    'analyst_sentiment_score': round(sentiment_score, 2),
                    'buy_ratings': int(buy_count),
                    'hold_ratings': int(hold_count),
                    'sell_ratings': int(sell_count),
                    'total_ratings': int(total),
                    'sentiment_label': 'BULLISH' if sentiment_score > 0.6 else 'BEARISH' if sentiment_score < 0.4 else 'NEUTRAL',
                    'consensus': f"{buy_count}B/{hold_count}H/{sell_count}S"
                }
                
                return str(data)
                
            except Exception as e:
                return f"Error analyzing analyst sentiment for {ticker}: {str(e)}"
        
        def aggregate_sentiment_tool(ticker: str) -> str:
            """Aggregate sentiment from all sources"""
            try:
                # This tool would call other tools and aggregate
                # For simplicity, returning a summary format
                data = {
                    'ticker': ticker,
                    'note': 'Use AnalyzeNewsSentiment, AnalyzeSocialSentiment, and AnalyzeAnalystSentiment separately, then aggregate scores.'
                }
                return str(data)
                
            except Exception as e:
                return f"Error aggregating sentiment for {ticker}: {str(e)}"
        
        tools = [
            Tool(
                name="AnalyzeNewsSentiment",
                func=analyze_news_sentiment_tool,
                description="Analyze sentiment from financial news headlines and articles. Scores news as BULLISH, BEARISH, or NEUTRAL based on keyword analysis. Returns sentiment score (0-1) and article counts."
            ),
            Tool(
                name="AnalyzeSocialSentiment",
                func=analyze_social_sentiment_tool,
                description="Analyze social media sentiment and buzz level using price momentum and volume indicators. Returns sentiment score (0-1), buzz level, and trend indicators."
            ),
            Tool(
                name="AnalyzeAnalystSentiment",
                func=analyze_analyst_sentiment_tool,
                description="Analyze professional analyst ratings and recommendations. Returns buy/hold/sell counts and consensus sentiment score (0-1)."
            ),
            Tool(
                name="AggregateSentiment",
                func=aggregate_sentiment_tool,
                description="Aggregate sentiment scores from all sources (news, social, analysts) into a comprehensive sentiment assessment."
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent executor"""
        try:
            prompt = hub.pull("hwchase17/react")
            
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=8  # Balanced: enough for news+social+analyst, but won't hit rate limits
            )
            
            return agent_executor
            
        except Exception as e:
            self.logger.error(f"Error creating Sentiment Analyzer Agent: {str(e)}")
            raise
    
    def analyze_comprehensive_sentiment(self, ticker: str) -> Dict:
        """
        Comprehensive sentiment analysis from all sources
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Aggregated sentiment assessment
        """
        try:
            query = f"Provide comprehensive sentiment analysis for {ticker} by analyzing news sentiment, social sentiment, and analyst ratings. Calculate an aggregated sentiment score and explain the market psychology."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'ticker': ticker,
                'sentiment_analysis': result['output'],
                'agent': 'SentimentAnalyzerAgent',
                'model': 'mixtral-8x7b-32768'
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive sentiment analysis: {str(e)}")
            return {'error': str(e)}
    
    def detect_contrarian_signals(self, ticker: str, valuation_data: Dict) -> Dict:
        """
        Detect contrarian opportunities (negative sentiment + strong fundamentals)
        
        Args:
            ticker: Stock ticker symbol
            valuation_data: Fundamental valuation data
            
        Returns:
            Contrarian signal assessment
        """
        try:
            val_score = valuation_data.get('valuation_score', 50)
            
            query = f"Analyze sentiment for {ticker} to detect contrarian opportunities. The stock has a valuation score of {val_score}/100. Look for cases where negative sentiment suppresses a fundamentally strong stock (contrarian buy signal) or positive sentiment inflates an overvalued stock (contrarian sell signal)."
            
            result = self.agent_executor.invoke({"input": query})
            
            return {
                'ticker': ticker,
                'contrarian_analysis': result['output'],
                'agent': 'SentimentAnalyzerAgent'
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting contrarian signals: {str(e)}")
            return {'error': str(e)}
    
    def _fallback_yahoo_news_sentiment(self, ticker: str) -> str:
        """
        Fallback method using simple Yahoo Finance news sentiment
        Used when enhanced fetcher fails
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return f"No recent news found for {ticker}"
            
            keywords_positive = ['beat', 'surge', 'rally', 'upgrade', 'strong', 'growth', 'profit', 'gain', 'bullish', 'outperform']
            keywords_negative = ['miss', 'drop', 'fall', 'downgrade', 'weak', 'loss', 'decline', 'bearish', 'underperform', 'concern']
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for item in news[:10]:
                title = item.get('title', '').lower()
                pos_score = sum(1 for kw in keywords_positive if kw in title)
                neg_score = sum(1 for kw in keywords_negative if kw in title)
                
                if pos_score > neg_score:
                    positive_count += 1
                elif neg_score > pos_score:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(news[:10])
            sentiment_score = (positive_count - negative_count) / total if total > 0 else 0
            normalized_score = (sentiment_score + 1) / 2
            
            return f"""
FALLBACK NEWS SENTIMENT (Yahoo Finance only)
Ticker: {ticker}
Articles: {total}
Sentiment Score: {round(normalized_score, 2)}
Positive: {positive_count} | Negative: {negative_count} | Neutral: {neutral_count}
"""
        except Exception as e:
            return f"Error in fallback news sentiment: {str(e)}"


# Example usage
if __name__ == "__main__":
    print("=== Sentiment Analyzer Agent Test ===\n")
    
    # Initialize agent
    agent = SentimentAnalyzerAgent()
    
    # Test sentiment analysis
    print("Analyzing sentiment for TSLA...")
    result = agent.analyze_comprehensive_sentiment("TSLA")
    print(f"\nSentiment Analysis:\n{result['sentiment_analysis']}")
    
    # Test contrarian detection
    print("\n\nDetecting contrarian signals...")
    contrarian = agent.detect_contrarian_signals("TSLA", {'valuation_score': 75})
    print(f"\nContrarian Analysis:\n{contrarian['contrarian_analysis']}")
