/**
 * MultiAgentAnalysisPanel - Deep AI analysis using multi-agent system
 *
 * Uses specialized AI agents for comprehensive insights:
 * - SentimentAnalyzerAgent: News & market sentiment
 * - FundamentalsAnalyzerAgent: Financial metrics & quality
 * - EnhancedValuationAgent: LSTM-DCF ML valuation
 * - SupervisorAgent: Orchestrated deep analysis
 */
"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  TrendingUp,
  TrendingDown,
  Target,
  Brain,
  Newspaper,
  BarChart3,
  LineChart,
  Loader2,
  Lightbulb,
  Zap,
  RefreshCw,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  fetchMultiAgentAnalysis,
  type MultiAgentAnalysisResponse,
  type MultiAgentSentiment,
  type MultiAgentFundamentals,
  type MultiAgentMLValuation,
  type MultiAgentSynthesis,
} from "@/lib/api";

interface MultiAgentAnalysisPanelProps {
  ticker: string;
  className?: string;
}

// =============================================================================
// MARKDOWN PARSING UTILITIES
// =============================================================================

/**
 * Safely extract text from analysis content which may be string or object
 */
function extractTextContent(content: unknown): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (typeof content === "object" && content !== null) {
    // Handle {type, text, extras} structure from LangChain
    if (
      "text" in content &&
      typeof (content as Record<string, unknown>).text === "string"
    ) {
      return (content as Record<string, unknown>).text as string;
    }
    // Try to stringify if it's a different object structure
    try {
      return JSON.stringify(content, null, 2);
    } catch {
      return String(content);
    }
  }
  return String(content);
}

/**
 * Parse inline markdown elements (bold, italic, code)
 */
function parseInlineMarkdown(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Match **bold**, *italic*, or `code`
    const match = remaining.match(/(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/);

    if (!match || match.index === undefined) {
      parts.push(remaining);
      break;
    }

    // Add text before the match
    if (match.index > 0) {
      parts.push(remaining.slice(0, match.index));
    }

    // Handle matched markdown
    if (match[2]) {
      // **bold** - Enhanced visibility
      parts.push(
        <strong key={key++} className="font-bold text-foreground">
          {match[2]}
        </strong>,
      );
    } else if (match[3]) {
      // *italic*
      parts.push(
        <em key={key++} className="italic text-foreground/90">
          {match[3]}
        </em>,
      );
    } else if (match[4]) {
      // `code` - Better contrast
      parts.push(
        <code
          key={key++}
          className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 text-sm font-mono text-slate-800 dark:text-slate-200"
        >
          {match[4]}
        </code>,
      );
    }

    remaining = remaining.slice(match.index + match[0].length);
  }

  return parts.length > 0 ? parts : [text];
}

/**
 * Render markdown content with proper formatting
 */
function MarkdownContent({ content }: { content: string }) {
  if (!content) {
    return <span className="text-muted-foreground">No content available.</span>;
  }

  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Skip empty lines but add spacing
    if (!trimmed) {
      elements.push(<div key={key++} className="h-2" />);
      continue;
    }

    // Headers ## - Key section headers
    if (trimmed.startsWith("## ")) {
      elements.push(
        <h3
          key={key++}
          className="font-bold text-base text-slate-900 dark:text-slate-100 mt-4 mb-2 border-b border-slate-200 dark:border-slate-700 pb-1"
        >
          {parseInlineMarkdown(trimmed.slice(3))}
        </h3>,
      );
      continue;
    }

    // Headers ### - Sub-section headers
    if (trimmed.startsWith("### ")) {
      elements.push(
        <h4
          key={key++}
          className="font-semibold text-sm text-slate-800 dark:text-slate-200 mt-3 mb-1.5"
        >
          {parseInlineMarkdown(trimmed.slice(4))}
        </h4>,
      );
      continue;
    }

    // Bullet points - More readable
    if (
      trimmed.startsWith("- ") ||
      trimmed.startsWith("* ") ||
      trimmed.startsWith("• ")
    ) {
      elements.push(
        <div key={key++} className="flex items-start gap-2.5 ml-3 my-1">
          <span className="text-blue-600 dark:text-blue-400 mt-1 text-sm">
            •
          </span>
          <span className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
            {parseInlineMarkdown(trimmed.slice(2))}
          </span>
        </div>,
      );
      continue;
    }

    // Numbered list - Better visibility
    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (numberedMatch) {
      elements.push(
        <div key={key++} className="flex items-start gap-2.5 ml-3 my-1">
          <span className="text-blue-600 dark:text-blue-400 font-semibold text-sm min-w-[1.25rem]">
            {numberedMatch[1]}.
          </span>
          <span className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
            {parseInlineMarkdown(numberedMatch[2])}
          </span>
        </div>,
      );
      continue;
    }

    // Regular paragraph - Improved readability
    elements.push(
      <p
        key={key++}
        className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed my-1.5"
      >
        {parseInlineMarkdown(trimmed)}
      </p>,
    );
  }

  return <div className="space-y-0">{elements}</div>;
}

// Custom Gemini-style sparkle icon
const GeminiSparkle = ({ className }: { className?: string }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    className={className}
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs>
      <linearGradient id="gemini-grad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="#4285f4" />
        <stop offset="50%" stopColor="#9b72cb" />
        <stop offset="100%" stopColor="#d96570" />
      </linearGradient>
    </defs>
    <path
      d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
      fill="url(#gemini-grad)"
    />
  </svg>
);

// =============================================================================
// LOCAL STORAGE CACHE UTILITIES
// =============================================================================

const CACHE_PREFIX = "emetix_multiagent_";
const CACHE_DURATION_MS = 60 * 60 * 1000; // 1 hour cache duration

interface CachedAnalysis {
  data: MultiAgentAnalysisResponse;
  timestamp: number;
  deepAnalysis: boolean;
}

function getCacheKey(ticker: string, deepAnalysis: boolean): string {
  return `${CACHE_PREFIX}${ticker.toUpperCase()}_deep=${deepAnalysis}`;
}

function getCachedAnalysis(
  ticker: string,
  deepAnalysis: boolean,
): MultiAgentAnalysisResponse | null {
  if (typeof window === "undefined") return null;

  try {
    const key = getCacheKey(ticker, deepAnalysis);
    const cached = localStorage.getItem(key);
    if (!cached) return null;

    const parsed: CachedAnalysis = JSON.parse(cached);
    const age = Date.now() - parsed.timestamp;

    // Check if cache is still valid
    if (age > CACHE_DURATION_MS) {
      localStorage.removeItem(key);
      return null;
    }

    return parsed.data;
  } catch (e) {
    console.warn("Failed to read cache:", e);
    return null;
  }
}

function setCachedAnalysis(
  ticker: string,
  deepAnalysis: boolean,
  data: MultiAgentAnalysisResponse,
): void {
  if (typeof window === "undefined") return;

  try {
    const key = getCacheKey(ticker, deepAnalysis);
    const cacheEntry: CachedAnalysis = {
      data,
      timestamp: Date.now(),
      deepAnalysis,
    };
    localStorage.setItem(key, JSON.stringify(cacheEntry));
  } catch (e) {
    console.warn("Failed to write cache:", e);
  }
}

function clearAnalysisCache(ticker?: string): void {
  if (typeof window === "undefined") return;

  try {
    if (ticker) {
      // Clear specific ticker cache
      localStorage.removeItem(getCacheKey(ticker, false));
      localStorage.removeItem(getCacheKey(ticker, true));
    } else {
      // Clear all multi-agent cache
      const keys = Object.keys(localStorage).filter((k) =>
        k.startsWith(CACHE_PREFIX),
      );
      keys.forEach((k) => localStorage.removeItem(k));
    }
  } catch (e) {
    console.warn("Failed to clear cache:", e);
  }
}

export function MultiAgentAnalysisPanel({
  ticker,
  className,
}: MultiAgentAnalysisPanelProps) {
  const [activeTab, setActiveTab] = useState<
    "synthesis" | "sentiment" | "fundamentals" | "ml"
  >("synthesis");
  const [deepAnalysis, setDeepAnalysis] = useState(false);
  const [cachedData, setCachedData] =
    useState<MultiAgentAnalysisResponse | null>(null);

  // Load cached data on mount and when ticker/deepAnalysis changes
  React.useEffect(() => {
    const cached = getCachedAnalysis(ticker, deepAnalysis);
    if (cached) {
      setCachedData(cached);
    }
  }, [ticker, deepAnalysis]);

  // Multi-agent analysis query - always use deepSynthesis for this panel
  const {
    data: fetchedData,
    isLoading,
    isFetching,
    refetch,
  } = useQuery<MultiAgentAnalysisResponse>({
    queryKey: ["multiagent-analysis", ticker, deepAnalysis],
    queryFn: async () => {
      const result = await fetchMultiAgentAnalysis(ticker, {
        includeSentiment: true,
        includeFundamentals: true,
        includeMLValuation: true,
        deepAnalysis: deepAnalysis,
        deepSynthesis: true, // Always use comprehensive diagnostic for Deep Analysis tab
      });
      // Cache the result
      setCachedAnalysis(ticker, deepAnalysis, result);
      setCachedData(result);
      return result;
    },
    enabled: false, // Only fetch when user requests
    staleTime: 30 * 60 * 1000, // 30 minutes
  });

  // Use cached data if available, otherwise use fetched data
  const data = fetchedData || cachedData;

  const handleStartAnalysis = () => {
    refetch();
  };

  const handleDeepAnalysis = () => {
    setDeepAnalysis(true);
    refetch();
  };

  const handleClearCache = () => {
    clearAnalysisCache(ticker);
    setCachedData(null);
  };

  // Show CTA if no data yet
  if (!data && !isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-600" />
            Multi-Agent Deep Analysis
          </CardTitle>
          <CardDescription>
            Get comprehensive insights from specialized AI agents
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="p-6 rounded-lg border border-dashed bg-gradient-to-r from-purple-50/50 via-blue-50/50 to-pink-50/50 dark:from-purple-950/20 dark:via-blue-950/20 dark:to-pink-950/20">
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="flex items-center gap-2">
                <GeminiSparkle className="h-8 w-8" />
                <span className="text-lg font-semibold bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 bg-clip-text text-transparent">
                  Multi-Agent System
                </span>
              </div>
              <p className="text-sm text-muted-foreground max-w-md">
                Leverage our specialized AI agents for sentiment analysis,
                fundamental screening, and ML-powered valuation.
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                <Badge variant="outline" className="gap-1">
                  <Newspaper className="h-3 w-3" />
                  Sentiment
                </Badge>
                <Badge variant="outline" className="gap-1">
                  <BarChart3 className="h-3 w-3" />
                  Fundamentals
                </Badge>
                <Badge variant="outline" className="gap-1">
                  <LineChart className="h-3 w-3" />
                  LSTM-DCF
                </Badge>
              </div>
              <Button
                onClick={handleStartAnalysis}
                className="gap-2 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
              >
                <GeminiSparkle className="h-4 w-4" />
                Run Multi-Agent Analysis
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Loading state
  if (isLoading || isFetching) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin text-purple-600" />
            Analyzing with AI Agents...
          </CardTitle>
          <CardDescription>
            Running sentiment, fundamentals, and ML valuation agents
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-32 w-full" />
          <div className="flex gap-2">
            <Skeleton className="h-8 w-24" />
            <Skeleton className="h-8 w-24" />
            <Skeleton className="h-8 w-24" />
          </div>
          <Skeleton className="h-48 w-full" />
        </CardContent>
      </Card>
    );
  }

  // Render results
  const isCached = !fetchedData && cachedData !== null;

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <GeminiSparkle className="h-5 w-5" />
              Multi-Agent Analysis
              <Badge variant="secondary" className="ml-2">
                {data?.agents_used?.length || 0} Agents
              </Badge>
              {isCached && (
                <Badge
                  variant="outline"
                  className="ml-1 gap-1 text-amber-600 border-amber-300 dark:text-amber-400 dark:border-amber-700"
                >
                  <Clock className="h-3 w-3" />
                  Cached
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="flex items-center gap-2 mt-1">
              <span>Deep AI insights for {ticker}</span>
              {data?.generated_at && (
                <span className="text-xs">
                  • {new Date(data.generated_at).toLocaleTimeString()}
                </span>
              )}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            {isCached && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleStartAnalysis}
                disabled={isFetching}
                className="gap-1 text-muted-foreground hover:text-foreground"
                title="Refresh analysis"
              >
                {isFetching ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <RefreshCw className="h-3 w-3" />
                )}
                Refresh
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleDeepAnalysis}
              disabled={isFetching}
              className="gap-1"
            >
              {isFetching ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Brain className="h-3 w-3" />
              )}
              Deep Analysis
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as typeof activeTab)}
        >
          <TabsList className="grid grid-cols-4 w-full">
            <TabsTrigger value="synthesis" className="gap-1">
              <Zap className="h-3 w-3" />
              <span className="hidden sm:inline">Synthesis</span>
            </TabsTrigger>
            <TabsTrigger value="sentiment" className="gap-1">
              <Newspaper className="h-3 w-3" />
              <span className="hidden sm:inline">Sentiment</span>
            </TabsTrigger>
            <TabsTrigger value="fundamentals" className="gap-1">
              <BarChart3 className="h-3 w-3" />
              <span className="hidden sm:inline">Fundamentals</span>
            </TabsTrigger>
            <TabsTrigger value="ml" className="gap-1">
              <LineChart className="h-3 w-3" />
              <span className="hidden sm:inline">ML Value</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="synthesis" className="mt-4">
            <SynthesisSection data={data?.sections?.synthesis} />
          </TabsContent>

          <TabsContent value="sentiment" className="mt-4">
            <SentimentSection data={data?.sections?.sentiment} />
          </TabsContent>

          <TabsContent value="fundamentals" className="mt-4">
            <FundamentalsSection data={data?.sections?.fundamentals} />
          </TabsContent>

          <TabsContent value="ml" className="mt-4">
            <MLValuationSection data={data?.sections?.ml_valuation} />
          </TabsContent>
        </Tabs>

        {/* Agents Used Footer */}
        <div className="mt-4 pt-4 border-t border-dashed">
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-xs text-muted-foreground">Agents used:</span>
            {data?.agents_used?.map((agent) => (
              <Badge key={agent} variant="outline" className="text-xs">
                {agent.replace("Agent", "")}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// SECTION COMPONENTS
// =============================================================================

function SynthesisSection({ data }: { data?: MultiAgentSynthesis }) {
  if (!data?.available) {
    return (
      <div className="p-4 rounded-lg bg-muted/30 text-center">
        <Lightbulb className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          {data?.error || "Synthesis not available. Try running the analysis."}
        </p>
      </div>
    );
  }

  const synthesisText = extractTextContent(data.synthesis);
  const scores = data.scores_summary;

  return (
    <div className="space-y-4">
      {/* Contrarian Signal Alert */}
      {data.contrarian_signal && (
        <div className="p-3 rounded-lg bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-950/40 dark:to-pink-950/40 border border-purple-300 dark:border-purple-700">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-purple-600" />
            <span className="font-semibold text-purple-800 dark:text-purple-200">
              Contrarian Signal Detected
            </span>
          </div>
          <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
            Market sentiment is bearish but fundamentals are strong. This could
            indicate an accumulation opportunity.
          </p>
        </div>
      )}

      {/* Scores Summary Cards */}
      {scores && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <div className="p-2 rounded-lg bg-muted/50 text-center">
            <div className="text-xs text-muted-foreground">Sentiment</div>
            <div
              className={cn(
                "text-lg font-bold",
                scores.sentiment >= 0.6
                  ? "text-green-600"
                  : scores.sentiment <= 0.4
                    ? "text-red-600"
                    : "text-amber-600",
              )}
            >
              {(scores.sentiment * 100).toFixed(0)}%
            </div>
          </div>
          <div className="p-2 rounded-lg bg-muted/50 text-center">
            <div className="text-xs text-muted-foreground">Quality</div>
            <div
              className={cn(
                "text-lg font-bold",
                scores.quality >= 0.6
                  ? "text-green-600"
                  : scores.quality <= 0.4
                    ? "text-red-600"
                    : "text-amber-600",
              )}
            >
              {(scores.quality * 100).toFixed(0)}%
            </div>
          </div>
          {scores.fair_value && (
            <div className="p-2 rounded-lg bg-muted/50 text-center">
              <div className="text-xs text-muted-foreground">Fair Value</div>
              <div className="text-lg font-bold text-blue-600">
                ${scores.fair_value.toFixed(2)}
              </div>
            </div>
          )}
          {scores.margin_of_safety && (
            <div className="p-2 rounded-lg bg-muted/50 text-center">
              <div className="text-xs text-muted-foreground">
                Margin of Safety
              </div>
              <div
                className={cn(
                  "text-lg font-bold",
                  scores.margin_of_safety >= 20
                    ? "text-green-600"
                    : scores.margin_of_safety >= 0
                      ? "text-amber-600"
                      : "text-red-600",
                )}
              >
                {scores.margin_of_safety.toFixed(1)}%
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main Synthesis Content */}
      <div className="p-5 rounded-lg bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-slate-800 dark:to-slate-800 border border-amber-200 dark:border-amber-700/50">
        <div className="flex items-start gap-3">
          <Zap className="h-5 w-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <h4 className="font-semibold text-base text-amber-900 dark:text-amber-100 mb-3">
              {data.deep_mode
                ? "Comprehensive Diagnostic"
                : "Combined Agent Insights"}
            </h4>
            <div className="text-slate-800 dark:text-slate-200">
              <MarkdownContent content={synthesisText} />
            </div>
          </div>
        </div>
      </div>
      {data.agents_combined && data.agents_combined.length > 0 && (
        <p className="text-xs text-muted-foreground text-center">
          Synthesized from: {data.agents_combined.join(", ")}
          {data.deep_mode && " (Deep Analysis Mode)"}
        </p>
      )}
    </div>
  );
}

function SentimentSection({ data }: { data?: MultiAgentSentiment }) {
  if (!data?.available) {
    return (
      <div className="p-4 rounded-lg bg-muted/30 text-center">
        <Newspaper className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          {data?.error || "Sentiment analysis not available"}
        </p>
      </div>
    );
  }

  const getSentimentColor = (label?: string) => {
    if (!label) return "text-muted-foreground";
    const lower = label.toLowerCase();
    if (lower.includes("bullish")) return "text-green-600";
    if (lower.includes("bearish")) return "text-red-600";
    return "text-amber-600";
  };

  const getSentimentIcon = (label?: string) => {
    if (!label) return <Target className="h-5 w-5" />;
    const lower = label.toLowerCase();
    if (lower.includes("bullish"))
      return <TrendingUp className="h-5 w-5 text-green-600" />;
    if (lower.includes("bearish"))
      return <TrendingDown className="h-5 w-5 text-red-600" />;
    return <Target className="h-5 w-5 text-amber-600" />;
  };

  const analysisText = extractTextContent(data.analysis);

  return (
    <div className="space-y-4">
      {/* Sentiment Score */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
        <div className="flex items-center gap-2">
          {getSentimentIcon(data.sentiment_label)}
          <span className="font-medium">Market Sentiment</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={cn(
              "font-medium",
              getSentimentColor(data.sentiment_label),
            )}
          >
            {data.sentiment_label || "Unknown"}
          </Badge>
          {data.sentiment_score !== undefined && (
            <span className="text-sm text-muted-foreground">
              ({(data.sentiment_score * 100).toFixed(0)}%)
            </span>
          )}
        </div>
      </div>

      {/* Analysis Content */}
      <div className="p-5 rounded-lg border border-slate-200 dark:border-slate-700 bg-card max-h-96 overflow-y-auto">
        <h4 className="font-semibold text-base text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
          <Newspaper className="h-4 w-4 text-blue-600 dark:text-blue-400" />
          Sentiment Analysis
        </h4>
        <div className="text-slate-700 dark:text-slate-300">
          <MarkdownContent content={analysisText} />
        </div>
      </div>
    </div>
  );
}

function FundamentalsSection({ data }: { data?: MultiAgentFundamentals }) {
  if (!data?.available) {
    return (
      <div className="p-4 rounded-lg bg-muted/30 text-center">
        <BarChart3 className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          {data?.error || "Fundamentals analysis not available"}
        </p>
      </div>
    );
  }

  const getScoreColor = (score?: number) => {
    if (!score) return "text-muted-foreground";
    if (score >= 0.7) return "text-green-600";
    if (score >= 0.4) return "text-amber-600";
    return "text-red-600";
  };

  const analysisText = extractTextContent(data.analysis);

  return (
    <div className="space-y-4">
      {/* Score Cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <div
            className={cn(
              "text-2xl font-bold",
              getScoreColor(data.quality_score),
            )}
          >
            {data.quality_score ? (data.quality_score * 100).toFixed(0) : "N/A"}
          </div>
          <div className="text-xs text-muted-foreground">Quality</div>
        </div>
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <div
            className={cn(
              "text-2xl font-bold",
              getScoreColor(data.growth_score),
            )}
          >
            {data.growth_score ? (data.growth_score * 100).toFixed(0) : "N/A"}
          </div>
          <div className="text-xs text-muted-foreground">Growth</div>
        </div>
        <div className="p-3 rounded-lg bg-muted/30 text-center">
          <div
            className={cn(
              "text-2xl font-bold",
              getScoreColor(data.value_score),
            )}
          >
            {data.value_score ? (data.value_score * 100).toFixed(0) : "N/A"}
          </div>
          <div className="text-xs text-muted-foreground">Value</div>
        </div>
      </div>

      {/* Analysis Content */}
      <div className="p-5 rounded-lg border border-slate-200 dark:border-slate-700 bg-card max-h-96 overflow-y-auto">
        <h4 className="font-semibold text-base text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-purple-600 dark:text-purple-400" />
          Fundamental Analysis
        </h4>
        <div className="text-slate-700 dark:text-slate-300">
          <MarkdownContent content={analysisText} />
        </div>
      </div>
    </div>
  );
}

function MLValuationSection({ data }: { data?: MultiAgentMLValuation }) {
  if (!data?.available) {
    return (
      <div className="p-4 rounded-lg bg-muted/30 text-center">
        <LineChart className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
        <p className="text-sm text-muted-foreground">
          {data?.error || "ML valuation not available"}
        </p>
      </div>
    );
  }

  const metrics = data.extracted_metrics;
  const analysisText = extractTextContent(data.analysis);

  // Check if we have any meaningful metrics
  const hasMetrics =
    metrics &&
    (metrics.fair_value || metrics.consensus_score || metrics.margin_of_safety);

  return (
    <div className="space-y-4">
      {/* Extracted Metrics - only show if we have data */}
      {hasMetrics && (
        <div className="grid grid-cols-3 gap-3">
          <div className="p-3 rounded-lg bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border border-green-200 dark:border-green-800 text-center">
            <div className="text-2xl font-bold text-green-700 dark:text-green-400">
              {metrics.fair_value ? `$${metrics.fair_value.toFixed(2)}` : "—"}
            </div>
            <div className="text-xs text-green-600 dark:text-green-500">
              Fair Value
            </div>
          </div>
          <div className="p-3 rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border border-blue-200 dark:border-blue-800 text-center">
            <div className="text-2xl font-bold text-blue-700 dark:text-blue-400">
              {metrics.consensus_score
                ? metrics.consensus_score.toFixed(0)
                : "—"}
            </div>
            <div className="text-xs text-blue-600 dark:text-blue-500">
              Consensus
            </div>
          </div>
          <div className="p-3 rounded-lg bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-amber-950/30 dark:to-yellow-950/30 border border-amber-200 dark:border-amber-800 text-center">
            <div className="text-2xl font-bold text-amber-700 dark:text-amber-400">
              {metrics.margin_of_safety
                ? `${metrics.margin_of_safety.toFixed(1)}%`
                : "—"}
            </div>
            <div className="text-xs text-amber-600 dark:text-amber-500">
              Margin of Safety
            </div>
          </div>
        </div>
      )}

      {/* Analysis Content */}
      <div className="p-5 rounded-lg border border-slate-200 dark:border-slate-700 bg-card max-h-96 overflow-y-auto">
        <h4 className="font-semibold text-base text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
          <LineChart className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          LSTM-DCF Valuation Analysis
        </h4>
        {analysisText ? (
          <div className="text-slate-700 dark:text-slate-300">
            <MarkdownContent content={analysisText} />
          </div>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-400 italic">
            ML valuation analysis is still processing or not available for this
            stock. The LSTM-DCF model requires sufficient historical data to
            generate predictions.
          </p>
        )}
      </div>
    </div>
  );
}

export default MultiAgentAnalysisPanel;
