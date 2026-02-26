/**
 * Stock Preview Panel (v2)
 *
 * AI-focused insight panel shown when clicking a stock row in the screener.
 * Instead of duplicating metrics from the table/detail page, this panel gives
 * users quick, human-readable takeaways to decide if they want to dig deeper:
 *
 * - AI-generated headline & one-liner (from quick analysis endpoint)
 * - Rich textual strengths/weaknesses derived from pillar scores + metrics
 * - Compact valuation snapshot (MoS + score only)
 * - CTA to full stock detail page
 */
"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  Minus,
  X,
  Loader2,
  Building2,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import {
  ClassificationBadge,
  ScoreBadge,
} from "@/components/pipeline/PipelineBadges";
import { useQuickAnalysis } from "@/hooks/use-pipeline";
import type { QualifiedStock, PillarScores } from "@/lib/api";

interface StockPreviewPanelProps {
  stock: QualifiedStock;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Insight Generator — turns quantitative pillar + metric data into sentences
// ---------------------------------------------------------------------------

interface Insight {
  text: string;
  type: "strength" | "weakness" | "neutral";
}

function getPillarScore(pillars: PillarScores, key: string): number {
  const pillar = pillars?.[key as keyof PillarScores];
  if (pillar && typeof pillar === "object" && "score" in pillar) {
    return pillar.score;
  }
  return 0;
}

function generateInsights(stock: QualifiedStock): Insight[] {
  const insights: Insight[] = [];
  const pillars = stock.pillar_scores;

  // Value pillar
  const valueScore = getPillarScore(pillars, "value");
  if (valueScore >= 75) {
    insights.push({
      text: `Strong value proposition — trades well below estimated fair value with a ${stock.margin_of_safety > 0 ? `${stock.margin_of_safety.toFixed(0)}% margin of safety` : "competitive valuation"}.`,
      type: "strength",
    });
  } else if (valueScore >= 60) {
    insights.push({
      text: "Reasonably valued relative to intrinsic worth, offering moderate upside potential.",
      type: "neutral",
    });
  } else if (valueScore > 0) {
    insights.push({
      text: "Valuation appears stretched — limited margin of safety at current prices.",
      type: "weakness",
    });
  }

  // Quality pillar
  const qualityScore = getPillarScore(pillars, "quality");
  if (qualityScore >= 75) {
    const roicNote =
      stock.fcf_roic != null && stock.fcf_roic > 0.15
        ? ` with ${(stock.fcf_roic * 100).toFixed(0)}% FCF ROIC`
        : "";
    insights.push({
      text: `High-quality business${roicNote} — strong capital efficiency and profitability.`,
      type: "strength",
    });
  } else if (qualityScore < 50 && qualityScore > 0) {
    insights.push({
      text: "Quality metrics below average — margins or returns on capital may be weak.",
      type: "weakness",
    });
  }

  // Growth pillar
  const growthScore = getPillarScore(pillars, "growth");
  if (growthScore >= 75) {
    const growthNote =
      stock.revenue_growth != null
        ? ` Revenue growing at ${(stock.revenue_growth * 100).toFixed(0)}% YoY.`
        : "";
    insights.push({
      text: `Strong growth trajectory.${growthNote}`,
      type: "strength",
    });
  } else if (growthScore < 50 && growthScore > 0) {
    insights.push({
      text: "Growth is decelerating or below sector average — may need a catalyst.",
      type: "weakness",
    });
  }

  // Safety pillar
  const safetyScore = getPillarScore(pillars, "safety");
  if (safetyScore >= 75) {
    const betaNote =
      stock.beta != null ? ` Beta of ${stock.beta.toFixed(2)}.` : "";
    insights.push({
      text: `Low-risk profile with controlled volatility.${betaNote}`,
      type: "strength",
    });
  } else if (safetyScore < 50 && safetyScore > 0) {
    const betaNote =
      stock.beta != null && stock.beta > 1.3
        ? ` High beta (${stock.beta.toFixed(2)}) means amplified market moves.`
        : "";
    insights.push({
      text: `Higher risk/volatility than average.${betaNote}`,
      type: "weakness",
    });
  }

  // LSTM predicted growth (unique ML insight)
  if (stock.lstm_predicted_growth != null) {
    const pct = (stock.lstm_predicted_growth * 100).toFixed(0);
    if (stock.lstm_predicted_growth > 0.1) {
      insights.push({
        text: `Our LSTM model forecasts ${pct}% growth — a bullish forward outlook.`,
        type: "strength",
      });
    } else if (stock.lstm_predicted_growth < -0.05) {
      insights.push({
        text: `LSTM model forecasts ${pct}% decline — caution on near-term outlook.`,
        type: "weakness",
      });
    }
  }

  // Balanced / unbalanced profile
  if (stock.analysis?.balanced) {
    insights.push({
      text: "Well-rounded profile with balanced scores across value, quality, growth, and safety.",
      type: "neutral",
    });
  }

  return insights;
}

// ---------------------------------------------------------------------------
// Company Description — matches section heading style (Strengths/Weaknesses)
// ---------------------------------------------------------------------------

function CompanyDescription({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const textRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const el = textRef.current;
    if (el) {
      setIsOverflowing(el.scrollHeight > el.clientHeight + 2);
    }
  }, [text]);

  return (
    <div>
      <p className="text-xs font-semibold text-foreground flex items-center gap-1 mb-1.5">
        <Building2 className="h-3 w-3" />
        About the Company
      </p>
      <div className={`relative ${expanded ? "max-h-28 overflow-y-auto" : ""}`}>
        <p
          ref={textRef}
          className={`text-xs text-foreground/80 leading-relaxed pl-4 ${
            expanded ? "" : "line-clamp-4"
          }`}
        >
          {text}
        </p>
        {!expanded && isOverflowing && (
          <div className="absolute bottom-0 left-0 right-0 h-5 bg-linear-to-t from-card to-transparent pointer-events-none" />
        )}
      </div>
      {isOverflowing && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-0.5 pl-4 mt-1 text-[10px] text-primary hover:underline"
        >
          {expanded ? (
            <>
              <ChevronUp className="h-3 w-3" /> Show less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3" /> Read more
            </>
          )}
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function StockPreviewPanel({ stock, onClose }: StockPreviewPanelProps) {
  const { data: quickAnalysis, isLoading: aiLoading } = useQuickAnalysis(
    stock.ticker,
  );
  const upside = stock.margin_of_safety ?? 0;
  const isPositive = upside > 0;
  const insights = generateInsights(stock);

  const strengths = insights.filter((i) => i.type === "strength");
  const weaknesses = insights.filter((i) => i.type === "weakness");
  const neutrals = insights.filter((i) => i.type === "neutral");

  return (
    <Card className="w-full border-x-4 border-x-primary shadow-lg overflow-hidden">
      {/* Header */}
      <CardHeader className="pb-2 pt-3 px-4">
        <div className="flex items-center justify-between">
          <div className="min-w-0">
            <CardTitle className="text-lg flex items-center gap-2">
              {stock.ticker}
              <ClassificationBadge classification={stock.classification} />
              <ScoreBadge score={stock.composite_score} />
            </CardTitle>
            <p className="text-xs text-muted-foreground truncate">
              {stock.company_name} · {stock.sector}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="px-4 pb-4 space-y-3">
        {/* AI Quick Headline */}
        <div className="rounded-md bg-primary/5 border border-primary/10 px-3 py-2">
          <div className="flex items-start gap-2">
            <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
            <div className="text-sm">
              {aiLoading ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Generating insight...
                </div>
              ) : quickAnalysis?.summary?.headline ? (
                <>
                  <p className="font-medium leading-snug">
                    {quickAnalysis.summary.headline}
                  </p>
                  {quickAnalysis.summary.one_liner && (
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {quickAnalysis.summary.one_liner}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-muted-foreground">
                  Score {stock.composite_score.toFixed(0)}/100 ·{" "}
                  {stock.classification.toUpperCase()} signal
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Company Description */}
        {!aiLoading && quickAnalysis?.summary?.description && (
          <CompanyDescription text={quickAnalysis.summary.description} />
        )}

        {/* Strengths */}
        {strengths.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-green-600 dark:text-green-400 flex items-center gap-1 mb-1.5">
              <ThumbsUp className="h-3 w-3" />
              Strengths
            </p>
            <ul className="space-y-1">
              {strengths.map((s, i) => (
                <li
                  key={i}
                  className="text-xs text-muted-foreground leading-relaxed pl-4 relative before:content-['•'] before:absolute before:left-1 before:text-green-500"
                >
                  {s.text}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Weaknesses */}
        {weaknesses.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-red-600 dark:text-red-400 flex items-center gap-1 mb-1.5">
              <ThumbsDown className="h-3 w-3" />
              Risk Factors
            </p>
            <ul className="space-y-1">
              {weaknesses.map((w, i) => (
                <li
                  key={i}
                  className="text-xs text-muted-foreground leading-relaxed pl-4 relative before:content-['•'] before:absolute before:left-1 before:text-red-500"
                >
                  {w.text}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Neutral observations */}
        {neutrals.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-muted-foreground flex items-center gap-1 mb-1.5">
              <Minus className="h-3 w-3" />
              Observations
            </p>
            <ul className="space-y-1">
              {neutrals.map((n, i) => (
                <li
                  key={i}
                  className="text-xs text-muted-foreground leading-relaxed pl-4 relative before:content-['•'] before:absolute before:left-1"
                >
                  {n.text}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Empty state */}
        {strengths.length === 0 &&
          weaknesses.length === 0 &&
          neutrals.length === 0 && (
            <p className="text-xs text-muted-foreground text-center py-2">
              Limited pillar data available for this stock.
            </p>
          )}

        {/* CTA */}
        <Link href={`/stock/${stock.ticker}`}>
          <Button className="w-full" size="sm">
            View Full Analysis
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link>
      </CardContent>
    </Card>
  );
}
