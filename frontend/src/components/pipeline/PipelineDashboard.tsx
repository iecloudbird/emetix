/**
 * PipelineDashboard - Phase 3 Quality Screening Pipeline Dashboard
 *
 * Main dashboard with Buy/Hold/Watch tabs displaying qualified stocks.
 * Shows 4-pillar scores, classifications, and fair value estimates.
 */
"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  RefreshCcw,
  TrendingUp,
  AlertCircle,
  Sparkles,
  Target,
  Eye,
} from "lucide-react";
import {
  usePipelineClassified,
  usePipelineSummary,
} from "@/hooks/use-pipeline";
import {
  ClassificationBadge,
  ScoreBadge,
} from "@/components/pipeline/PipelineBadges";
import { PillarScoreSummary } from "@/components/charts/PillarRadarChart";
import type { QualifiedStock } from "@/lib/api";

interface PipelineDashboardProps {
  onStockSelect?: (ticker: string) => void;
}

export function PipelineDashboard({ onStockSelect }: PipelineDashboardProps) {
  const [activeTab, setActiveTab] = useState<"buy" | "hold" | "watch">("buy");
  const {
    data: classified,
    isLoading,
    error,
    refetch,
  } = usePipelineClassified();
  const { data: summary } = usePipelineSummary();

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent className="flex items-center gap-2 py-4">
          <AlertCircle className="h-5 w-5 text-red-500" />
          <span className="text-red-700">Failed to load pipeline data</span>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      {/* <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <SummaryCard
          label="Buy"
          count={classified?.counts.buy ?? 0}
          color="green"
          isLoading={isLoading}
        />
        <SummaryCard
          label="Hold"
          count={classified?.counts.hold ?? 0}
          color="blue"
          isLoading={isLoading}
        />
        <SummaryCard
          label="Watch"
          count={classified?.counts.watch ?? 0}
          color="yellow"
          isLoading={isLoading}
        />
        <SummaryCard
          label="Total Qualified"
          count={classified?.counts.total ?? 0}
          color="gray"
          isLoading={isLoading}
        />
      </div> */}

      {/* Main Tabs */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Quality Pipeline
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RefreshCcw
              className={`mr-2 h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>
        </CardHeader>
        <CardContent>
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as "buy" | "hold" | "watch")}
          >
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger
                value="buy"
                className="data-[state=active]:bg-green-100"
              >
                Buy ({classified?.counts.buy ?? 0})
              </TabsTrigger>
              <TabsTrigger
                value="hold"
                className="data-[state=active]:bg-blue-100"
              >
                Hold ({classified?.counts.hold ?? 0})
              </TabsTrigger>
              <TabsTrigger
                value="watch"
                className="data-[state=active]:bg-yellow-100"
              >
                Watch ({classified?.counts.watch ?? 0})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="buy" className="mt-4">
              <StockTable
                stocks={classified?.classified?.buy ?? []}
                isLoading={isLoading}
                onStockSelect={onStockSelect}
                emptyMessage="No stocks meet BUY criteria (MoS ≥ 20%, Score ≥ 70)"
              />
            </TabsContent>

            <TabsContent value="hold" className="mt-4">
              <StockTable
                stocks={classified?.classified?.hold ?? []}
                isLoading={isLoading}
                onStockSelect={onStockSelect}
                emptyMessage="No stocks meet HOLD criteria (MoS -10% to +20%, Score ≥ 70)"
              />
            </TabsContent>

            <TabsContent value="watch" className="mt-4">
              <StockTable
                stocks={classified?.classified?.watch ?? []}
                isLoading={isLoading}
                onStockSelect={onStockSelect}
                emptyMessage="No stocks in WATCH list"
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Last Scan Info */}
      {summary && (
        <div className="text-xs text-muted-foreground">
          Last scan:{" "}
          {summary.pipeline?.last_scan?.completed_at
            ? new Date(summary.pipeline.last_scan.completed_at).toLocaleString()
            : "Never"}
        </div>
      )}
    </div>
  );
}

// ============================================================
// Sub-components
// ============================================================

interface SummaryCardProps {
  label: string;
  count: number;
  color: "green" | "blue" | "yellow" | "gray";
  isLoading: boolean;
}

function SummaryCard({ label, count, color, isLoading }: SummaryCardProps) {
  const colorClasses = {
    green: "border-green-200 bg-green-50",
    blue: "border-blue-200 bg-blue-50",
    yellow: "border-yellow-200 bg-yellow-50",
    gray: "border-gray-200 bg-gray-50",
  };

  const textColors = {
    green: "text-green-700",
    blue: "text-blue-700",
    yellow: "text-yellow-700",
    gray: "text-gray-700",
  };

  return (
    <Card className={colorClasses[color]}>
      <CardContent className="pt-4">
        <div className="text-sm text-muted-foreground">{label}</div>
        {isLoading ? (
          <Skeleton className="mt-1 h-8 w-16" />
        ) : (
          <div className={`text-3xl font-bold ${textColors[color]}`}>
            {count}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface StockTableProps {
  stocks: QualifiedStock[];
  isLoading: boolean;
  onStockSelect?: (ticker: string) => void;
  emptyMessage: string;
}

function StockTable({
  stocks,
  isLoading,
  onStockSelect,
  emptyMessage,
}: StockTableProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  if (stocks.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-muted-foreground">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">Ticker</TableHead>
            <TableHead>Company</TableHead>
            <TableHead className="text-center">Score</TableHead>
            <TableHead className="text-center">Pillars</TableHead>
            <TableHead className="text-right">Price</TableHead>
            <TableHead className="text-right">Fair Value</TableHead>
            <TableHead className="text-right">MoS</TableHead>
            <TableHead className="text-center">Why Listed</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {stocks.map((stock) => (
            <TableRow
              key={stock.ticker}
              className="cursor-pointer hover:bg-muted/50"
              onClick={() => onStockSelect?.(stock.ticker)}
            >
              <TableCell className="font-medium">
                <div className="flex items-center gap-2">
                  {stock.ticker}
                  <ClassificationBadge
                    classification={stock.classification}
                    size="sm"
                  />
                </div>
              </TableCell>
              <TableCell>
                <div>
                  <div className="font-medium">{stock.company_name}</div>
                  <div className="text-xs text-muted-foreground">
                    {stock.sector}
                  </div>
                </div>
              </TableCell>
              <TableCell className="text-center">
                <ScoreBadge score={stock.composite_score} />
              </TableCell>
              <TableCell className="text-center">
                <PillarScoreSummary pillarScores={stock.pillar_scores} />
              </TableCell>
              <TableCell className="text-right">
                ${stock.current_price?.toFixed(2) ?? "N/A"}
              </TableCell>
              <TableCell className="text-right">
                <div className="flex flex-col items-end">
                  <span className="font-medium text-blue-600">
                    $
                    {stock.lstm_fair_value?.toFixed(2) ??
                      stock.fair_value?.toFixed(2) ??
                      "N/A"}
                  </span>
                  {stock.lstm_fair_value && (
                    <span className="text-[10px] text-muted-foreground">
                      LSTM-DCF
                    </span>
                  )}
                </div>
              </TableCell>
              <TableCell className="text-right">
                <span
                  className={
                    stock.margin_of_safety >= 20
                      ? "text-green-600 font-medium"
                      : stock.margin_of_safety >= 0
                      ? "text-yellow-600"
                      : "text-red-600"
                  }
                >
                  {stock.margin_of_safety >= 0 ? "+" : ""}
                  {stock.margin_of_safety?.toFixed(1)}%
                </span>
              </TableCell>
              <TableCell className="text-center">
                <WhyListedBadge
                  classification={stock.classification}
                  score={stock.composite_score}
                  mos={stock.margin_of_safety}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// WhyListedBadge - Shows a short reason why the stock is in this category
interface WhyListedBadgeProps {
  classification: string;
  score: number;
  mos: number;
}

function WhyListedBadge({ classification, score, mos }: WhyListedBadgeProps) {
  let reason = "";
  let icon = <Target className="h-3 w-3" />;
  let tooltip = "";

  const classLower = classification?.toLowerCase() || "";

  if (classLower === "buy") {
    if (mos >= 30) {
      reason = "Deep Value";
      tooltip = `Significantly undervalued with ${mos.toFixed(
        0
      )}% margin of safety`;
      icon = <Sparkles className="h-3 w-3" />;
    } else if (score >= 80) {
      reason = "Top Quality";
      tooltip = `Excellent composite score of ${score.toFixed(0)}/100`;
      icon = <Sparkles className="h-3 w-3" />;
    } else {
      reason = "Value + Quality";
      tooltip = `Good value (${mos.toFixed(
        0
      )}% MoS) with solid quality (${score.toFixed(0)} score)`;
    }
  } else if (classLower === "hold") {
    if (score >= 75) {
      reason = "Quality Hold";
      tooltip = `High quality stock (${score.toFixed(
        0
      )} score) near fair value`;
    } else {
      reason = "Fair Value";
      tooltip = `Trading near estimated fair value with ${score.toFixed(
        0
      )} quality score`;
    }
  } else {
    if (mos < -10) {
      reason = "Overvalued";
      tooltip = `Currently overvalued by ${Math.abs(mos).toFixed(0)}%`;
      icon = <Eye className="h-3 w-3" />;
    } else if (score < 60) {
      reason = "Quality Risk";
      tooltip = `Quality concerns (score ${score.toFixed(
        0
      )}) - monitor for improvement`;
      icon = <Eye className="h-3 w-3" />;
    } else {
      reason = "Monitor";
      tooltip = "Interesting but not yet actionable";
      icon = <Eye className="h-3 w-3" />;
    }
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="text-xs cursor-help flex items-center gap-1"
          >
            {icon}
            {reason}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          <p className="max-w-[200px] text-xs">{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export default PipelineDashboard;
