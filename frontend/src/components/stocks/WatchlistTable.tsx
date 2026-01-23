/**
 * WatchlistTable Component
 *
 * Displays watchlist stocks in a sortable table format.
 * Suitable for dashboard overview and full watchlist page.
 * Enhanced with risk profile suitability indicators (Phase 2).
 */
"use client";

import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { TrendingUp, TrendingDown, Filter } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Stock, ValuationStatus } from "@/lib/api";
import { SuitabilityBadge } from "@/components/risk-profile/SuitabilityBadge";
import { PositionSizingTooltip } from "@/components/risk-profile/PositionSizingTooltip";
import { useLocalRiskProfile } from "@/hooks/useRiskProfile";
import { useState, useMemo } from "react";

// Risk profile settings for filtering
interface RiskProfileFilter {
  enabled: boolean;
  betaRange: { min: number; max: number };
  requiredMoS: number;
  portfolioValue?: number;
}

interface WatchlistTableProps {
  stocks: Stock[];
  showRank?: boolean;
  showForwardMetrics?: boolean;
  showSuitability?: boolean;
  riskProfile?: RiskProfileFilter | null;
}

export function WatchlistTable({
  stocks,
  showRank = true,
  showForwardMetrics = false,
  showSuitability = false,
  riskProfile = null,
}: WatchlistTableProps) {
  const [filterByProfile, setFilterByProfile] = useState(false);
  const { profileId } = useLocalRiskProfile();

  // Calculate suitability for each stock based on risk profile
  const getSuitability = (
    stock: Stock
  ): "excellent" | "good" | "moderate" | "poor" | "unknown" => {
    if (!riskProfile) return "unknown";

    const beta = stock.beta ?? 1.0;
    // Use raw (uncapped) MoS for accurate suitability calculation
    const mos = stock.margin_of_safety_raw ?? stock.margin_of_safety ?? 0;
    const { betaRange, requiredMoS } = riskProfile;

    const betaInRange = beta >= betaRange.min && beta <= betaRange.max;
    const mosOk = mos >= requiredMoS;

    if (betaInRange && mosOk) return "excellent";
    if (betaInRange || mosOk) return "moderate";
    return "poor";
  };

  // Filter stocks based on profile if enabled
  const displayedStocks = useMemo(() => {
    if (!filterByProfile || !riskProfile) return stocks;

    return stocks.filter((stock) => {
      const suitability = getSuitability(stock);
      return suitability === "excellent" || suitability === "good";
    });
  }, [stocks, filterByProfile, riskProfile]);
  const getRecommendationColor = (rec: string) => {
    const recUpper = rec?.toUpperCase() || "";
    if (recUpper.includes("BUY") || recUpper.includes("STRONG")) {
      return "bg-green-500";
    } else if (recUpper.includes("SELL") || recUpper.includes("AVOID")) {
      return "bg-red-500";
    }
    return "bg-yellow-500";
  };

  const getValuationStatusBadge = (status?: ValuationStatus) => {
    switch (status) {
      case "SIGNIFICANTLY_UNDERVALUED":
        return (
          <Badge className="bg-green-600 text-xs">Significantly Under</Badge>
        );
      case "MODERATELY_UNDERVALUED":
        return <Badge className="bg-green-500 text-xs">Moderately Under</Badge>;
      case "SLIGHTLY_UNDERVALUED":
        return <Badge className="bg-green-400 text-xs">Slightly Under</Badge>;
      case "FAIRLY_VALUED":
        return <Badge className="bg-gray-500 text-xs">Fair Value</Badge>;
      case "SLIGHTLY_OVERVALUED":
        return <Badge className="bg-orange-400 text-xs">Slightly Over</Badge>;
      case "OVERVALUED":
        return <Badge className="bg-red-500 text-xs">Overvalued</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Profile Filter Toggle */}
      {showSuitability && riskProfile && (
        <div className="flex items-center justify-between px-2 py-2 bg-muted/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <Label
              htmlFor="profile-filter"
              className="text-sm text-muted-foreground"
            >
              Show only suitable stocks for my profile
            </Label>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              MoS ≥ {riskProfile.requiredMoS}% | Beta:{" "}
              {riskProfile.betaRange.min.toFixed(1)}-
              {riskProfile.betaRange.max.toFixed(1)}
            </span>
            <Switch
              id="profile-filter"
              checked={filterByProfile}
              onCheckedChange={setFilterByProfile}
            />
          </div>
        </div>
      )}

      <Table>
        <TableHeader>
          <TableRow>
            {showRank && <TableHead className="w-12">#</TableHead>}
            {showSuitability && (
              <TableHead className="w-16 text-center">Fit</TableHead>
            )}
            <TableHead>Ticker</TableHead>
            <TableHead>Company</TableHead>
            <TableHead className="text-right">Price</TableHead>
            <TableHead className="text-right">Fair Value</TableHead>
            <TableHead className="text-right">MoS</TableHead>
            {showForwardMetrics && (
              <TableHead className="text-right">Fwd P/E</TableHead>
            )}
            <TableHead className="text-right">Score</TableHead>
            <TableHead className="text-center">Status</TableHead>
            {showSuitability && riskProfile?.portfolioValue && (
              <TableHead className="w-12 text-center">Size</TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {displayedStocks.map((stock, index) => {
            // Use raw (uncapped) margin of safety if available
            const mos =
              stock.margin_of_safety_raw ?? stock.margin_of_safety ?? 0;
            const isPositive = mos > 0;
            const suitability = getSuitability(stock);

            return (
              <TableRow
                key={stock.ticker}
                className={cn(
                  "cursor-pointer hover:bg-muted/50",
                  showSuitability && suitability === "poor" && "opacity-60"
                )}
              >
                {showRank && (
                  <TableCell className="font-medium text-muted-foreground">
                    {stock.rank ?? index + 1}
                  </TableCell>
                )}
                {showSuitability && (
                  <TableCell className="text-center">
                    <SuitabilityBadge
                      suitability={suitability}
                      beta={stock.beta ?? undefined}
                      marginOfSafety={mos}
                      profileBetaRange={riskProfile?.betaRange}
                      requiredMoS={riskProfile?.requiredMoS}
                      compact
                    />
                  </TableCell>
                )}
                <TableCell>
                  <Link
                    href={`/stock/${stock.ticker}`}
                    className="font-bold hover:underline"
                  >
                    {stock.ticker}
                  </Link>
                </TableCell>
                <TableCell className="max-w-50 truncate">
                  {stock.company_name}
                </TableCell>
                <TableCell className="text-right font-medium">
                  ${stock.current_price.toFixed(2)}
                </TableCell>
                <TableCell className="text-right">
                  {stock.fair_value
                    ? `$${stock.fair_value.toFixed(2)}`
                    : stock.lstm_fair_value
                    ? `$${stock.lstm_fair_value.toFixed(2)}`
                    : "N/A"}
                </TableCell>
                <TableCell
                  className={cn(
                    "text-right font-medium",
                    isPositive ? "text-green-600" : "text-red-600"
                  )}
                >
                  <span className="flex items-center justify-end gap-1">
                    {isPositive ? (
                      <TrendingUp className="w-4 h-4" />
                    ) : (
                      <TrendingDown className="w-4 h-4" />
                    )}
                    {isPositive ? "+" : ""}
                    {mos.toFixed(1)}%
                  </span>
                </TableCell>
                {showForwardMetrics && (
                  <TableCell className="text-right text-sm">
                    {stock.forward_metrics?.forward_pe
                      ? stock.forward_metrics.forward_pe.toFixed(1)
                      : stock.forward_pe
                      ? stock.forward_pe.toFixed(1)
                      : "N/A"}
                    {stock.forward_metrics?.pe_trend === "IMPROVING" && (
                      <span className="text-green-500 ml-1">↓</span>
                    )}
                    {stock.forward_metrics?.pe_trend === "DECLINING" && (
                      <span className="text-red-500 ml-1">↑</span>
                    )}
                  </TableCell>
                )}
                <TableCell className="text-right">
                  <span
                    className={cn(
                      "font-bold",
                      stock.valuation_score >= 70 && "text-green-600",
                      stock.valuation_score < 50 && "text-red-600"
                    )}
                  >
                    {stock.valuation_score.toFixed(0)}
                  </span>
                </TableCell>
                <TableCell className="text-center">
                  {getValuationStatusBadge(stock.valuation_status)}
                </TableCell>
                {showSuitability && riskProfile?.portfolioValue && (
                  <TableCell className="text-center">
                    <PositionSizingTooltip
                      ticker={stock.ticker}
                      currentPrice={stock.current_price}
                      marginOfSafety={stock.margin_of_safety ?? 0}
                      beta={stock.beta ?? 1.0}
                      profileId={profileId}
                    />
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

      {/* Empty state when filtering */}
      {filterByProfile && displayedStocks.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <p>No stocks match your risk profile criteria.</p>
          <p className="text-sm mt-1">
            Try adjusting your profile or disable the filter.
          </p>
        </div>
      )}
    </div>
  );
}

// Helper to calculate max position based on beta
function calculateMaxPosition(beta: number): number {
  // Conservative position sizing: higher beta = smaller position
  if (beta < 0.8) return 10; // Low volatility: max 10%
  if (beta < 1.0) return 8;
  if (beta < 1.2) return 6;
  if (beta < 1.5) return 4;
  return 3; // High volatility: max 3%
}
