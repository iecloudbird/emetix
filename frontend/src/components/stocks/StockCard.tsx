/**
 * StockCard Component
 *
 * Displays a stock summary with key metrics, valuation score, and recommendation.
 * Reusable component following atomic design principles.
 */
"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Stock } from "@/lib/api";

interface StockCardProps {
  stock: Stock;
  rank?: number;
  showDetails?: boolean;
}

export function StockCard({ stock, rank, showDetails = true }: StockCardProps) {
  // Use raw (uncapped) margin of safety if available
  const upside = stock.margin_of_safety_raw ?? stock.margin_of_safety ?? 0;
  const isPositive = upside > 0;
  const isNegative = upside < 0;

  const getRecommendationColor = (rec: string) => {
    const recUpper = rec?.toUpperCase() || "";
    if (recUpper.includes("BUY") || recUpper.includes("STRONG")) {
      return "bg-green-500 hover:bg-green-600";
    } else if (recUpper.includes("SELL") || recUpper.includes("AVOID")) {
      return "bg-red-500 hover:bg-red-600";
    }
    return "bg-yellow-500 hover:bg-yellow-600";
  };

  const getRiskColor = (risk?: string) => {
    switch (risk?.toUpperCase()) {
      case "LOW":
        return "bg-green-100 text-green-800";
      case "HIGH":
        return "bg-red-100 text-red-800";
      default:
        return "bg-yellow-100 text-yellow-800";
    }
  };

  return (
    <Link href={`/stock/${stock.ticker}`}>
      <Card className="hover:shadow-lg transition-shadow cursor-pointer">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {rank && (
                <span className="text-2xl font-bold text-muted-foreground">
                  #{rank}
                </span>
              )}
              <div>
                <CardTitle className="text-lg">{stock.ticker}</CardTitle>
                <p className="text-sm text-muted-foreground truncate max-w-[180px]">
                  {stock.company_name}
                </p>
              </div>
            </div>
            <Badge className={getRecommendationColor(stock.recommendation)}>
              {stock.recommendation}
            </Badge>
          </div>
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            {/* Price & Upside */}
            <div>
              <p className="text-2xl font-bold">
                ${stock.current_price.toFixed(2)}
              </p>
              <div
                className={cn(
                  "flex items-center gap-1 text-sm",
                  isPositive && "text-green-600",
                  isNegative && "text-red-600",
                  !isPositive && !isNegative && "text-gray-500"
                )}
              >
                {isPositive && <TrendingUp className="w-4 h-4" />}
                {isNegative && <TrendingDown className="w-4 h-4" />}
                {!isPositive && !isNegative && <Minus className="w-4 h-4" />}
                <span>
                  {isPositive ? "+" : ""}
                  {upside.toFixed(1)}% upside
                </span>
              </div>
            </div>

            {/* Valuation Score */}
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Valuation Score</p>
              <p
                className={cn(
                  "text-2xl font-bold",
                  stock.valuation_score >= 70 && "text-green-600",
                  stock.valuation_score >= 50 &&
                    stock.valuation_score < 70 &&
                    "text-yellow-600",
                  stock.valuation_score < 50 && "text-red-600"
                )}
              >
                {stock.valuation_score.toFixed(0)}
              </p>
            </div>
          </div>

          {showDetails && (
            <div className="mt-4 pt-4 border-t">
              <div className="flex justify-between text-sm">
                <div>
                  <p className="text-muted-foreground">LSTM Fair Value</p>
                  <p className="font-medium">
                    {stock.lstm_fair_value
                      ? `$${stock.lstm_fair_value.toFixed(2)}`
                      : "N/A"}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-muted-foreground">Sector</p>
                  <p className="font-medium">{stock.sector}</p>
                </div>
              </div>

              {stock.risk_level && (
                <div className="mt-2">
                  <Badge
                    variant="outline"
                    className={getRiskColor(stock.risk_level)}
                  >
                    {stock.risk_level} Risk
                  </Badge>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}
