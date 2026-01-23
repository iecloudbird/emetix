/**
 * PriceChart Component
 *
 * Displays stock price history using Recharts.
 * Supports 1Y and 5Y timeframes with interactive tooltips.
 */
"use client";

import { useState, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ChartDataPoint } from "@/lib/api";

interface PriceChartProps {
  data1Y: ChartDataPoint[];
  data5Y: ChartDataPoint[];
  currentPrice?: number;
  fairValue?: number;
  ticker: string;
}

type TimeRange = "1Y" | "5Y";

// Custom tooltip component for modern styling
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;

  const price = payload[0]?.value;
  const date = new Date(label);

  return (
    <div className="bg-popover/95 backdrop-blur-sm border border-border rounded-lg shadow-lg px-3 py-2 min-w-[120px]">
      <p className="text-[10px] text-muted-foreground font-medium mb-0.5">
        {date.toLocaleDateString("en-US", {
          weekday: "short",
          month: "short",
          day: "numeric",
          year: "numeric",
        })}
      </p>
      <div className="flex items-center gap-1.5">
        <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
        <span className="text-sm font-semibold">${price?.toFixed(2)}</span>
      </div>
    </div>
  );
}

export function PriceChart({
  data1Y,
  data5Y,
  currentPrice,
  fairValue,
  ticker,
}: PriceChartProps) {
  const [range, setRange] = useState<TimeRange>("1Y");

  // Normalize data to use 'close' field (backend may return 'price')
  const rawData = range === "1Y" ? data1Y : data5Y;
  const data = useMemo(
    () =>
      (rawData ?? [])
        .map((d) => ({
          ...d,
          close: Number(d.close ?? d.price ?? 0),
        }))
        .filter((d) => d.close > 0 && !isNaN(d.close) && d.date),
    [rawData]
  );

  // Calculate Y-axis domain to optionally include fair value
  const { yDomain, showFairValueLine } = useMemo(() => {
    if (!data.length)
      return { yDomain: ["auto", "auto"], showFairValueLine: false };

    const prices = data.map((d) => d.close);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Check if fair value is within a reasonable display range (within 2x of max price)
    const shouldShowFairValue =
      fairValue && fairValue <= maxPrice * 2 && fairValue >= minPrice * 0.5;

    if (shouldShowFairValue && fairValue) {
      // Extend domain to include fair value with some padding
      const domainMin = Math.min(minPrice, fairValue) * 0.95;
      const domainMax = Math.max(maxPrice, fairValue) * 1.05;
      return { yDomain: [domainMin, domainMax], showFairValueLine: true };
    }

    // Default: just show price range with padding
    return {
      yDomain: [minPrice - priceRange * 0.05, maxPrice + priceRange * 0.05],
      showFairValueLine: false,
    };
  }, [data, fairValue]);

  // Format for display
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      month: "short",
      year: "2-digit",
    });
  };

  const formatPrice = (value: number) => `$${value.toFixed(2)}`;

  // Handle empty data
  if (!data || data.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle>{ticker} Price History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            No price data available for this timeframe
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle>{ticker} Price History</CardTitle>
          <div className="flex gap-1">
            <Button
              variant={range === "1Y" ? "default" : "outline"}
              size="sm"
              onClick={() => setRange("1Y")}
            >
              1Y
            </Button>
            <Button
              variant={range === "5Y" ? "default" : "outline"}
              size="sm"
              onClick={() => setRange("5Y")}
            >
              5Y
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 10, right: 20, bottom: 5, left: 0 }}
            >
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(128, 128, 128, 0.15)"
                vertical={false}
              />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 11, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tickFormatter={formatPrice}
                tick={{ fontSize: 11, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
                domain={yDomain as [number, number]}
                width={65}
              />
              <Tooltip
                content={<CustomTooltip />}
                cursor={{
                  stroke: "rgba(128, 128, 128, 0.3)",
                  strokeDasharray: "4 4",
                }}
              />

              {/* Price area with gradient fill */}
              <Area
                type="monotone"
                dataKey="close"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#priceGradient)"
                dot={false}
                activeDot={{
                  r: 6,
                  fill: "#3b82f6",
                  stroke: "#fff",
                  strokeWidth: 2,
                }}
              />

              {/* Fair value reference line - only show if within visible range */}
              {showFairValueLine && fairValue && (
                <ReferenceLine
                  y={fairValue}
                  stroke="#22c55e"
                  strokeWidth={2}
                  strokeDasharray="6 4"
                  label={{
                    value: `Fair Value: $${fairValue.toFixed(0)}`,
                    position: "insideTopRight",
                    fontSize: 11,
                    fill: "#22c55e",
                    fontWeight: 500,
                  }}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-muted-foreground">Price</span>
          </div>
          {fairValue && (
            <div className="flex items-center gap-2">
              <div
                className="w-4 h-0.5 bg-green-500"
                style={{
                  backgroundImage:
                    "repeating-linear-gradient(90deg, #22c55e 0, #22c55e 4px, transparent 4px, transparent 6px)",
                }}
              />
              <span className="text-muted-foreground">
                LSTM Fair Value:{" "}
                <span
                  className={cn(
                    "font-medium",
                    showFairValueLine
                      ? "text-green-600 dark:text-green-400"
                      : "text-muted-foreground"
                  )}
                >
                  ${fairValue.toFixed(2)}
                </span>
                {!showFairValueLine && (
                  <span className="text-xs ml-1 text-muted-foreground/70">
                    (off chart)
                  </span>
                )}
              </span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
