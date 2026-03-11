/**
 * StockInfographic Component
 *
 * Analyst-style visual cards for stock thesis building.
 * Revenue bars + growth overlay, margin trends, multiple compression panels,
 * beat-down detection badge, and performance summary.
 *
 * Uses Recharts (already in project).
 */
"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
  Area,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchStockInfographic,
  type InfographicResponse,
  type MultipleData,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Zap,
  BarChart3,
} from "lucide-react";

interface StockInfographicProps {
  ticker: string;
}

// Format large numbers into readable format
function fmtNum(n: number | null | undefined): string {
  if (n == null) return "N/A";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n / 1e12).toFixed(1)}T`;
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(0)}M`;
  return `$${n.toLocaleString()}`;
}

function fmtPct(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(1)}%`;
}

// Custom tooltip shared across charts
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-popover/95 backdrop-blur-sm border border-border rounded-lg shadow-lg px-3 py-2 text-xs">
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2">
          <div
            className="w-2 h-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-semibold">
            {typeof p.value === "number"
              ? p.name.includes("%") ||
                p.name.includes("Margin") ||
                p.name.includes("Growth")
                ? `${p.value.toFixed(1)}%`
                : fmtNum(p.value)
              : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function StockInfographic({ ticker }: StockInfographicProps) {
  const [data, setData] = useState<InfographicResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchStockInfographic(ticker)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[...Array(4)].map((_, i) => (
            <Skeleton key={i} className="h-[280px]" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          {error || "No infographic data available"}
        </CardContent>
      </Card>
    );
  }

  const { overview, beat_down } = data;

  return (
    <div className="space-y-4">
      {/* Header with beat-down badge */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-lg font-bold">
            {overview.name} ({overview.ticker})
          </h3>
          <p className="text-sm text-muted-foreground">
            {overview.sector} · {overview.industry}
          </p>
        </div>
        <div className="flex gap-2">
          {beat_down.is_beat_down && (
            <Badge variant="destructive" className="flex items-center gap-1">
              <Zap className="w-3 h-3" />
              Beat-Down Opportunity
            </Badge>
          )}
          {data.revenue_cagr && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              {data.revenue_cagr.cagr}% CAGR ({data.revenue_cagr.years}yr)
            </Badge>
          )}
        </div>
      </div>

      {/* Beat-down signals */}
      {beat_down.signals.length > 0 && (
        <div
          className={cn(
            "rounded-lg border px-4 py-3 text-sm",
            beat_down.is_beat_down
              ? "border-amber-500/50 bg-amber-500/10"
              : "border-border bg-muted/30",
          )}
        >
          <div className="flex items-center gap-2 font-medium mb-1">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            {beat_down.is_beat_down
              ? "Beat-Down Signals Detected"
              : "Partial Signals"}
          </div>
          <div className="flex flex-wrap gap-2">
            {beat_down.signals.map((s, i) => (
              <Badge key={i} variant="outline" className="text-xs">
                {s}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Main chart grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 1. Annual Revenue + YoY Growth */}
        <RevenueGrowthChart data={data} />

        {/* 2. Margin Trends */}
        <MarginTrendsChart data={data} />

        {/* 3. Multiple Compression */}
        <MultiplesCompressionChart multiples={data.multiples} />

        {/* 4. Performance + Net Income */}
        <PerformanceCard data={data} />
      </div>
    </div>
  );
}

// ============================================================================
// SUB-CHARTS
// ============================================================================

/** Revenue bars + YoY growth line overlay (Fiscal.ai style) */
function RevenueGrowthChart({ data }: { data: InfographicResponse }) {
  // Prefer annual data; fall back to quarterly
  const useAnnual = data.annual_revenue.length >= 3;
  const chartData = useAnnual
    ? data.annual_revenue.map((r) => ({
        period: r.year,
        revenue: r.revenue,
        growth: r.yoy_growth ?? null,
      }))
    : data.quarterly_revenue.map((r) => ({
        period: r.period,
        revenue: r.revenue,
        growth: r.yoy_growth ?? null,
      }));

  if (!chartData.length) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Revenue</CardTitle>
        </CardHeader>
        <CardContent className="h-[240px] flex items-center justify-center text-muted-foreground text-sm">
          No revenue data
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-1.5">
            <BarChart3 className="w-4 h-4" />
            {useAnnual ? "Annual" : "Quarterly"} Revenue
          </CardTitle>
          {data.revenue_cagr && (
            <span className="text-xs text-muted-foreground">
              CAGR {data.revenue_cagr.cagr}%
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={chartData}
              margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(128,128,128,0.15)"
                vertical={false}
              />
              <XAxis
                dataKey="period"
                tick={{ fontSize: 10, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                yAxisId="rev"
                tickFormatter={(v: number) => fmtNum(v).replace("$", "")}
                tick={{ fontSize: 10, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
                width={55}
              />
              <YAxis
                yAxisId="growth"
                orientation="right"
                tickFormatter={(v: number) => `${v}%`}
                tick={{ fontSize: 10, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
                width={45}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
              <Bar
                yAxisId="rev"
                dataKey="revenue"
                name="Revenue"
                fill="#3b82f6"
                radius={[3, 3, 0, 0]}
                maxBarSize={40}
              />
              <Line
                yAxisId="growth"
                dataKey="growth"
                name="YoY Growth %"
                stroke="#22c55e"
                strokeWidth={2}
                dot={{ r: 3, fill: "#22c55e" }}
                connectNulls
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

/** Margin trend lines (gross, operating, net) */
function MarginTrendsChart({ data }: { data: InfographicResponse }) {
  const chartData = data.margin_trends;
  if (!chartData.length) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Margin Trends</CardTitle>
        </CardHeader>
        <CardContent className="h-[240px] flex items-center justify-center text-muted-foreground text-sm">
          No margin data
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Margin Trends</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(128,128,128,0.15)"
                vertical={false}
              />
              <XAxis
                dataKey="period"
                tick={{ fontSize: 10, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                tickFormatter={(v: number) => `${v}%`}
                tick={{ fontSize: 10, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
                width={45}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
              {chartData.some((d) => d.gross_margin != null) && (
                <Line
                  dataKey="gross_margin"
                  name="Gross Margin %"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                />
              )}
              {chartData.some((d) => d.operating_margin != null) && (
                <Line
                  dataKey="operating_margin"
                  name="Op Margin %"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                />
              )}
              {chartData.some((d) => d.net_margin != null) && (
                <Line
                  dataKey="net_margin"
                  name="Net Margin %"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

/** Four-panel multiples compression (P/S, P/FCF, P/B, P/E) */
function MultiplesCompressionChart({
  multiples,
}: {
  multiples: Record<string, MultipleData>;
}) {
  const entries = Object.entries(multiples).slice(0, 6);
  if (!entries.length) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Valuation Multiples</CardTitle>
        </CardHeader>
        <CardContent className="h-[240px] flex items-center justify-center text-muted-foreground text-sm">
          No multiples data
        </CardContent>
      </Card>
    );
  }

  const chartData = entries.map(([key, m]) => ({
    name: m.label,
    current: m.current,
    peak: m.peak_est ?? m.current,
    compression: m.compression_pct ?? 0,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center justify-between">
          <span>Multiple Compression</span>
          <span className="text-xs font-normal text-muted-foreground">
            Current vs 52-week peak est.
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(128,128,128,0.15)"
                horizontal={false}
              />
              <XAxis
                type="number"
                tick={{ fontSize: 10, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 11, fill: "currentColor" }}
                tickLine={false}
                axisLine={false}
                width={55}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend iconSize={8} wrapperStyle={{ fontSize: 10 }} />
              <Bar
                dataKey="peak"
                name="Peak Est."
                fill="#94a3b8"
                radius={[0, 3, 3, 0]}
                maxBarSize={16}
              />
              <Bar
                dataKey="current"
                name="Current"
                radius={[0, 3, 3, 0]}
                maxBarSize={16}
              >
                {chartData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={
                      entry.compression >= 30
                        ? "#22c55e"
                        : entry.compression >= 15
                          ? "#f59e0b"
                          : "#3b82f6"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        {/* Compression badges below chart */}
        <div className="flex flex-wrap gap-1.5 mt-2">
          {chartData
            .filter((d) => d.compression > 0)
            .map((d, i) => (
              <Badge
                key={i}
                variant={d.compression >= 30 ? "default" : "secondary"}
                className={cn(
                  "text-[10px]",
                  d.compression >= 30 && "bg-green-600 hover:bg-green-700",
                )}
              >
                {d.name} −{d.compression.toFixed(0)}%
              </Badge>
            ))}
        </div>
      </CardContent>
    </Card>
  );
}

/** Performance returns + net income summary */
function PerformanceCard({ data }: { data: InfographicResponse }) {
  const perf = data.performance;
  const periods = [
    { key: "1d", label: "1D" },
    { key: "1w", label: "1W" },
    { key: "1m", label: "1M" },
    { key: "3m", label: "3M" },
    { key: "6m", label: "6M" },
    { key: "1y", label: "1Y" },
  ] as const;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Performance & Income</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Performance bars */}
        <div className="space-y-1.5">
          {periods.map(({ key, label }) => {
            const val = perf[key] as number | null | undefined;
            if (val == null) return null;
            const isPositive = val >= 0;
            const width = Math.min(Math.abs(val) * 2, 100);
            return (
              <div key={key} className="flex items-center gap-2 text-xs">
                <span className="w-6 text-muted-foreground">{label}</span>
                <div className="flex-1 h-4 bg-muted/30 rounded-sm overflow-hidden relative">
                  <div
                    className={cn(
                      "h-full rounded-sm",
                      isPositive ? "bg-green-500/70" : "bg-red-500/70",
                    )}
                    style={{ width: `${width}%` }}
                  />
                </div>
                <span
                  className={cn(
                    "w-16 text-right font-medium",
                    isPositive ? "text-green-600" : "text-red-600",
                  )}
                >
                  {fmtPct(val)}
                </span>
              </div>
            );
          })}
        </div>

        {/* 52-week range */}
        {perf["52w_low"] != null && perf["52w_high"] != null && (
          <div className="text-xs text-muted-foreground flex justify-between border-t pt-2">
            <span>52W Low: ${(perf["52w_low"] as number).toFixed(2)}</span>
            <span>52W High: ${(perf["52w_high"] as number).toFixed(2)}</span>
          </div>
        )}

        {/* Net income mini chart */}
        {data.quarterly_net_income.length > 0 && (
          <div>
            <p className="text-xs text-muted-foreground mb-1">
              Quarterly Net Income
            </p>
            <div className="h-[80px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={data.quarterly_net_income}
                  margin={{ top: 2, right: 0, bottom: 0, left: 0 }}
                >
                  <XAxis
                    dataKey="period"
                    tick={{ fontSize: 8 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar
                    dataKey="net_income"
                    name="Net Income"
                    radius={[2, 2, 0, 0]}
                    maxBarSize={24}
                  >
                    {data.quarterly_net_income.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.net_income >= 0 ? "#22c55e" : "#ef4444"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
