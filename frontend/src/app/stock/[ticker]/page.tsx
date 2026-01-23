/**
 * Stock Detail Page
 *
 * Deep dive into individual stock with:
 * - Price charts (1Y/5Y)
 * - Key metrics and ratios
 * - LSTM fair value analysis
 * - AI-generated justification
 * - Profile suitability assessment (Phase 2)
 */
"use client";

import { use } from "react";
import { useStock, useCharts } from "@/hooks/use-stocks";
import { useLocalRiskProfile } from "@/hooks/useRiskProfile";
import { PriceChart } from "@/components/charts/PriceChart";
import { ProfileSuitabilityCard } from "@/components/profile/ProfileSuitabilityCard";
import { AIAnalysisPanel } from "@/components/stocks/AIAnalysisPanel";
import { MultiAgentAnalysisPanel } from "@/components/stocks/MultiAgentAnalysisPanel";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  TrendingUp,
  TrendingDown,
  ArrowLeft,
  DollarSign,
  BarChart3,
  Shield,
  Target,
  Percent,
} from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";

interface StockPageProps {
  params: Promise<{ ticker: string }>;
}

export default function StockPage({ params }: StockPageProps) {
  const { ticker } = use(params);
  const {
    data: stockData,
    isLoading: stockLoading,
    error: stockError,
  } = useStock(ticker);
  const { data: chartData, isLoading: chartLoading } = useCharts(ticker);
  const { rawProfile: profile } = useLocalRiskProfile();

  if (stockLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-[400px]" />
      </div>
    );
  }

  if (stockError || !stockData?.data) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-red-500">Stock Not Found</h1>
        <p className="text-muted-foreground mt-2">
          Could not load data for {ticker.toUpperCase()}
        </p>
        <Link href="/">
          <Button className="mt-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </Link>
      </div>
    );
  }

  const stock = stockData.data;
  const upside = stock.margin_of_safety ?? 0;
  const isUndervalued = upside > 0;

  const getRecommendationColor = (rec: string) => {
    const recUpper = rec?.toUpperCase() || "";
    if (recUpper.includes("BUY") || recUpper.includes("STRONG")) {
      return "bg-green-500";
    } else if (recUpper.includes("SELL") || recUpper.includes("AVOID")) {
      return "bg-red-500";
    }
    return "bg-yellow-500";
  };

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <Link href="/">
        <Button variant="ghost" size="sm">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
      </Link>

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-4xl font-bold">{stock.ticker}</h1>
            <Badge
              className={getRecommendationColor(stock.recommendation)}
              variant="default"
            >
              {stock.recommendation}
            </Badge>
            {stock.risk_level && (
              <Badge variant="outline">{stock.risk_level} Risk</Badge>
            )}
          </div>
          <p className="text-xl text-muted-foreground">{stock.company_name}</p>
          <p className="text-sm text-muted-foreground">{stock.sector}</p>
        </div>

        <div className="text-right">
          <p className="text-4xl font-bold">
            ${stock.current_price.toFixed(2)}
          </p>
          <div
            className={cn(
              "flex items-center justify-end gap-2 text-lg",
              isUndervalued ? "text-green-600" : "text-red-600"
            )}
          >
            {isUndervalued ? (
              <TrendingUp className="w-5 h-5" />
            ) : (
              <TrendingDown className="w-5 h-5" />
            )}
            <span>
              {isUndervalued ? "+" : ""}
              {upside.toFixed(1)}% to fair value
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <MetricCard
          icon={Target}
          label="Fair Value"
          value={
            stock.lstm_fair_value
              ? `$${stock.lstm_fair_value.toFixed(2)}`
              : "N/A"
          }
          sublabel="LSTM-DCF"
        />
        <MetricCard
          icon={BarChart3}
          label="Valuation Score"
          value={stock.valuation_score.toFixed(0)}
          sublabel="0-100 scale"
          valueColor={
            stock.valuation_score >= 70 ? "text-green-600" : undefined
          }
        />
        <MetricCard
          icon={Percent}
          label="P/E Ratio"
          value={stock.pe_ratio?.toFixed(1) ?? "N/A"}
          sublabel="Trailing"
        />
        <MetricCard
          icon={DollarSign}
          label="P/B Ratio"
          value={stock.pb_ratio?.toFixed(2) ?? "N/A"}
          sublabel="Price/Book"
        />
        <MetricCard
          icon={Shield}
          label="Beta"
          value={stock.beta?.toFixed(2) ?? "N/A"}
          sublabel="Market Risk"
          valueColor={
            stock.beta && stock.beta < 1 ? "text-green-600" : undefined
          }
        />
        <MetricCard
          icon={Percent}
          label="Dividend Yield"
          value={
            stock.dividend_yield
              ? `${(stock.dividend_yield * 100).toFixed(2)}%`
              : "N/A"
          }
          sublabel="Annual"
        />
      </div>

      {/* Profile Suitability Card - Phase 2 */}
      <ProfileSuitabilityCard stock={stock} profile={profile} />

      {/* Tabs for Chart & Analysis */}
      <Tabs defaultValue="chart" className="space-y-4">
        <TabsList>
          <TabsTrigger value="chart">Price Chart</TabsTrigger>
          <TabsTrigger value="analysis">AI Analysis</TabsTrigger>
          <TabsTrigger value="deep">Deep Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="chart">
          {chartLoading ? (
            <Skeleton className="h-[400px]" />
          ) : chartData?.data?.charts ? (
            <PriceChart
              data1Y={chartData.data.charts.price_1y}
              data5Y={chartData.data.charts.price_5y}
              currentPrice={stock.current_price}
              fairValue={stock.lstm_fair_value ?? undefined}
              ticker={stock.ticker}
            />
          ) : (
            <Card>
              <CardContent className="py-8 text-center text-muted-foreground">
                Chart data not available
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="analysis">
          <AIAnalysisPanel ticker={stock.ticker} />
        </TabsContent>

        <TabsContent value="deep">
          <MultiAgentAnalysisPanel ticker={stock.ticker} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Metric Card Component (local, reusable)
interface MetricCardProps {
  icon: React.ElementType;
  label: string;
  value: string;
  sublabel?: string;
  valueColor?: string;
}

function MetricCard({
  icon: Icon,
  label,
  value,
  sublabel,
  valueColor,
}: MetricCardProps) {
  return (
    <Card className="py-1">
      <CardContent className="pt-2 pb-2 px-3">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Icon className="w-3.5 h-3.5" />
          <span className="text-xs">{label}</span>
        </div>
        <p className={cn("text-lg font-bold mt-0.5", valueColor)}>{value}</p>
        {sublabel && (
          <p className="text-[10px] text-muted-foreground leading-tight">
            {sublabel}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
