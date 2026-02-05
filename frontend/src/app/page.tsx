/**
 * Top Picks Page (Home)
 *
 * Curated stock recommendations with analyst-like justifications.
 * Integrates with Personal Risk Profile for suitability badges.
 *
 * Data Source: /api/pipeline/curated (Stage 3 curated watchlist)
 */
"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useCuratedWatchlist } from "@/hooks/use-stocks";
import { useLocalRiskProfile } from "@/hooks/useRiskProfile";
import { usePipelineSummary } from "@/hooks/use-pipeline";
import { RiskAssessmentModal } from "@/components/risk-profile/RiskAssessmentModal";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Shield,
  Award,
  Clock,
  User,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Sparkles,
  Target,
  Eye,
  Filter,
  ChevronLeft,
  ChevronRight,
  Trophy,
} from "lucide-react";
import type { CuratedStock } from "@/lib/api";

// Risk suitability calculation based on user's profile
function calculateSuitability(
  stock: CuratedStock,
  profile: {
    profileType: string;
    betaRange: { min: number; max: number };
    requiredMoS: number;
  } | null,
): {
  suitable: boolean;
  reason: string;
  level: "good" | "caution" | "warning";
} {
  if (!profile) {
    return {
      suitable: true,
      reason: "Complete risk assessment for personalized suitability",
      level: "caution",
    };
  }

  const score = stock.composite_score;
  const conviction = stock.conviction;
  const classification = stock.classification;
  const mos = stock.margin_of_safety ?? 0;
  const subCategory = stock.watch_sub_category || "";

  // Conservative profiles - need high conviction + good MoS + high score
  if (profile.profileType === "conservative") {
    // Excellent fit: Strong conviction OR high MoS with good score
    if ((conviction === "Strong" || mos >= 30) && score >= 75) {
      return {
        suitable: true,
        reason: "High conviction & quality - excellent fit",
        level: "good",
      };
    }
    // Good fit: Moderate conviction with decent score
    if (
      conviction === "Moderate" &&
      score >= 70 &&
      mos >= profile.requiredMoS
    ) {
      return {
        suitable: true,
        reason: "Good quality, meets MoS requirement",
        level: "good",
      };
    }
    // Watch stocks are too speculative
    if (classification === "Watch" || subCategory === "Speculative") {
      return {
        suitable: false,
        reason: "Too speculative for conservative profile",
        level: "warning",
      };
    }
    // Negative MoS = overvalued
    if (mos < 0) {
      return {
        suitable: false,
        reason: `Overvalued (MoS ${mos.toFixed(0)}%) - wait for better entry`,
        level: "warning",
      };
    }
    // Default caution
    return {
      suitable: true,
      reason: "Consider smaller position",
      level: "caution",
    };
  }

  // Moderate profiles - balanced approach
  if (profile.profileType === "moderate") {
    // Good fit: Score >= 65 with positive or near-zero MoS
    if (score >= 65 && mos >= -20) {
      return {
        suitable: true,
        reason: "Matches moderate risk tolerance",
        level: "good",
      };
    }
    // Speculative watch needs caution
    if (subCategory === "Speculative" || classification === "Watch") {
      return {
        suitable: true,
        reason: "Higher risk - limit position size",
        level: "caution",
      };
    }
    // Significantly overvalued
    if (mos < -50) {
      return {
        suitable: false,
        reason: `Significantly overvalued (MoS ${mos.toFixed(0)}%)`,
        level: "warning",
      };
    }
    return {
      suitable: true,
      reason: "Within risk tolerance",
      level: "caution",
    };
  }

  // Aggressive profiles - most stocks are suitable
  if (profile.profileType === "aggressive") {
    if (score >= 55) {
      return {
        suitable: true,
        reason: "Within aggressive risk tolerance",
        level: "good",
      };
    }
    return {
      suitable: true,
      reason: "Higher risk opportunity",
      level: "caution",
    };
  }

  // Default for unknown profile types
  return {
    suitable: true,
    reason: "Review against your criteria",
    level: "caution",
  };
}

// Conviction badge component
function ConvictionBadge({ conviction }: { conviction: string }) {
  const variants: Record<string, { bg: string; text: string }> = {
    Strong: {
      bg: "bg-green-100 dark:bg-green-900/30",
      text: "text-green-700 dark:text-green-400",
    },
    Moderate: {
      bg: "bg-blue-100 dark:bg-blue-900/30",
      text: "text-blue-700 dark:text-blue-400",
    },
    Monitor: {
      bg: "bg-amber-100 dark:bg-amber-900/30",
      text: "text-amber-700 dark:text-amber-400",
    },
  };

  const variant = variants[conviction] || variants.Monitor;

  return (
    <Badge className={`${variant.bg} ${variant.text} text-xs font-medium`}>
      {conviction}
    </Badge>
  );
}

// Classification badge
function ClassBadge({ classification }: { classification: string }) {
  const colors: Record<string, string> = {
    Buy: "bg-green-500 text-white",
    Hold: "bg-blue-500 text-white",
    Watch: "bg-amber-500 text-white",
  };

  return (
    <Badge
      className={`${colors[classification] || "bg-gray-500 text-white"} text-xs`}
    >
      {classification}
    </Badge>
  );
}

// Score badge component
function ScoreBadge({ score }: { score: number }) {
  const getColor = () => {
    if (score >= 80)
      return "text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/30";
    if (score >= 70)
      return "text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30";
    if (score >= 60)
      return "text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30";
    return "text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30";
  };

  return (
    <Badge variant="outline" className={`font-bold ${getColor()}`}>
      {score.toFixed(0)}
    </Badge>
  );
}

// Suitability indicator
function SuitabilityIndicator({
  suitability,
}: {
  suitability: {
    suitable: boolean;
    reason: string;
    level: "good" | "caution" | "warning";
  };
}) {
  const icons = {
    good: <CheckCircle2 className="h-4 w-4 text-green-500" />,
    caution: <AlertCircle className="h-4 w-4 text-amber-500" />,
    warning: <XCircle className="h-4 w-4 text-red-500" />,
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger>{icons[suitability.level]}</TooltipTrigger>
        <TooltipContent>
          <p className="text-xs max-w-[200px]">{suitability.reason}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const ITEMS_PER_PAGE = 15;

export default function TopPicksPage() {
  const router = useRouter();
  const [showRiskModal, setShowRiskModal] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState<"buy" | "hold">("buy");
  const [currentPage, setCurrentPage] = React.useState(1);
  const {
    profile: riskProfile,
    hasProfile,
    refresh: refreshProfile,
  } = useLocalRiskProfile();
  const { data: summary } = usePipelineSummary();
  const {
    data: curatedData,
    isLoading,
    error,
    refetch,
  } = useCuratedWatchlist();

  // Extract watchlists from response - handle both direct format and nested format
  const strongBuy = React.useMemo(() => {
    if (!curatedData) return [];
    // Check if it's in the watchlists format or direct format
    if ("watchlists" in curatedData && curatedData.watchlists) {
      return (
        (curatedData.watchlists as { strong_buy?: CuratedStock[] })
          .strong_buy || []
      );
    }
    return curatedData.strong_buy || [];
  }, [curatedData]);

  const moderateBuy = React.useMemo(() => {
    if (!curatedData) return [];
    if ("watchlists" in curatedData && curatedData.watchlists) {
      return (
        (curatedData.watchlists as { moderate_buy?: CuratedStock[] })
          .moderate_buy || []
      );
    }
    return curatedData.moderate_buy || [];
  }, [curatedData]);

  const hold = React.useMemo(() => {
    if (!curatedData) return [];
    if ("watchlists" in curatedData && curatedData.watchlists) {
      return (curatedData.watchlists as { hold?: CuratedStock[] }).hold || [];
    }
    return curatedData.hold || [];
  }, [curatedData]);

  // All buy stocks combined
  const allBuy = [...strongBuy, ...moderateBuy];

  // Get stocks based on active tab
  const tabStocks = activeTab === "buy" ? allBuy : hold;

  // Pagination
  const totalPages = Math.ceil(tabStocks.length / ITEMS_PER_PAGE);
  const paginatedStocks = React.useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    return tabStocks.slice(start, start + ITEMS_PER_PAGE);
  }, [tabStocks, currentPage]);

  // Reset page when tab changes
  React.useEffect(() => {
    setCurrentPage(1);
  }, [activeTab]);

  // Stats
  const totalCurated = allBuy.length + hold.length;
  const avgScore =
    totalCurated > 0
      ? [...allBuy, ...hold].reduce((sum, s) => sum + s.composite_score, 0) /
        totalCurated
      : 0;

  const handleStockSelect = (ticker: string) => {
    router.push(`/stock/${ticker}`);
  };

  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      {/* Risk Assessment Modal */}
      <RiskAssessmentModal
        open={showRiskModal}
        onOpenChange={(open) => {
          setShowRiskModal(open);
          if (!open) refreshProfile();
        }}
        forceShow={showRiskModal}
      />

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Trophy className="h-8 w-8 text-amber-500" />
            Top Picks
          </h1>
          <p className="text-muted-foreground">
            Curated recommendations with analyst-like justifications
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Risk Profile Status */}
          {hasProfile && riskProfile ? (
            <Badge
              variant="outline"
              className="flex items-center gap-1.5 py-1.5 px-3 cursor-pointer hover:bg-muted"
              onClick={() => setShowRiskModal(true)}
            >
              <Shield className="h-3.5 w-3.5" />
              {riskProfile.profileType.charAt(0).toUpperCase() +
                riskProfile.profileType.slice(1).replace("_", " ")}{" "}
              Profile
            </Badge>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={() => setShowRiskModal(true)}
              className="flex items-center gap-1.5"
            >
              <User className="h-3.5 w-3.5" />
              Take Risk Assessment
            </Button>
          )}

          <Badge variant="outline" className="w-fit">
            <Clock className="mr-1 h-3 w-3" />
            {curatedData?.metadata?.generated_at
              ? new Date(curatedData.metadata.generated_at as string).toLocaleDateString()
              : summary?.pipeline?.last_scan?.completed_at
                ? new Date(summary.pipeline.last_scan.completed_at).toLocaleDateString()
                : isLoading
                  ? "Loading..."
                  : new Date().toLocaleDateString()}
          </Badge>
        </div>
      </div>

      {/* Risk Profile CTA for non-profiled users */}
      {!hasProfile && (
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 border-blue-200 dark:border-blue-800">
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-full">
                <Shield className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="font-medium">Personalize Your Recommendations</p>
                <p className="text-sm text-muted-foreground">
                  Complete a quick risk assessment to see which stocks match
                  your profile
                </p>
              </div>
            </div>
            <Button onClick={() => setShowRiskModal(true)}>
              Get Started
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Buy Signals</p>
                <p className="text-2xl font-bold text-green-600">
                  {allBuy.length}
                </p>
              </div>
              <Target className="h-8 w-8 text-green-500/20" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Hold & Monitor</p>
                <p className="text-2xl font-bold text-blue-600">
                  {hold.length}
                </p>
              </div>
              <Eye className="h-8 w-8 text-blue-500/20" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Score</p>
                <p className="text-2xl font-bold">{avgScore.toFixed(1)}</p>
              </div>
              <Award className="h-8 w-8 text-amber-500/20" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Universe</p>
                <p className="text-2xl font-bold">
                  {summary?.pipeline?.qualified_total || "500+"}
                </p>
              </div>
              <Sparkles className="h-8 w-8 text-purple-500/20" />
            </div>
            <Link
              href="/screener"
              className="text-xs text-blue-600 hover:underline flex items-center gap-1 mt-1"
            >
              View all in screener <ArrowRight className="h-3 w-3" />
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* Main Table with Tabs */}
      <Card>
        <CardHeader className="pb-2">
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as "buy" | "hold")}
          >
            <TabsList className="grid w-full grid-cols-2 lg:w-[300px]">
              <TabsTrigger
                value="buy"
                className="gap-2 data-[state=active]:bg-green-100 dark:data-[state=active]:bg-green-900/30"
              >
                <Target className="h-4 w-4" />
                Buy ({allBuy.length})
              </TabsTrigger>
              <TabsTrigger
                value="hold"
                className="gap-2 data-[state=active]:bg-blue-100 dark:data-[state=active]:bg-blue-900/30"
              >
                <Eye className="h-4 w-4" />
                Hold ({hold.length})
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : error ? (
            <div className="flex h-32 items-center justify-center gap-2 text-muted-foreground">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <span>Failed to load curated watchlist</span>
              <Button variant="outline" size="sm" onClick={() => refetch()}>
                Retry
              </Button>
            </div>
          ) : paginatedStocks.length === 0 ? (
            <div className="flex h-32 items-center justify-center text-muted-foreground">
              No stocks in this category
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[100px]">Ticker</TableHead>
                      <TableHead>Sector</TableHead>
                      <TableHead className="text-center">Score</TableHead>
                      <TableHead className="text-center">Class</TableHead>
                      <TableHead className="text-center">Conviction</TableHead>
                      <TableHead className="text-right">Fair Value</TableHead>
                      <TableHead className="text-right">MoS</TableHead>
                      {hasProfile && (
                        <TableHead className="text-center">Fit</TableHead>
                      )}
                      <TableHead className="max-w-[250px]">
                        Justification
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paginatedStocks.map((stock) => {
                      const suitability = calculateSuitability(
                        stock,
                        riskProfile,
                      );
                      return (
                        <TableRow
                          key={stock.ticker}
                          className="cursor-pointer hover:bg-muted/50"
                          onClick={() => handleStockSelect(stock.ticker)}
                        >
                          <TableCell className="font-bold">
                            {stock.ticker}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline" className="text-xs">
                              {stock.sector}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-center">
                            <ScoreBadge score={stock.composite_score} />
                          </TableCell>
                          <TableCell className="text-center">
                            <ClassBadge classification={stock.classification} />
                          </TableCell>
                          <TableCell className="text-center">
                            <ConvictionBadge conviction={stock.conviction} />
                          </TableCell>
                          <TableCell className="text-right">
                            <span className="text-blue-600 dark:text-blue-400 font-medium">
                              ${stock.fair_value?.toFixed(2) || "N/A"}
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            <span
                              className={
                                (stock.margin_of_safety ?? 0) > 0
                                  ? "text-green-600 dark:text-green-400 font-medium"
                                  : "text-red-600 dark:text-red-400 font-medium"
                              }
                            >
                              {stock.margin_of_safety !== undefined &&
                              stock.margin_of_safety !== null
                                ? `${stock.margin_of_safety > 0 ? "+" : ""}${stock.margin_of_safety.toFixed(1)}%`
                                : "N/A"}
                            </span>
                          </TableCell>
                          {hasProfile && (
                            <TableCell className="text-center">
                              <SuitabilityIndicator suitability={suitability} />
                            </TableCell>
                          )}
                          <TableCell className="max-w-[250px]">
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="text-xs text-muted-foreground truncate block">
                                    {stock.justification?.short ||
                                      "Quality fundamentals"}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent
                                  side="left"
                                  className="max-w-xs"
                                >
                                  <p className="text-xs">
                                    {stock.justification?.short ||
                                      "Quality stock with strong fundamentals"}
                                  </p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t">
                  <div className="text-sm text-muted-foreground">
                    Page {currentPage} of {totalPages}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setCurrentPage((p) => Math.min(totalPages, p + 1))
                      }
                      disabled={currentPage === totalPages}
                    >
                      Next
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* View All Link */}
      <div className="flex justify-center pt-4">
        <Link href="/screener">
          <Button variant="outline" className="gap-2">
            <Filter className="h-4 w-4" />
            View All Qualified Stocks in Screener
            <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>

      {/* Footer */}
      <div className="text-center text-sm text-muted-foreground pt-8 mt-8 border-t">
        <p>© 2026 Emetix - A00303759 Final Year Project</p>
        <p className="mt-1">Built with care for retail investors</p>
      </div>
    </div>
  );
}
