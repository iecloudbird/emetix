/**
 * Stock Screener Page
 *
 * Full analysis universe with search, filter, and sorting capabilities.
 * Shows all 500+ qualified stocks from the 3-stage pipeline.
 */
"use client";

import * as React from "react";
import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  usePipelineClassified,
  usePipelineSummary,
} from "@/hooks/use-pipeline";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
  Filter,
  Clock,
  Info,
  Search,
  RefreshCcw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import {
  ClassificationBadge,
  ScoreBadge,
} from "@/components/pipeline/PipelineBadges";
import type { QualifiedStock } from "@/lib/api";

const ITEMS_PER_PAGE = 25;

// Extract unique sectors from stocks
function getUniqueSectors(stocks: QualifiedStock[]): string[] {
  const sectors = new Set<string>();
  stocks.forEach((s) => {
    if (s.sector) sectors.add(s.sector);
  });
  return Array.from(sectors).sort();
}

export default function ScreenerPage() {
  const router = useRouter();
  const { data: summary } = usePipelineSummary();
  const {
    data: classified,
    isLoading,
    error,
    refetch,
  } = usePipelineClassified();

  // Filters
  const [activeTab, setActiveTab] = useState<"all" | "buy" | "hold" | "watch">(
    "all",
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [sectorFilter, setSectorFilter] = useState<string>("all");
  const [minScore, setMinScore] = useState<number>(0); // Default to 0 (no filter)
  const [currentPage, setCurrentPage] = useState(1);

  // Combine all stocks for "all" tab
  const allStocks = useMemo(() => {
    if (!classified?.classified) return [];
    return [
      ...(classified.classified.buy || []),
      ...(classified.classified.hold || []),
      ...(classified.classified.watch || []),
    ];
  }, [classified]);

  // Get stocks based on active tab
  const tabStocks = useMemo(() => {
    if (activeTab === "all") return allStocks;
    return classified?.classified?.[activeTab] || [];
  }, [activeTab, allStocks, classified]);

  // Get unique sectors for filter
  const sectors = useMemo(() => getUniqueSectors(allStocks), [allStocks]);

  // Filter stocks
  const filteredStocks = useMemo(() => {
    let stocks = tabStocks;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      stocks = stocks.filter(
        (s) =>
          s.ticker.toLowerCase().includes(query) ||
          s.company_name?.toLowerCase().includes(query),
      );
    }

    // Sector filter
    if (sectorFilter !== "all") {
      stocks = stocks.filter((s) => s.sector === sectorFilter);
    }

    // Min score filter (only apply if > 0)
    if (minScore > 0) {
      stocks = stocks.filter((s) => s.composite_score >= minScore);
    }

    return stocks;
  }, [tabStocks, searchQuery, sectorFilter, minScore]);

  // Pagination
  const totalPages = Math.ceil(filteredStocks.length / ITEMS_PER_PAGE);
  const paginatedStocks = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredStocks.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredStocks, currentPage]);

  // Reset page when filters change
  React.useEffect(() => {
    setCurrentPage(1);
  }, [activeTab, searchQuery, sectorFilter, minScore]);

  const handleStockSelect = (ticker: string) => {
    router.push(`/stock/${ticker}`);
  };

  if (error) {
    return (
      <div className="container mx-auto px-4 py-6">
        <Card className="border-red-200 bg-red-50">
          <CardContent className="flex items-center gap-2 py-4">
            <Info className="h-5 w-5 text-red-500" />
            <span className="text-red-700">Failed to load screener data</span>
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Filter className="h-8 w-8 text-purple-500" />
            Stock Screener
          </h1>
          <p className="text-muted-foreground">
            Browse {allStocks.length} qualified stocks with advanced filters
          </p>
        </div>
        <Badge variant="outline" className="w-fit">
          <Clock className="mr-1 h-3 w-3" />
          Last updated:{" "}
          {summary?.pipeline?.last_scan?.completed_at
            ? new Date(summary.pipeline.last_scan.completed_at).toLocaleString()
            : "Never"}
        </Badge>
      </div>

      {/* Filters Bar */}
      <Card>
        <CardContent className="pt-4">
          <div className="flex flex-wrap items-center gap-4">
            {/* Search */}
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search ticker or company..."
                value={searchQuery}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setSearchQuery(e.target.value)
                }
                className="pl-9"
              />
            </div>

            {/* Sector Filter */}
            <Select value={sectorFilter} onValueChange={setSectorFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Sectors" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sectors</SelectItem>
                {sectors.map((sector) => (
                  <SelectItem key={sector} value={sector}>
                    {sector}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Min Score Filter */}
            <Select
              value={minScore.toString()}
              onValueChange={(v) => setMinScore(parseInt(v))}
            >
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="Min Score" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">All Scores</SelectItem>
                <SelectItem value="50">Score ≥ 50</SelectItem>
                <SelectItem value="60">Score ≥ 60</SelectItem>
                <SelectItem value="65">Score ≥ 65</SelectItem>
                <SelectItem value="70">Score ≥ 70</SelectItem>
                <SelectItem value="75">Score ≥ 75</SelectItem>
                <SelectItem value="80">Score ≥ 80</SelectItem>
              </SelectContent>
            </Select>

            {/* Refresh */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => refetch()}
              disabled={isLoading}
            >
              <RefreshCcw
                className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
              />
            </Button>
          </div>

          {/* Results Summary */}
          <div className="mt-3 flex items-center gap-4 text-sm text-muted-foreground">
            <span>
              Showing <strong>{filteredStocks.length}</strong> of{" "}
              <strong>{tabStocks.length}</strong> stocks
            </span>
            <span className="text-muted-foreground/50">|</span>
            <span className="text-green-600">
              Buy: {classified?.classified?.buy?.length || 0}
            </span>
            <span className="text-blue-600">
              Hold: {classified?.classified?.hold?.length || 0}
            </span>
            <span className="text-amber-600">
              Watch: {classified?.classified?.watch?.length || 0}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Compact Methodology Info Bar */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground bg-muted/30 rounded-lg px-4 py-2.5 border">
        <TooltipProvider>
          <div className="flex items-center gap-1">
            <Info className="h-3.5 w-3.5" />
            <span className="font-medium text-foreground">
              Screening Criteria:
            </span>
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help border-b border-dotted border-muted-foreground/50">
                4-Pillar Score (Value, Quality, Growth, Safety)
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p className="text-xs">
                <strong>Value:</strong> MoS, P/E vs sector, FCF yield
                <br />
                <strong>Quality:</strong> FCF ROIC, ROE, margins
                <br />
                <strong>Growth:</strong> Revenue, earnings, LSTM forecast
                <br />
                <strong>Safety:</strong> Beta, volatility, drawdown
              </p>
            </TooltipContent>
          </Tooltip>

          <span className="text-muted-foreground/50">•</span>

          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help text-green-600 font-medium border-b border-dotted border-green-600/50">
                BUY: MoS≥20% + Score≥70
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs">
                Strong margin of safety with high quality score
              </p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help text-blue-600 font-medium border-b border-dotted border-blue-600/50">
                HOLD: MoS -10%~20%
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs">
                Fair valued with score ≥70, monitor for better entry
              </p>
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help text-amber-600 font-medium border-b border-dotted border-amber-600/50">
                WATCH: Qualified
              </span>
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-xs">
                Passed screening, needs further analysis or better price
              </p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Main Table with Tabs */}
      <Card>
        <CardHeader className="pb-2">
          <Tabs
            value={activeTab}
            onValueChange={(v) =>
              setActiveTab(v as "all" | "buy" | "hold" | "watch")
            }
          >
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="all">All ({allStocks.length})</TabsTrigger>
              <TabsTrigger
                value="buy"
                className="data-[state=active]:bg-green-100 dark:data-[state=active]:bg-green-900/30"
              >
                Buy ({classified?.classified?.buy?.length || 0})
              </TabsTrigger>
              <TabsTrigger
                value="hold"
                className="data-[state=active]:bg-blue-100 dark:data-[state=active]:bg-blue-900/30"
              >
                Hold ({classified?.classified?.hold?.length || 0})
              </TabsTrigger>
              <TabsTrigger
                value="watch"
                className="data-[state=active]:bg-amber-100 dark:data-[state=active]:bg-amber-900/30"
              >
                Watch ({classified?.classified?.watch?.length || 0})
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
          ) : paginatedStocks.length === 0 ? (
            <div className="flex h-32 items-center justify-center text-muted-foreground">
              No stocks match your filters
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[100px]">Ticker</TableHead>
                      <TableHead>Company</TableHead>
                      <TableHead>Sector</TableHead>
                      <TableHead className="text-center">Score</TableHead>
                      <TableHead className="text-center">Class</TableHead>
                      <TableHead className="text-right">Price</TableHead>
                      <TableHead className="text-right">Fair Value</TableHead>
                      <TableHead className="text-right">MoS</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paginatedStocks.map((stock) => (
                      <TableRow
                        key={stock.ticker}
                        className="cursor-pointer hover:bg-muted/50"
                        onClick={() => handleStockSelect(stock.ticker)}
                      >
                        <TableCell className="font-medium">
                          {stock.ticker}
                        </TableCell>
                        <TableCell>
                          <div className="max-w-[200px] truncate">
                            {stock.company_name}
                          </div>
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
                          <ClassificationBadge
                            classification={stock.classification}
                          />
                        </TableCell>
                        <TableCell className="text-right">
                          ${stock.current_price?.toFixed(2) || "N/A"}
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
                                ? "text-green-600 dark:text-green-400"
                                : "text-red-600 dark:text-red-400"
                            }
                          >
                            {stock.margin_of_safety !== null
                              ? `${stock.margin_of_safety > 0 ? "+" : ""}${stock.margin_of_safety.toFixed(1)}%`
                              : "N/A"}
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
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

      {/* Footer */}
      <div className="text-center text-sm text-muted-foreground pt-8 mt-8 border-t">
        <p>© 2026 Emetix - A00303759 Final Year Project</p>
        <p className="mt-1">Built with care for retail investors</p>
      </div>
    </div>
  );
}
