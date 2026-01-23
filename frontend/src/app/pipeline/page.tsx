/**
 * Pipeline Page - Phase 3 Quality Screening Pipeline
 *
 * Shows the 3-stage automated screening results:
 * - Buy/Hold/Watch classified stocks
 * - 4-pillar scores visualization
 * - Momentum indicators
 */
"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { PipelineDashboard } from "@/components/pipeline/PipelineDashboard";
import { usePipelineSummary } from "@/hooks/use-pipeline";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Clock, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function PipelinePage() {
  const router = useRouter();
  const { data: summary } = usePipelineSummary();

  const handleStockSelect = (ticker: string) => {
    router.push(`/stock/${ticker}`);
  };

  return (
    <div className="container mx-auto px-4 py-6 space-y-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Sparkles className="h-8 w-8 text-purple-500" />
            Quality Pipeline
          </h1>
          <p className="text-muted-foreground">
            Automated screening with 4-pillar scoring
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

      {/* Main Dashboard */}
      <PipelineDashboard onStockSelect={handleStockSelect} />
    </div>
  );
}
