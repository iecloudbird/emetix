/**
 * Position Sizing Tooltip Component
 *
 * Shows personalized position sizing recommendation on hover.
 */
"use client";

import * as React from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info, AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { usePositionSizing } from "@/hooks/useRiskProfile";
import type { PositionSizingRequest } from "@/types/risk-profile";

interface Props {
  ticker: string;
  currentPrice: number;
  marginOfSafety: number;
  beta: number;
  profileId: string | null;
}

export function PositionSizingTooltip({
  ticker,
  currentPrice,
  marginOfSafety,
  beta,
  profileId,
}: Props) {
  const { mutate, data, isPending } = usePositionSizing();

  const handleOpen = () => {
    if (!profileId || data) return;

    const request: PositionSizingRequest = {
      profile_id: profileId,
      ticker,
      current_price: currentPrice,
      margin_of_safety: marginOfSafety,
      beta,
    };

    mutate(request);
  };

  if (!profileId) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <button className="p-1 hover:bg-muted rounded-full transition-colors">
              <Info className="h-4 w-4 text-muted-foreground" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" className="max-w-xs">
            <p className="text-sm">
              Complete your{" "}
              <a
                href="/risk-assessment"
                className="text-primary underline"
              >
                risk assessment
              </a>{" "}
              to see personalized position sizing.
            </p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip onOpenChange={(open) => open && handleOpen()}>
        <TooltipTrigger asChild>
          <button className="p-1 hover:bg-muted rounded-full transition-colors">
            <Info className="h-4 w-4 text-muted-foreground hover:text-primary" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="right" className="w-80 p-0">
          <div className="p-4 space-y-3">
            <div className="flex items-center justify-between border-b pb-2">
              <span className="font-semibold">Position Sizing: {ticker}</span>
              {isPending && <Loader2 className="h-4 w-4 animate-spin" />}
            </div>

            {data ? (
              <>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-muted-foreground">Max Position</div>
                    <div className="font-semibold text-lg">
                      {data.max_position_percent.toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Max Value</div>
                    <div className="font-semibold text-lg">
                      ${data.max_position_value.toLocaleString()}
                    </div>
                  </div>
                  <div className="col-span-2">
                    <div className="text-muted-foreground">Max Shares</div>
                    <div className="font-semibold text-lg">
                      {data.max_shares} shares
                    </div>
                  </div>
                </div>

                {data.risk_factors.length > 0 && (
                  <div className="space-y-1">
                    <div className="flex items-center gap-1 text-sm text-yellow-500">
                      <AlertTriangle className="h-3 w-3" />
                      <span className="font-medium">Risk Factors:</span>
                    </div>
                    <ul className="text-xs text-muted-foreground space-y-1">
                      {data.risk_factors.map((factor, i) => (
                        <li key={i}>• {factor}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="pt-2 border-t">
                  <p className="text-xs text-muted-foreground">
                    {data.recommendation}
                  </p>
                </div>

                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <CheckCircle className="h-3 w-3" />
                  <span>{data.methodology}</span>
                </div>
              </>
            ) : (
              <div className="text-sm text-muted-foreground">
                Calculating personalized position size...
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
