/**
 * Pipeline Badge Components
 *
 * Visual badges for pipeline data display:
 * - TriggerBadge: Shows attention trigger types (52W Drop, Quality Growth, Deep Value)
 * - ClassificationBadge: Shows Buy/Hold/Watch classification
 * - MomentumIndicator: Shows 50MA/200MA accumulation status
 */
"use client";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TrendingDown, Sparkles, DollarSign, ArrowUpDown } from "lucide-react";

// ============================================================
// Trigger Badge
// ============================================================

interface TriggerBadgeProps {
  trigger: string;
  showTooltip?: boolean;
}

const triggerConfig: Record<
  string,
  { label: string; icon: React.ReactNode; color: string; description: string }
> = {
  "52w_drop": {
    label: "52W Drop",
    icon: <TrendingDown className="h-3 w-3" />,
    color: "bg-orange-100 text-orange-800 border-orange-200",
    description:
      "Stock dropped 60%+ from 52-week high but maintains positive FCF",
  },
  quality_growth: {
    label: "Quality Growth",
    icon: <Sparkles className="h-3 w-3" />,
    color: "bg-purple-100 text-purple-800 border-purple-200",
    description: "Passes Quality Growth Gate (ROIC + Revenue Growth)",
  },
  deep_value: {
    label: "Deep Value",
    icon: <DollarSign className="h-3 w-3" />,
    color: "bg-green-100 text-green-800 border-green-200",
    description: "Margin of Safety > 40% or FCF Yield > 10%",
  },
};

export function TriggerBadge({
  trigger,
  showTooltip = true,
}: TriggerBadgeProps) {
  // Parse trigger (might be like "quality_growth:path2")
  const baseTrigger = trigger.split(":")[0];
  const path = trigger.includes(":path") ? trigger.split(":path")[1] : null;

  const config = triggerConfig[baseTrigger] || {
    label: trigger,
    icon: null,
    color: "bg-gray-100 text-gray-800 border-gray-200",
    description: trigger,
  };

  const badge = (
    <Badge
      variant="outline"
      className={`${config.color} gap-1 text-xs font-medium`}
    >
      {config.icon}
      {config.label}
      {path && <span className="opacity-70">P{path}</span>}
    </Badge>
  );

  if (!showTooltip) return badge;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>{badge}</TooltipTrigger>
        <TooltipContent>
          <p className="max-w-xs text-sm">{config.description}</p>
          {path && (
            <p className="mt-1 text-xs text-muted-foreground">
              Qualification Path {path}
            </p>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// ============================================================
// Classification Badge
// ============================================================

interface ClassificationBadgeProps {
  classification: "buy" | "hold" | "watch" | string;
  size?: "sm" | "md" | "lg";
}

const classificationConfig: Record<
  string,
  { label: string; color: string; description: string }
> = {
  buy: {
    label: "BUY",
    color: "bg-green-500 text-white",
    description: "MoS ≥ 20% AND Score ≥ 70",
  },
  hold: {
    label: "HOLD",
    color: "bg-blue-500 text-white",
    description: "MoS -10% to +20% AND Score ≥ 70",
  },
  watch: {
    label: "WATCH",
    color: "bg-yellow-500 text-black",
    description: "Monitor for better entry",
  },
};

export function ClassificationBadge({
  classification,
  size = "md",
}: ClassificationBadgeProps) {
  const config = classificationConfig[classification.toLowerCase()] || {
    label: classification.toUpperCase(),
    color: "bg-gray-500 text-white",
    description: "",
  };

  const sizeClasses = {
    sm: "px-2 py-0.5 text-xs",
    md: "px-3 py-1 text-sm",
    lg: "px-4 py-1.5 text-base",
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={`inline-flex items-center rounded-md font-bold ${config.color} ${sizeClasses[size]}`}
          >
            {config.label}
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p className="text-sm">{config.description}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// ============================================================
// Momentum Indicator
// ============================================================

interface MomentumIndicatorProps {
  priceVs200ma: number | null;
  priceVs50ma: number | null;
  compact?: boolean;
}

export function MomentumIndicator({
  priceVs200ma,
  priceVs50ma,
  compact = false,
}: MomentumIndicatorProps) {
  const below200 = priceVs200ma !== null && priceVs200ma < 0;
  const above50 = priceVs50ma !== null && priceVs50ma > 0;
  const idealEntry = below200 && above50;

  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1">
              <ArrowUpDown
                className={`h-4 w-4 ${
                  idealEntry
                    ? "text-green-500"
                    : below200
                    ? "text-yellow-500"
                    : "text-gray-400"
                }`}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <div className="text-sm">
              <p>
                200MA:{" "}
                {priceVs200ma !== null ? `${priceVs200ma.toFixed(1)}%` : "N/A"}
              </p>
              <p>
                50MA:{" "}
                {priceVs50ma !== null ? `${priceVs50ma.toFixed(1)}%` : "N/A"}
              </p>
              {idealEntry && (
                <p className="mt-1 font-medium text-green-600">
                  ✓ Ideal Entry Zone
                </p>
              )}
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <div className="space-y-1 text-sm">
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">vs 200MA</span>
        <span
          className={
            priceVs200ma !== null
              ? priceVs200ma < 0
                ? "text-green-600"
                : "text-red-600"
              : "text-gray-400"
          }
        >
          {priceVs200ma !== null ? `${priceVs200ma.toFixed(1)}%` : "N/A"}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground">vs 50MA</span>
        <span
          className={
            priceVs50ma !== null
              ? priceVs50ma > 0
                ? "text-green-600"
                : "text-yellow-600"
              : "text-gray-400"
          }
        >
          {priceVs50ma !== null ? `${priceVs50ma.toFixed(1)}%` : "N/A"}
        </span>
      </div>
      {idealEntry && (
        <div className="mt-2 rounded-md bg-green-50 px-2 py-1 text-center text-xs font-medium text-green-700">
          ✓ Accumulation Zone
        </div>
      )}
    </div>
  );
}

// ============================================================
// Score Badge (for composite score)
// ============================================================

interface ScoreBadgeProps {
  score: number;
  label?: string;
}

export function ScoreBadge({ score, label }: ScoreBadgeProps) {
  const getColor = (s: number) => {
    if (s >= 75) return "bg-green-100 text-green-800";
    if (s >= 60) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  };

  return (
    <div
      className={`inline-flex items-center rounded-full px-2.5 py-1 ${getColor(
        score
      )}`}
    >
      {label && <span className="mr-1 text-xs opacity-70">{label}</span>}
      <span className="font-bold">{score.toFixed(0)}</span>
    </div>
  );
}
