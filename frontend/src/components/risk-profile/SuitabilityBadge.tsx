/**
 * Suitability Badge Component
 *
 * Visual indicator for stock suitability based on user's risk profile.
 */
"use client";

import * as React from "react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CheckCircle, AlertTriangle, XCircle, HelpCircle } from "lucide-react";

type Suitability = "excellent" | "good" | "moderate" | "poor" | "unknown";

interface Props {
  suitability: Suitability;
  beta?: number;
  marginOfSafety?: number;
  profileBetaRange?: { min: number; max: number };
  requiredMoS?: number;
  compact?: boolean;
}

const suitabilityConfig = {
  excellent: {
    icon: CheckCircle,
    color: "bg-green-500/10 text-green-500 border-green-500/20",
    label: "Excellent Fit",
    description: "This stock matches your risk profile well",
  },
  good: {
    icon: CheckCircle,
    color: "bg-green-500/10 text-green-600 border-green-500/20",
    label: "Good Fit",
    description: "This stock is suitable for your profile",
  },
  moderate: {
    icon: AlertTriangle,
    color: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
    label: "Moderate Fit",
    description: "Some aspects may not match your preferences",
  },
  poor: {
    icon: XCircle,
    color: "bg-red-500/10 text-red-500 border-red-500/20",
    label: "Poor Fit",
    description: "This stock may be too risky for your profile",
  },
  unknown: {
    icon: HelpCircle,
    color: "bg-muted text-muted-foreground border-muted",
    label: "Unknown",
    description: "Complete risk assessment to see suitability",
  },
};

export function SuitabilityBadge({
  suitability,
  beta,
  marginOfSafety,
  profileBetaRange,
  requiredMoS,
  compact = false,
}: Props) {
  const config = suitabilityConfig[suitability];
  const Icon = config.icon;

  const getTooltipDetails = () => {
    const details: string[] = [];

    if (beta !== undefined && profileBetaRange) {
      if (beta >= profileBetaRange.min && beta <= profileBetaRange.max) {
        details.push(`✓ Beta (${beta.toFixed(2)}) within your range`);
      } else if (beta < profileBetaRange.min) {
        details.push(
          `⚠ Beta (${beta.toFixed(
            2
          )}) below your range - may be too conservative`
        );
      } else {
        details.push(
          `⚠ Beta (${beta.toFixed(2)}) above your range - may be too risky`
        );
      }
    }

    if (marginOfSafety !== undefined && requiredMoS !== undefined) {
      if (marginOfSafety >= requiredMoS) {
        details.push(
          `✓ Margin of Safety (${marginOfSafety.toFixed(
            1
          )}%) meets your threshold`
        );
      } else {
        details.push(
          `⚠ Margin of Safety (${marginOfSafety.toFixed(
            1
          )}%) below your ${requiredMoS}% threshold`
        );
      }
    }

    return details;
  };

  const details = getTooltipDetails();

  if (compact) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="inline-flex items-center justify-center">
              <Icon
                className={`h-5 w-5 ${
                  suitability === "excellent" || suitability === "good"
                    ? "text-green-500"
                    : suitability === "moderate"
                    ? "text-yellow-500"
                    : suitability === "poor"
                    ? "text-red-500"
                    : "text-muted-foreground"
                }`}
              />
            </span>
          </TooltipTrigger>
          <TooltipContent side="right" className="max-w-xs">
            <div className="space-y-2">
              <p className="font-medium">{config.label}</p>
              <p className="text-sm text-muted-foreground">
                {config.description}
              </p>
              {details.length > 0 && (
                <ul className="text-xs space-y-1 pt-2 border-t">
                  {details.map((detail, i) => (
                    <li key={i}>{detail}</li>
                  ))}
                </ul>
              )}
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className={`${config.color} gap-1 cursor-help`}
          >
            <Icon className="h-3 w-3" />
            {config.label}
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-xs">
          <div className="space-y-2">
            <p className="text-sm">{config.description}</p>
            {details.length > 0 && (
              <ul className="text-xs space-y-1 pt-2 border-t">
                {details.map((detail, i) => (
                  <li key={i}>{detail}</li>
                ))}
              </ul>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
