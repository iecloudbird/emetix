/**
 * Profile Suitability Card
 *
 * Shows suitability badge, position sizing, and comparison
 * of stock metrics to user's risk profile on stock detail page.
 */
"use client";

import { useEffect } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  UserCheck,
  UserX,
  AlertTriangle,
  TrendingUp,
  Shield,
  Target,
  DollarSign,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import type { Stock } from "@/lib/api";
import type { RiskProfileResponse } from "@/types/risk-profile";
import { usePositionSizing } from "@/hooks/useRiskProfile";

interface ProfileSuitabilityCardProps {
  stock: Stock;
  profile: RiskProfileResponse | null;
}

type SuitabilityRating = "excellent" | "good" | "moderate" | "poor";

interface SuitabilityResult {
  rating: SuitabilityRating;
  suitable: boolean;
  betaInRange: boolean;
  mosAboveThreshold: boolean;
  reasons: string[];
}

function getSuitability(
  stock: Stock,
  profile: RiskProfileResponse
): SuitabilityResult {
  const beta = stock.beta ?? 1.0;
  const mos = stock.margin_of_safety ?? 0;

  const betaRange = profile.suitable_beta_range;
  const mosThreshold = profile.emotional_buffer?.adjusted_mos_threshold ?? 20;

  const betaInRange = beta >= betaRange.min && beta <= betaRange.max;
  const mosAboveThreshold = mos >= mosThreshold;

  const reasons: string[] = [];

  if (!betaInRange) {
    if (beta < betaRange.min) {
      reasons.push(
        `Beta (${beta.toFixed(2)}) is below your minimum (${betaRange.min})`
      );
    } else {
      reasons.push(
        `Beta (${beta.toFixed(2)}) exceeds your max tolerance (${
          betaRange.max
        })`
      );
    }
  }

  if (!mosAboveThreshold) {
    reasons.push(
      `Margin of Safety (${mos.toFixed(
        1
      )}%) is below your threshold (${mosThreshold.toFixed(1)}%)`
    );
  }

  // Calculate rating
  let rating: SuitabilityRating;
  if (betaInRange && mosAboveThreshold) {
    // Both criteria met
    if (mos >= mosThreshold * 1.5) {
      rating = "excellent";
    } else {
      rating = "good";
    }
  } else if (betaInRange || mosAboveThreshold) {
    rating = "moderate";
    if (betaInRange) {
      reasons.unshift("Beta is within your tolerance");
    }
    if (mosAboveThreshold) {
      reasons.unshift("MoS meets your threshold");
    }
  } else {
    rating = "poor";
  }

  return {
    rating,
    suitable: betaInRange && mosAboveThreshold,
    betaInRange,
    mosAboveThreshold,
    reasons,
  };
}

export function ProfileSuitabilityCard({
  stock,
  profile,
}: ProfileSuitabilityCardProps) {
  const positionMutation = usePositionSizing();

  // Fetch position sizing when we have profile
  useEffect(() => {
    if (
      !profile ||
      !stock.beta ||
      positionMutation.isPending ||
      positionMutation.data
    )
      return;

    positionMutation.mutate({
      profile_id: profile.profile_id,
      ticker: stock.ticker,
      current_price: stock.current_price,
      margin_of_safety: stock.margin_of_safety ?? 0,
      beta: stock.beta,
    });
  }, [profile?.profile_id, stock.ticker]); // eslint-disable-line react-hooks/exhaustive-deps

  const positionSizing = positionMutation.data;
  const loadingPosition = positionMutation.isPending;

  // No profile - show prompt to create one
  if (!profile) {
    return (
      <Card className="border-dashed">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-muted-foreground">
            <UserX className="w-5 h-5" />
            Profile Suitability
          </CardTitle>
          <CardDescription>
            Create a risk profile to see if this stock suits you
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Link href="/profile">
            <Button variant="outline" className="w-full">
              <Target className="w-4 h-4 mr-2" />
              Take Risk Assessment
            </Button>
          </Link>
        </CardContent>
      </Card>
    );
  }

  const suitability = getSuitability(stock, profile);

  const ratingConfig = {
    excellent: {
      badge: "bg-green-500 hover:bg-green-600",
      icon: UserCheck,
      label: "Excellent Match",
      description: "This stock aligns perfectly with your risk profile",
    },
    good: {
      badge: "bg-emerald-500 hover:bg-emerald-600",
      icon: UserCheck,
      label: "Good Match",
      description: "This stock suits your risk profile well",
    },
    moderate: {
      badge: "bg-yellow-500 hover:bg-yellow-600",
      icon: AlertTriangle,
      label: "Moderate Match",
      description: "Consider the factors below before investing",
    },
    poor: {
      badge: "bg-red-500 hover:bg-red-600",
      icon: UserX,
      label: "Poor Match",
      description: "This stock may not suit your risk profile",
    },
  };

  const config = ratingConfig[suitability.rating];
  const Icon = config.icon;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Icon className="w-5 h-5" />
            Profile Suitability
          </CardTitle>
          <Badge className={config.badge}>{config.label}</Badge>
        </div>
        <CardDescription>{config.description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Beta Comparison */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium">Stock Beta</span>
          </div>
          <div className="text-right">
            <span
              className={cn(
                "text-lg font-bold",
                suitability.betaInRange ? "text-green-600" : "text-red-600"
              )}
            >
              {stock.beta?.toFixed(2) ?? "N/A"}
            </span>
            <p className="text-xs text-muted-foreground">
              Your range: {profile.suitable_beta_range.min.toFixed(1)} -{" "}
              {profile.suitable_beta_range.max.toFixed(1)}
            </p>
          </div>
        </div>

        {/* MoS Comparison */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-muted-foreground" />
            <span className="text-sm font-medium">Margin of Safety</span>
          </div>
          <div className="text-right">
            <span
              className={cn(
                "text-lg font-bold",
                suitability.mosAboveThreshold
                  ? "text-green-600"
                  : "text-red-600"
              )}
            >
              {(stock.margin_of_safety ?? 0).toFixed(1)}%
            </span>
            <p className="text-xs text-muted-foreground">
              Your threshold:{" "}
              {(profile.emotional_buffer?.adjusted_mos_threshold ?? 20).toFixed(
                1
              )}
              %
            </p>
          </div>
        </div>

        <Separator />

        {/* Position Sizing */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <DollarSign className="w-4 h-4" />
            Position Sizing Recommendation
          </div>
          {loadingPosition ? (
            <p className="text-sm text-muted-foreground">
              Calculating optimal position...
            </p>
          ) : positionSizing ? (
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-primary/10">
                <p className="text-xs text-muted-foreground">Max Position</p>
                <p className="text-lg font-bold">
                  {positionSizing.max_position_percent.toFixed(1)}%
                </p>
              </div>
              <div className="p-3 rounded-lg bg-primary/10">
                <p className="text-xs text-muted-foreground">Max Shares</p>
                <p className="text-lg font-bold">{positionSizing.max_shares}</p>
              </div>
              <div className="col-span-2 p-3 rounded-lg bg-primary/10">
                <p className="text-xs text-muted-foreground">Max Investment</p>
                <p className="text-lg font-bold">
                  $
                  {positionSizing.max_position_value.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              Position sizing not available
            </p>
          )}
        </div>

        {/* Reasons / Warnings */}
        {suitability.reasons.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Info className="w-4 h-4" />
                Key Considerations
              </div>
              <ul className="space-y-1">
                {suitability.reasons.map((reason, i) => (
                  <li
                    key={i}
                    className="text-sm text-muted-foreground flex items-start gap-2"
                  >
                    <span
                      className={cn(
                        "mt-1.5 w-1.5 h-1.5 rounded-full",
                        suitability.suitable ? "bg-green-500" : "bg-yellow-500"
                      )}
                    />
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}

        {/* Profile Summary */}
        <Separator />
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Profile:{" "}
            <span className="capitalize font-medium">
              {profile.overall_risk_profile}
            </span>
          </span>
          <Link
            href="/profile"
            className="hover:underline hover:text-foreground"
          >
            Update Profile →
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
