/**
 * Profile Results Component
 *
 * Displays risk profile assessment results with gauges and recommendations.
 */
"use client";

import * as React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import type { RiskProfileResponse } from "@/types/risk-profile";
import {
  Shield,
  Target,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Info,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";

interface Props {
  profile: RiskProfileResponse;
}

// Score gauge component
function ScoreGauge({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: number;
  icon: React.ElementType;
  color: "green" | "yellow" | "blue" | "purple";
}) {
  const colorClasses = {
    green: "text-green-500",
    yellow: "text-yellow-500",
    blue: "text-blue-500",
    purple: "text-purple-500",
  };

  const progressColorClasses = {
    green: "[&>div]:bg-green-500",
    yellow: "[&>div]:bg-yellow-500",
    blue: "[&>div]:bg-blue-500",
    purple: "[&>div]:bg-purple-500",
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className={`h-4 w-4 ${colorClasses[color]}`} />
          <span className="text-sm font-medium">{label}</span>
        </div>
        <span className={`text-2xl font-bold ${colorClasses[color]}`}>
          {value.toFixed(0)}
        </span>
      </div>
      <Progress
        value={value}
        className={`h-2 ${progressColorClasses[color]}`}
      />
    </div>
  );
}

export function ProfileResults({ profile }: Props) {
  const profileColors: Record<
    string,
    { bg: string; text: string; border: string }
  > = {
    conservative: {
      bg: "bg-green-500/10",
      text: "text-green-500",
      border: "border-green-500/20",
    },
    moderate: {
      bg: "bg-yellow-500/10",
      text: "text-yellow-500",
      border: "border-yellow-500/20",
    },
    aggressive: {
      bg: "bg-red-500/10",
      text: "text-red-500",
      border: "border-red-500/20",
    },
  };

  const profileType = profile.overall_risk_profile || "moderate";
  const colors = profileColors[profileType] || profileColors.moderate;

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      {/* Header Card */}
      <Card className={`${colors.bg} ${colors.border} border-2`}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl">Your Risk Profile</CardTitle>
            <Badge
              variant="outline"
              className={`${colors.text} ${colors.border}`}
            >
              {profileType.charAt(0).toUpperCase() + profileType.slice(1)}
            </Badge>
          </div>
          <CardDescription>Profile ID: {profile.profile_id}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <ScoreGauge
              label="Risk Capacity"
              value={profile.risk_capacity?.score ?? 50}
              icon={Shield}
              color="blue"
            />
            <ScoreGauge
              label="Risk Tolerance"
              value={profile.risk_tolerance?.score ?? 50}
              icon={Target}
              color="purple"
            />
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Your Personalized Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg border p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">Emotional Buffer</span>
              </div>
              <div className="text-3xl font-bold">
                {(profile.emotional_buffer?.factor ?? 1.0).toFixed(2)}x
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Safety margin multiplier
              </p>
            </div>

            <div className="rounded-lg border p-4">
              <div className="flex items-center gap-2 text-muted-foreground mb-1">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm">Required MoS</span>
              </div>
              <div className="text-3xl font-bold text-primary">
                {(
                  profile.emotional_buffer?.adjusted_mos_threshold ?? 20
                ).toFixed(0)}
                %
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Minimum margin of safety
              </p>
            </div>
          </div>

          <Separator className="my-4" />

          <div className="rounded-lg border p-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Info className="h-4 w-4" />
              <span className="text-sm font-medium">Suitable Beta Range</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="text-sm text-muted-foreground">Min</div>
                <div className="text-xl font-semibold">
                  {profile.suitable_beta_range.min.toFixed(2)}
                </div>
              </div>
              <div className="text-2xl text-muted-foreground">—</div>
              <div className="flex-1 text-right">
                <div className="text-sm text-muted-foreground">Max</div>
                <div className="text-xl font-semibold">
                  {profile.suitable_beta_range.max.toFixed(2)}
                </div>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Stocks with beta outside this range may be too risky or too
              conservative for your profile.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            Personalized Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-3">
            {profile.recommendations.map((rec, index) => (
              <li key={index} className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                <span className="text-sm">{rec}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex gap-4">
        <Link href="/watchlist" className="flex-1">
          <Button className="w-full" size="lg">
            View Suitable Stocks
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </Link>
        <Button variant="outline" size="lg">
          Retake Assessment
        </Button>
      </div>
    </div>
  );
}
