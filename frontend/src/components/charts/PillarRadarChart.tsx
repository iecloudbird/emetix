/**
 * PillarRadarChart Component
 *
 * Displays 4-pillar scores (Value, Quality, Growth, Safety) in a radar chart.
 * Part of Phase 3 Quality Screening Pipeline.
 */
"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { PillarScores } from "@/lib/api";

interface PillarRadarChartProps {
  pillarScores: PillarScores;
  compositeScore: number;
  className?: string;
}

interface RadarDataPoint {
  pillar: string;
  score: number;
  fullMark: number;
}

export function PillarRadarChart({
  pillarScores,
  compositeScore,
  className,
}: PillarRadarChartProps) {
  // Transform pillar scores into radar chart data
  const data: RadarDataPoint[] = [
    { pillar: "Value", score: pillarScores.value.score, fullMark: 100 },
    { pillar: "Quality", score: pillarScores.quality.score, fullMark: 100 },
    { pillar: "Growth", score: pillarScores.growth.score, fullMark: 100 },
    { pillar: "Safety", score: pillarScores.safety.score, fullMark: 100 },
  ];

  // Get color based on composite score
  const getScoreColor = (score: number) => {
    if (score >= 75) return "#16a34a"; // green-600
    if (score >= 60) return "#ca8a04"; // yellow-600
    return "#dc2626"; // red-600
  };

  const fillColor = getScoreColor(compositeScore);

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span>4-Pillar Score</span>
          <span className="text-2xl font-bold" style={{ color: fillColor }}>
            {compositeScore.toFixed(0)}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid />
            <PolarAngleAxis
              dataKey="pillar"
              tick={{ fill: "#6b7280", fontSize: 12 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={{ fill: "#9ca3af", fontSize: 10 }}
            />
            <Radar
              name="Score"
              dataKey="score"
              stroke={fillColor}
              fill={fillColor}
              fillOpacity={0.4}
              strokeWidth={2}
            />
            <Tooltip
              formatter={(value) => [
                `${(value as number).toFixed(1)}`,
                "Score",
              ]}
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
              }}
            />
          </RadarChart>
        </ResponsiveContainer>

        {/* Pillar breakdown */}
        <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
          {data.map((item) => (
            <div
              key={item.pillar}
              className="flex items-center justify-between rounded-md bg-muted/50 px-3 py-2"
            >
              <span className="font-medium">{item.pillar}</span>
              <span
                className={
                  item.score >= 75
                    ? "text-green-600"
                    : item.score >= 50
                    ? "text-yellow-600"
                    : "text-red-600"
                }
              >
                {item.score.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * PillarScoreBar - Simple horizontal bar for inline pillar display
 */
interface PillarScoreBarProps {
  pillar: string;
  score: number;
  showLabel?: boolean;
}

export function PillarScoreBar({
  pillar,
  score,
  showLabel = true,
}: PillarScoreBarProps) {
  const getBarColor = (s: number) => {
    if (s >= 75) return "bg-green-500";
    if (s >= 50) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="flex items-center gap-2">
      {showLabel && (
        <span className="w-16 text-xs font-medium text-muted-foreground">
          {pillar}
        </span>
      )}
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${getBarColor(score)} transition-all`}
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
      <span className="w-8 text-xs font-medium text-right">
        {score.toFixed(0)}
      </span>
    </div>
  );
}

/**
 * PillarScoreSummary - Compact 4-pillar summary for list views
 */
interface PillarScoreSummaryProps {
  pillarScores: PillarScores;
  className?: string;
}

export function PillarScoreSummary({
  pillarScores,
  className,
}: PillarScoreSummaryProps) {
  const pillars = [
    { name: "V", score: pillarScores.value.score, full: "Value" },
    { name: "Q", score: pillarScores.quality.score, full: "Quality" },
    { name: "G", score: pillarScores.growth.score, full: "Growth" },
    { name: "S", score: pillarScores.safety.score, full: "Safety" },
  ];

  return (
    <div className={`flex gap-1 ${className}`}>
      {pillars.map((p) => (
        <div
          key={p.name}
          className={`flex h-6 w-6 items-center justify-center rounded text-xs font-bold ${
            p.score >= 75
              ? "bg-green-100 text-green-700"
              : p.score >= 50
              ? "bg-yellow-100 text-yellow-700"
              : "bg-red-100 text-red-700"
          }`}
          title={`${p.full}: ${p.score.toFixed(1)}`}
        >
          {p.name}
        </div>
      ))}
    </div>
  );
}
