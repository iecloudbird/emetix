/**
 * SectorPieChart Component
 *
 * Displays sector distribution of watchlist stocks.
 * Uses Recharts PieChart for visual breakdown.
 */
"use client";

import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { SectorData } from "@/lib/api";

interface SectorPieChartProps {
  sectors: SectorData[];
}

// Color palette for sectors
const COLORS = [
  "#3b82f6", // blue
  "#22c55e", // green
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // purple
  "#06b6d4", // cyan
  "#ec4899", // pink
  "#f97316", // orange
  "#14b8a6", // teal
  "#6366f1", // indigo
];

export function SectorPieChart({ sectors }: SectorPieChartProps) {
  const data = sectors.map((s, i) => ({
    name: s.sector,
    value: s.stock_count,
    avgScore: s.avg_valuation_score,
    color: COLORS[i % COLORS.length],
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sector Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} (${((percent ?? 0) * 100).toFixed(0)}%)`
                }
                labelLine={false}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name, props) => {
                  const numValue = value as number;
                  const payload = props.payload as { avgScore: number };
                  return [
                    `${numValue} stocks (Avg Score: ${payload.avgScore.toFixed(
                      1
                    )})`,
                    name as string,
                  ];
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
