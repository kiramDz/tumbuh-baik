"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface CorrelationScatterProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  showTrendLine?: boolean;
  maxDataPoints?: number;
}

const chartConfig = {
  production: {
    label: "Production (tons)",
    color: "hsl(var(--chart-1))",
  },
  climate: {
    label: "Climate Potential (FSCI)",
    color: "hsl(var(--chart-2))",
  },
  overperforming: {
    label: "Overperforming",
    color: "#10B981", // Green
  },
  aligned: {
    label: "Aligned",
    color: "#3B82F6", // Blue
  },
  underperforming: {
    label: "Underperforming",
    color: "#F59E0B", // Orange
  },
} satisfies ChartConfig;

export function CorrelationScatter({
  className,
  analysisParams,
  level = "kabupaten",
  showTrendLine = true,
  maxDataPoints = 50,
}: CorrelationScatterProps) {
  // Default parameters
  const defaultParams: TwoLevelAnalysisParams = {
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    districts: "all",
  };

  const params = analysisParams || defaultParams;

  const {
    data: analysisData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["two-level-analysis", params],
    queryFn: () => getTwoLevelAnalysis(params),
    refetchOnWindowFocus: false,
  });

  // Process scatter data
  const scatterData = useMemo(() => {
    if (!analysisData) return [];

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    const processed = sourceData
      .map((item: any) => ({
        name: level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
        x: item.aggregated_fsci_score || item.fsci_score || 0, // Climate potential
        y: item.latest_production_tons || 0, // Production
        category: item.performance_gap_category || "aligned",
        efficiency: item.production_efficiency_score || 0,
        correlation: item.climate_production_correlation || 0,
      }))
      .filter((item) => item.x > 0 && item.y > 0) // Remove zero values
      .sort((a, b) => b.y - a.y) // Sort by production descending
      .slice(0, maxDataPoints); // Limit data points for performance

    return processed;
  }, [analysisData, level, maxDataPoints]);

  // Calculate correlation statistics

  const correlationStats = useMemo(() => {
    if (scatterData.length < 2) return null;
    const n = scatterData.length;
    const sumX = scatterData.reduce((sum, d) => sum + d.x, 0);
    const sumY = scatterData.reduce((sum, d) => sum + d.y, 0);
    const sumXY = scatterData.reduce((sum, d) => sum + d.x * d.y, 0);
    const sumXX = scatterData.reduce((sum, d) => sum + d.x * d.x, 0);
    const sumYY = scatterData.reduce((sum, d) => sum + d.y * d.y, 0);

    const correlation =
      (n * sumXY - sumX * sumY) /
      Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

    // Linear regression for trend line
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return {
      correlation: isNaN(correlation) ? 0 : correlation,
      slope,
      intercept,
      strength:
        Math.abs(correlation) >= 0.7
          ? "Strong"
          : Math.abs(correlation) >= 0.5
          ? "Moderate"
          : Math.abs(correlation) >= 0.3
          ? "Weak"
          : "Very Weak",
    };
  }, [scatterData]);

  // Performance category counts
  const categoryCounts = useMemo(() => {
    const counts = scatterData.reduce((acc, item) => {
      acc[item.category] = (acc[item.category] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      overperforming: counts.overperforming || 0,
      aligned: counts.aligned || 0,
      underperforming: counts.underperforming || 0,
    };
  }, [scatterData]);

  // Custom dot component with category colors
  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props;
    const color =
      chartConfig[payload.category as keyof typeof chartConfig]?.color ||
      chartConfig.aligned.color;

    return (
      <circle
        cx={cx}
        cy={cy}
        r={6}
        fill={color}
        stroke={color}
        strokeWidth={2}
        fillOpacity={0.7}
        className="hover:r-8 transition-all duration-200"
      />
    );
  };

  // Trend line data
  const trendLineData = useMemo(() => {
    if (!correlationStats || !showTrendLine) return [];

    const minX = Math.min(...scatterData.map((d) => d.x));
    const maxX = Math.max(...scatterData.map((d) => d.x));

    return [
      {
        x: minX,
        y: correlationStats.slope * minX + correlationStats.intercept,
      },
      {
        x: maxX,
        y: correlationStats.slope * maxX + correlationStats.intercept,
      },
    ];
  }, [correlationStats, scatterData, showTrendLine]);

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading correlation analysis...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !analysisData) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            <Icons.activity className="h-8 w-8 mx-auto mb-2" />
            <p>Error loading correlation data</p>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (scatterData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.target className="h-8 w-8 mx-auto mb-2" />
            <p>No correlation data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Icons.activity className="h-5 w-5 mr-2" />
            Climate vs Production Correlation
            <Badge variant="outline" className="ml-2">
              {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Level
            </Badge>
          </div>
          {correlationStats && (
            <div className="flex items-center space-x-2">
              {correlationStats.correlation > 0 ? (
                <Icons.trendingUp className="h-4 w-4 text-green-600" />
              ) : (
                <Icons.trendingDown className="h-4 w-4 text-red-600" />
              )}
              <Badge variant="secondary">
                r = {correlationStats.correlation.toFixed(3)} (
                {correlationStats.strength})
              </Badge>
            </div>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent>
        {/* Summary Staistics */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
            <div className="text-2xl font-bold text-green-700">
              {categoryCounts.overperforming}
            </div>
            <div className="text-xs text-green-600">Overperforming</div>
          </div>
          <div className="text-center p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="text-2xl font-bold text-blue-700">
              {categoryCounts.aligned}
            </div>
            <div className="text-xs text-blue-600">Aligned</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded-lg border border-orange-200">
            <div className="text-2xl font-bold text-orange-700">
              {categoryCounts.underperforming}
            </div>
            <div className="text-xs text-orange-600">Underperforming</div>
          </div>
        </div>
        {/* Scatter Plot */}
        <ChartContainer config={chartConfig} className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart
              data={scatterData}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="x"
                name="Climate Potential (FSCI)"
                domain={["dataMin - 5", "dataMax + 5"]}
                tickFormatter={(value) => `${value.toFixed(0)}`}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Production (tons)"
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
              />
              <ChartTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="bg-white p-3 border rounded-lg shadow-lg">
                        <p className="font-semibold">{data.name}</p>
                        <p className="text-sm">FSCI: {data.x.toFixed(1)}</p>
                        <p className="text-sm">
                          Production: {(data.y / 1000).toFixed(1)}K tons
                        </p>
                        <p className="text-sm">
                          Category:{" "}
                          <span className="capitalize">{data.category}</span>
                        </p>
                        <p className="text-sm">
                          Efficiency: {(data.efficiency * 100).toFixed(1)}%
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter dataKey="y" fill="#8884d8" shape={<CustomDot />} />

              {/* Trend Line */}
              {showTrendLine && trendLineData.length > 0 && (
                <ReferenceLine
                  segment={trendLineData}
                  stroke="#FF6B6B"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </ChartContainer>
        {/* Legend */}
        <div className="flex justify-center mt-4">
          <div className="flex items-center space-x-6">
            {Object.entries(categoryCounts).map(([category, count]) => (
              <div key={category} className="flex items-center space-x-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{
                    backgroundColor:
                      chartConfig[category as keyof typeof chartConfig].color,
                  }}
                />
                <span className="text-sm capitalize">
                  {category} ({count})
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
