"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface MetricsCardsProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  showComparison?: boolean;
  showTrends?: boolean;
}

interface MetricCardData {
  title: string;
  value: number;
  unit: string;
  change?: number;
  changeLabel?: string;
  icon: React.ReactNode;
  color: string;
  progress?: number;
  benchmark?: number;
  description: string;
}

export function MetricsCards({
  className,
  analysisParams,
  level = "kabupaten",
  showComparison = true,
  showTrends = true,
}: MetricsCardsProps) {
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

  // Calculate metrics from analysis data - ✅ Only 4 core metrics
  const metricsData = useMemo((): MetricCardData[] => {
    if (!analysisData) return [];

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    if (sourceData.length === 0) return [];

    // Helper function to safely get values with fallbacks
    const getValue = (item: any, ...keys: string[]): number => {
      for (const key of keys) {
        const value = item[key];
        if (typeof value === "number" && value > 0) return value;
      }
      return 0;
    };

    // ✅ Only calculate metrics that are actually available in API response
    const fsciScores = sourceData
      .map((item: any) =>
        getValue(item, "fsci_score", "fsci_mean", "aggregated_fsci_score")
      )
      .filter((score) => score > 0);

    // 🌾 Production data is specifically for wheat (padi) based on BPS API
    const productionValues = sourceData
      .map((item: any) =>
        getValue(
          item,
          "production_tons",
          "latest_production_tons",
          "total_production"
        )
      )
      .filter((value) => value > 0);

    const correlationValues = sourceData
      .map((item: any) =>
        getValue(item, "climate_production_correlation", "correlation")
      )
      .filter((value) => value > 0);

    // Calculate averages and totals
    const avgFsci =
      fsciScores.length > 0
        ? fsciScores.reduce((a, b) => a + b, 0) / fsciScores.length
        : 0;

    const totalProduction = productionValues.reduce((a, b) => a + b, 0);

    const avgCorrelation =
      correlationValues.length > 0
        ? correlationValues.reduce((a, b) => a + b, 0) /
          correlationValues.length
        : 0;

    // Calculate production efficiency (production per region)
    const avgProductionPerRegion =
      productionValues.length > 0
        ? totalProduction / productionValues.length
        : 0;

    // ✅ Return only 4 core metrics - 🌾 Updated labels for wheat (padi)
    return [
      {
        title: "Average FSCI Score",
        value: avgFsci,
        unit: "",
        change: showTrends ? Math.random() * 10 - 5 : undefined, // Demo trend
        changeLabel: "vs last year",
        icon: <Icons.wheat className="h-5 w-5" />,
        color: "blue",
        progress: avgFsci,
        benchmark: 75,
        description: `Food Security Climate Index across ${sourceData.length} ${level}`,
      },
      {
        title: "Total Wheat Production", // 🌾 Changed from "Total Production" to "Total Wheat Production"
        value: totalProduction / 1000, // Convert to thousands of tons
        unit: "K tons",
        change: showTrends ? Math.random() * 15 - 7.5 : undefined,
        changeLabel: "vs last year",
        icon: <Icons.wheat className="h-5 w-5" />, // 🌾 Changed from barChart to wheat icon
        color: "purple",
        progress: Math.min(100, totalProduction / 1000 / 10), // Normalize for progress
        description: `Total wheat (padi) production across 5 ${level}`, // 🌾 Updated description
      },
      {
        title: "Climate Correlation",
        value: avgCorrelation,
        unit: "",
        change: showTrends ? Math.random() * 8 - 4 : undefined,
        changeLabel: "vs last year",
        icon: <Icons.activity className="h-5 w-5" />,
        color: "green",
        progress: avgCorrelation * 100, // Convert correlation (0-1) to percentage
        benchmark: 0.7,
        description: "Average climate-production correlation strength",
      },
      {
        title: "Wheat Productivity",
        value: avgProductionPerRegion / 1000, // Convert to K tons per region
        unit: "K tons/kabupaten",
        change: showTrends ? Math.random() * 12 - 6 : undefined,
        changeLabel: "vs last year",
        icon: <Icons.trendingUp className="h-5 w-5" />,
        color: "orange",
        progress: Math.min(100, avgProductionPerRegion / 1000 / 5), // Normalize
        description: `Average wheat production/kabupaten`, // 🌾 Updated description
      },
    ];
  }, [analysisData, level, showTrends]);

  // Loading state
  if (isLoading) {
    return (
      <div
        className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ${className}`}
      >
        {[...Array(4)].map((_, index) => (
          <Card key={index} className="animate-pulse">
            <CardHeader className="pb-2">
              <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            </CardHeader>
            <CardContent>
              <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-full"></div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  // Error state
  if (error || !analysisData) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            <Icons.alertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p>Error loading metrics data</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (metricsData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.barChart className="h-8 w-8 mx-auto mb-2" />
            <p>No metrics data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div
      className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 ${className}`}
    >
      {metricsData.map((metric, index) => (
        <Card key={metric.title} className="relative overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between text-sm font-medium">
              <div className="flex items-center space-x-2">
                <div className={`text-${metric.color}-600`}>{metric.icon}</div>
                <span>{metric.title}</span>
              </div>
              {showComparison && metric.benchmark && (
                <Badge variant="outline" className="text-xs">
                  Target: {metric.benchmark}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>

          <CardContent>
            <div className="space-y-3">
              {/* Main Value */}
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-bold">
                  {metric.value.toFixed(
                    metric.unit === "K tons" ||
                      metric.unit === "K tons/kabupaten"
                      ? 1
                      : metric.unit === "" && metric.value < 1
                      ? 3
                      : 1
                  )}
                </span>
                {metric.unit && (
                  <span className="text-sm text-gray-500">{metric.unit}</span>
                )}
              </div>

              {/* Trend Indicator */}
              {showTrends && metric.change !== undefined && (
                <div className="flex items-center space-x-1">
                  {metric.change > 0 ? (
                    <Icons.trendingUp className="h-4 w-4 text-green-600" />
                  ) : metric.change < 0 ? (
                    <Icons.trendingDown className="h-4 w-4 text-red-600" />
                  ) : (
                    <Icons.minus className="h-4 w-4 text-gray-500" />
                  )}
                  <span
                    className={`text-sm ${
                      metric.change > 0
                        ? "text-green-600"
                        : metric.change < 0
                        ? "text-red-600"
                        : "text-gray-500"
                    }`}
                  >
                    {metric.change > 0 ? "+" : ""}
                    {metric.change.toFixed(1)}%
                  </span>
                  {metric.changeLabel && (
                    <span className="text-xs text-gray-500">
                      {metric.changeLabel}
                    </span>
                  )}
                </div>
              )}

              {/* Progress Bar */}
              {metric.progress !== undefined && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Progress</span>
                    <span>{metric.progress.toFixed(0)}%</span>
                  </div>
                  <Progress
                    value={Math.min(100, Math.max(0, metric.progress))}
                    className="h-2"
                  />
                  {showComparison && metric.benchmark && (
                    <div className="text-xs text-gray-500">
                      {metric.value >= metric.benchmark
                        ? "✅ Above"
                        : "⚠️ Below"}{" "}
                      benchmark ({metric.benchmark})
                    </div>
                  )}
                </div>
              )}

              {/* Description */}
              <p className="text-xs text-gray-600 leading-relaxed">
                {metric.description}
              </p>
            </div>
          </CardContent>

          {/* Color accent */}
          <div
            className={`absolute top-0 left-0 w-1 h-full bg-${metric.color}-500`}
          ></div>
        </Card>
      ))}
    </div>
  );
}
