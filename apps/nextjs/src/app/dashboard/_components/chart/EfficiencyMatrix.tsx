"use client";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  Cell,
  Tooltip,
  PieChart,
  Pie,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from "recharts";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface EfficiencyMatrixProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  showPerformanceGap?: boolean;
  maxRegions?: number;
}

const chartConfig = {
  efficiency: {
    label: "Production Efficiency (%)",
    color: "hsl(var(--chart-1))",
  },
  potential: {
    label: "Climate Potential",
    color: "hsl(var(--chart-2))",
  },
  gap: {
    label: "Performance Gap",
    color: "hsl(var(--chart-3))",
  },
  overperforming: {
    label: "Overperforming",
    color: "#10B981",
  },
  aligned: {
    label: "Aligned",
    color: "#3B82F6",
  },
  underperforming: {
    label: "Underperforming",
    color: "#F59E0B",
  },
  critical: {
    label: "Critical Gap",
    color: "#EF4444",
  },
} satisfies ChartConfig;

export function EfficiencyMatrix({
  className,
  analysisParams,
  level = "kabupaten",
  showPerformanceGap = true,
  maxRegions = 10,
}: EfficiencyMatrixProps) {
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

  // Efficiency quadrant classification
  const getEfficiencyQuadrant = (
    fsciScore: number,
    efficiency: number
  ): string => {
    const medianFSCI = 60; // Threshold for high/low climate potential
    const medianEfficiency = 70; // Threshold for high/low efficiency

    if (fsciScore >= medianFSCI && efficiency >= medianEfficiency) {
      return "High Potential - High Efficiency";
    } else if (fsciScore >= medianFSCI && efficiency < medianEfficiency) {
      return "High Potential - Low Efficiency";
    } else if (fsciScore < medianFSCI && efficiency >= medianEfficiency) {
      return "Low Potential - High Efficiency";
    } else {
      return "Low Potential - Low Efficiency";
    }
  };

  // Process efficiency matrix data
  const matrixData = useMemo(() => {
    if (!analysisData) return [];

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    const processed = sourceData
      .map((item: any) => {
        const fsciScore = item.aggregated_fsci_score || item.fsci_score || 0;
        const productionTons = item.latest_production_tons || 0;
        const efficiency = item.production_efficiency_score || 0;
        const category = item.performance_gap_category || "aligned";

        // Calculate performance gap (difference between potential and actual efficiency)
        const expectedProduction = fsciScore * 1000; // Simple scaling factor
        const performanceGap =
          expectedProduction > 0
            ? ((productionTons - expectedProduction) / expectedProduction) * 100
            : 0;

        return {
          name:
            level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
          fsciScore: fsciScore,
          production: productionTons,
          efficiency: efficiency * 100, // Convert to percentage
          category: category,
          performanceGap: performanceGap,
          climate_correlation: item.climate_production_correlation || 0,
          // Matrix quadrant classification
          quadrant: getEfficiencyQuadrant(fsciScore, efficiency * 100),
        };
      })
      .filter((item) => item.fsciScore > 0 && item.production > 0)
      .sort((a, b) => b.efficiency - a.efficiency) // Sort by efficiency descending
      .slice(0, maxRegions);

    return processed;
  }, [analysisData, level, maxRegions]);

  // Quadrant distribution
  const quadrantDistribution = useMemo(() => {
    const distribution = matrixData.reduce((acc, item) => {
      acc[item.quadrant] = (acc[item.quadrant] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return [
      {
        name: "High Pot. - High Eff.",
        value: distribution["High Potential - High Efficiency"] || 0,
        color: chartConfig.overperforming.color,
        fullName: "High Potential - High Efficiency",
      },
      {
        name: "High Pot. - Low Eff.",
        value: distribution["High Potential - Low Efficiency"] || 0,
        color: chartConfig.underperforming.color,
        fullName: "High Potential - Low Efficiency",
      },
      {
        name: "Low Pot. - High Eff.",
        value: distribution["Low Potential - High Efficiency"] || 0,
        color: chartConfig.aligned.color,
        fullName: "Low Potential - High Efficiency",
      },
      {
        name: "Low Pot. - Low Eff.",
        value: distribution["Low Potential - Low Efficiency"] || 0,
        color: chartConfig.critical.color,
        fullName: "Low Potential - Low Efficiency",
      },
    ].filter((item) => item.value > 0);
  }, [matrixData]);

  // Top performers and underperformers
  const performanceAnalysis = useMemo(() => {
    const sorted = [...matrixData].sort(
      (a, b) => b.performanceGap - a.performanceGap
    );

    return {
      topPerformers: sorted.slice(0, 5),
      underperformers: sorted.slice(-5).reverse(),
      averageEfficiency:
        matrixData.length > 0
          ? matrixData.reduce((sum, item) => sum + item.efficiency, 0) /
            matrixData.length
          : 0,
      averageGap:
        matrixData.length > 0
          ? matrixData.reduce(
              (sum, item) => sum + Math.abs(item.performanceGap),
              0
            ) / matrixData.length
          : 0,
    };
  }, [matrixData]);

  // Efficiency matrix scatter data for bar chart
  const efficiencyBars = useMemo(() => {
    return matrixData.slice(0, 10).map((item) => ({
      name:
        item.name.length > 15 ? `${item.name.substring(0, 15)}...` : item.name,
      efficiency: item.efficiency,
      fsci: item.fsciScore,
      gap: Math.abs(item.performanceGap),
      category: item.category,
    }));
  }, [matrixData]);

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading efficiency matrix...</span>
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
            <Icons.alertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p>Error loading efficiency data</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (matrixData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.barChart className="h-8 w-8 mx-auto mb-2" />
            <p>No efficiency data available</p>
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
            <Icons.barChart className="h-5 w-5 mr-2" />
            Production Efficiency Matrix
            <Badge variant="outline" className="ml-2">
              {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Level
            </Badge>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="secondary">
              Avg: {performanceAnalysis.averageEfficiency.toFixed(1)}%
            </Badge>
            <Badge variant="outline">
              Gap: {performanceAnalysis.averageGap.toFixed(1)}%
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
            <Icons.trendingUp className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold text-green-700">
              {performanceAnalysis.topPerformers.length}
            </div>
            <div className="text-xs text-green-600">Top Performers</div>
          </div>
          <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
            <Icons.target className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold text-blue-700">
              {performanceAnalysis.averageEfficiency.toFixed(0)}%
            </div>
            <div className="text-xs text-blue-600">Avg Efficiency</div>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg border border-orange-200">
            <Icons.alertTriangle className="h-8 w-8 mx-auto mb-2 text-orange-600" />
            <div className="text-2xl font-bold text-orange-700">
              {performanceAnalysis.underperformers.length}
            </div>
            <div className="text-xs text-orange-600">Need Improvement</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
            <Icons.activity className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold text-purple-700">
              {performanceAnalysis.averageGap.toFixed(0)}%
            </div>
            <div className="text-xs text-purple-600">Avg Gap</div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Quadrant Distribution Pie Chart */}
          <div>
            <h4 className="text-sm font-medium mb-3 flex items-center">
              <Icons.pieChart className="h-4 w-4 mr-2" />
              Efficiency Quadrants
            </h4>
            <ChartContainer config={chartConfig} className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={quadrantDistribution}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    nameKey="name"
                    label={({ name, value, percent }) =>
                      `${name}: ${(percent * 100).toFixed(0)}%`
                    }
                  >
                    {quadrantDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-3 border rounded-lg shadow-lg">
                            <p className="font-semibold">{data.fullName}</p>
                            <p className="text-sm">
                              Count: {data.value} regions
                            </p>
                            <p className="text-sm">
                              Percentage:{" "}
                              {((data.value / matrixData.length) * 100).toFixed(
                                1
                              )}
                              %
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>

          {/* Efficiency Bar Chart */}
          <div>
            <h4 className="text-sm font-medium mb-3 flex items-center">
              <Icons.barChart className="h-4 w-4 mr-2" />
              Top 10 Efficiency Rankings
            </h4>
            <ChartContainer config={chartConfig} className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={efficiencyBars}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={10}
                  />
                  <YAxis tickFormatter={(value) => `${value}%`} />
                  <ChartTooltip
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-3 border rounded-lg shadow-lg">
                            <p className="font-semibold">{label}</p>
                            <p className="text-sm">
                              Efficiency: {data.efficiency.toFixed(1)}%
                            </p>
                            <p className="text-sm">
                              FSCI Score: {data.fsci.toFixed(1)}
                            </p>
                            <p className="text-sm">
                              Performance Gap: {data.gap.toFixed(1)}%
                            </p>
                            <p className="text-sm capitalize">
                              Category: {data.category}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar
                    dataKey="efficiency"
                    fill={chartConfig.efficiency.color}
                  />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </div>

        {/* Performance Lists */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Performers */}
          <div>
            <h4 className="text-sm font-medium mb-3 flex items-center">
              <Icons.trophy className="h-4 w-4 mr-2 text-yellow-600" />
              Top Performers
            </h4>
            <div className="space-y-2">
              {performanceAnalysis.topPerformers.map((item, index) => (
                <div
                  key={item.name}
                  className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-200"
                >
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center justify-center w-6 h-6 bg-green-600 text-white text-xs font-bold rounded-full">
                      {index + 1}
                    </div>
                    <div>
                      <p className="font-medium text-sm">{item.name}</p>
                      <p className="text-xs text-gray-600">
                        FSCI: {item.fsciScore.toFixed(1)} | Gap:{" "}
                        {item.performanceGap.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-green-700">
                      {item.efficiency.toFixed(1)}%
                    </div>
                    <Progress value={item.efficiency} className="w-16 h-2" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Underperformers */}
          <div>
            <h4 className="text-sm font-medium mb-3 flex items-center">
              <Icons.alertTriangle className="h-4 w-4 mr-2 text-orange-600" />
              Needs Improvement
            </h4>
            <div className="space-y-2">
              {performanceAnalysis.underperformers.map((item, index) => (
                <div
                  key={item.name}
                  className="flex items-center justify-between p-3 bg-orange-50 rounded-lg border border-orange-200"
                >
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center justify-center w-6 h-6 bg-orange-600 text-white text-xs font-bold rounded-full">
                      {matrixData.length -
                        performanceAnalysis.underperformers.length +
                        index +
                        1}
                    </div>
                    <div>
                      <p className="font-medium text-sm">{item.name}</p>
                      <p className="text-xs text-gray-600">
                        FSCI: {item.fsciScore.toFixed(1)} | Gap:{" "}
                        {item.performanceGap.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-orange-700">
                      {item.efficiency.toFixed(1)}%
                    </div>
                    <Progress value={item.efficiency} className="w-16 h-2" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
