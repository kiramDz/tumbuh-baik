"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
  Area,
  AreaChart,
  Cell,
  Pie,
  PieChart,
  ScatterChart,
  Scatter,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface FSCIComponentsProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  selectedRegion?: string | null;
  maxRegions?: number;
  showTrends?: boolean;
}

const chartConfig = {
  fsci: {
    label: "FSCI Score",
    color: "hsl(var(--chart-1))",
  },
  production: {
    label: "Production (K tons)",
    color: "#8B5CF6", // Purple
  },
  correlation: {
    label: "Climate Correlation",
    color: "#10B981", // Green
  },
  excellent: {
    label: "Excellent (75+)",
    color: "#059669",
  },
  good: {
    label: "Good (60-74)",
    color: "#3B82F6",
  },
  fair: {
    label: "Fair (45-59)",
    color: "#F59E0B",
  },
  poor: {
    label: "Poor (<45)",
    color: "#DC2626",
  },
} satisfies ChartConfig;

export function FSCIComponents({
  className,
  analysisParams,
  level = "kabupaten",
  selectedRegion = null,
  maxRegions = 15,
  showTrends = true,
}: FSCIComponentsProps) {
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

  // Helper functions
  const getScoreClassification = (score: number) => {
    if (score >= 75) return "excellent";
    if (score >= 60) return "good";
    if (score >= 45) return "fair";
    return "poor";
  };

  const getPerformanceLabel = (score: number) => {
    if (score >= 75) return "Excellent";
    if (score >= 60) return "Good";
    if (score >= 45) return "Fair";
    return "Needs Improvement";
  };

  const getValue = (item: any, ...keys: string[]): number => {
    for (const key of keys) {
      const value = item[key];
      if (typeof value === "number" && value > 0) return value;
    }
    return 0;
  };

  const {
    data: analysisData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["two-level-analysis", params],
    queryFn: () => getTwoLevelAnalysis(params),
    refetchOnWindowFocus: false,
  });

  // Process available data (FSCI + Production + Correlation)
  const componentsData = useMemo(() => {
    if (!analysisData) return [];

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    const processed = sourceData
      .map((item: any) => {
        const fsciScore = getValue(
          item,
          "fsci_score",
          "fsci_mean",
          "aggregated_fsci_score"
        );
        const production = getValue(
          item,
          "production_tons",
          "latest_production_tons",
          "total_production"
        );
        const correlation = getValue(
          item,
          "climate_production_correlation",
          "correlation"
        );

        return {
          name:
            level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
          fsci: fsciScore,
          production: production / 1000, // Convert to K tons
          correlation: correlation,
          classification: getScoreClassification(fsciScore),
          performanceLabel: getPerformanceLabel(fsciScore),
          // Calculate efficiency metrics
          efficiency: production > 0 ? fsciScore / (production / 1000) : 0,
          // Determine if region is high-performing
          isHighPerforming: fsciScore >= 75,
        };
      })
      .filter((item) => item.fsci > 0)
      .sort((a, b) => b.fsci - a.fsci)
      .slice(0, maxRegions);

    return processed;
  }, [analysisData, level, maxRegions]);

  // Selected region detailed analysis
  const selectedRegionData = useMemo(() => {
    if (!selectedRegion) return null;
    return componentsData.find((item) => item.name === selectedRegion) || null;
  }, [componentsData, selectedRegion]);

  // Performance distribution analysis
  const distributionAnalysis = useMemo(() => {
    const classifications = componentsData.reduce((acc, item) => {
      acc[item.classification] = (acc[item.classification] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return [
      {
        name: "Excellent",
        value: classifications.excellent || 0,
        color: chartConfig.excellent.color,
        range: "75-100",
      },
      {
        name: "Good",
        value: classifications.good || 0,
        color: chartConfig.good.color,
        range: "60-74",
      },
      {
        name: "Fair",
        value: classifications.fair || 0,
        color: chartConfig.fair.color,
        range: "45-59",
      },
      {
        name: "Poor",
        value: classifications.poor || 0,
        color: chartConfig.poor.color,
        range: "<45",
      },
    ].filter((item) => item.value > 0);
  }, [componentsData]);

  // Summary statistics
  const summaryStats = useMemo(() => {
    if (componentsData.length === 0) return null;

    const fsciScores = componentsData.map((item) => item.fsci);
    const productions = componentsData.map((item) => item.production);
    const correlations = componentsData
      .map((item) => item.correlation)
      .filter((c) => c > 0);

    return {
      avgFsci: fsciScores.reduce((a, b) => a + b, 0) / fsciScores.length,
      totalProduction: productions.reduce((a, b) => a + b, 0),
      avgCorrelation:
        correlations.length > 0
          ? correlations.reduce((a, b) => a + b, 0) / correlations.length
          : 0,
      highPerformingCount: componentsData.filter(
        (item) => item.isHighPerforming
      ).length,
      regionsAnalyzed: componentsData.length,
    };
  }, [componentsData]);

  // Performance vs Production analysis
  const performanceProductionData = useMemo(() => {
    return componentsData.map((item) => ({
      name:
        item.name.length > 15 ? `${item.name.substring(0, 15)}...` : item.name,
      fsci: item.fsci,
      production: item.production,
      correlation: item.correlation,
      classification: item.classification,
      fullName: item.name,
    }));
  }, [componentsData]);

  // Trend data (FSCI vs Production efficiency)
  const trendData = useMemo(() => {
    return componentsData.slice(0, 10).map((item) => ({
      name:
        item.name.length > 12 ? `${item.name.substring(0, 12)}...` : item.name,
      fsci: item.fsci,
      production: item.production,
      efficiency: item.efficiency,
      fullName: item.name,
    }));
  }, [componentsData]);

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading FSCI analysis...</span>
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
            <p>Error loading FSCI analysis</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (componentsData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.barChart className="h-8 w-8 mx-auto mb-2" />
            <p>No FSCI data available</p>
            <p className="text-sm mt-1">
              Component analysis requires individual PCI/PSI/CRS scores
            </p>
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
            <Icons.wheat className="h-5 w-5 mr-2" />
            FSCI Performance Analysis
            <Badge variant="outline" className="ml-2">
              {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Level
            </Badge>
          </div>
          {summaryStats && (
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">
                Avg FSCI: {summaryStats.avgFsci.toFixed(1)}
              </Badge>
              <Badge variant="outline">
                {summaryStats.regionsAnalyzed} regions
              </Badge>
            </div>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Summary Statistics */}
            {summaryStats && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <Icons.wheat className="h-8 w-8 mx-auto mb-2 text-blue-600" />
                  <div className="text-2xl font-bold text-blue-700">
                    {summaryStats.avgFsci.toFixed(1)}
                  </div>
                  <div className="text-xs text-blue-600">Average FSCI</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <Icons.wheat className="h-8 w-8 mx-auto mb-2 text-purple-600" />
                  <div className="text-2xl font-bold text-purple-700">
                    {summaryStats.totalProduction.toFixed(0)}
                  </div>
                  <div className="text-xs text-purple-600">
                    Total Production (K tons)
                  </div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
                  <Icons.activity className="h-8 w-8 mx-auto mb-2 text-green-600" />
                  <div className="text-2xl font-bold text-green-700">
                    {summaryStats.avgCorrelation.toFixed(2)}
                  </div>
                  <div className="text-xs text-green-600">
                    Avg Climate Correlation
                  </div>
                </div>
                <div className="text-center p-4 bg-emerald-50 rounded-lg border border-emerald-200">
                  <Icons.award className="h-8 w-8 mx-auto mb-2 text-emerald-600" />
                  <div className="text-2xl font-bold text-emerald-700">
                    {summaryStats.highPerformingCount}
                  </div>
                  <div className="text-xs text-emerald-600">
                    High Performing Regions
                  </div>
                </div>
              </div>
            )}

            {/* Distribution Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* FSCI Performance Distribution */}
              <div>
                <h4 className="text-sm font-medium mb-3 flex items-center">
                  <Icons.pieChart className="h-4 w-4 mr-2" />
                  FSCI Performance Distribution
                </h4>
                <ChartContainer
                  config={chartConfig}
                  className="h-[250px] w-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={distributionAnalysis}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        nameKey="name"
                        label={({ name, value, percent }) =>
                          `${name}: ${(typeof percent === "number"
                            ? percent * 100
                            : 0
                          ).toFixed(0)}%`
                        }
                      >
                        {distributionAnalysis.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <ChartTooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-3 border rounded-lg shadow-lg">
                                <p className="font-semibold">
                                  {data.name} Performance
                                </p>
                                <p className="text-sm">
                                  Count: {data.value} regions
                                </p>
                                <p className="text-sm">Range: {data.range}</p>
                                <p className="text-sm">
                                  Percentage:{" "}
                                  {(
                                    (data.value / componentsData.length) *
                                    100
                                  ).toFixed(1)}
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

              {/* FSCI vs Production Scatter */}
              <div>
                <h4 className="text-sm font-medium mb-3 flex items-center">
                  <Icons.scatter className="h-4 w-4 mr-2" />
                  FSCI Score vs Production
                </h4>
                <ChartContainer
                  config={chartConfig}
                  className="h-[250px] w-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart data={performanceProductionData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="fsci"
                        name="FSCI Score"
                        domain={["dataMin - 5", "dataMax + 5"]}
                      />
                      <YAxis
                        dataKey="production"
                        name="Production (K tons)"
                        domain={["dataMin - 1", "dataMax + 1"]}
                      />
                      <Scatter
                        dataKey="production"
                        fill={chartConfig.fsci.color}
                      >
                        {performanceProductionData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={
                              chartConfig[
                                entry.classification as keyof typeof chartConfig
                              ]?.color || chartConfig.fsci.color
                            }
                          />
                        ))}
                      </Scatter>
                      <ChartTooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-white p-3 border rounded-lg shadow-lg">
                                <p className="font-semibold">{data.fullName}</p>
                                <p className="text-sm">
                                  FSCI Score: {data.fsci.toFixed(1)}
                                </p>
                                <p className="text-sm">
                                  Production: {data.production.toFixed(1)} K
                                  tons
                                </p>
                                <p className="text-sm">
                                  Correlation: {data.correlation.toFixed(2)}
                                </p>
                                <p className="text-sm">
                                  Performance: {data.classification}
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </div>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            {/* FSCI Performance Bar Chart */}
            <div>
              <h4 className="text-sm font-medium mb-3 flex items-center">
                <Icons.barChart className="h-4 w-4 mr-2" />
                FSCI Performance Ranking - Top 10 Regions
              </h4>
              <ChartContainer config={chartConfig} className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={trendData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      fontSize={10}
                    />
                    <YAxis tickFormatter={(value) => `${value}`} />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-white p-3 border rounded-lg shadow-lg">
                              <p className="font-semibold">{data.fullName}</p>
                              <p className="text-sm">
                                FSCI Score: {data.fsci.toFixed(1)}
                              </p>
                              <p className="text-sm">
                                Production: {data.production.toFixed(1)} K tons
                              </p>
                              <p className="text-sm">
                                Efficiency: {data.efficiency.toFixed(2)}
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar dataKey="fsci" fill={chartConfig.fsci.color} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>

            {/* Performance Rankings */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h5 className="text-sm font-medium text-green-700 mb-3">
                  Top Performers (FSCI ≥ 75)
                </h5>
                <div className="space-y-2">
                  {componentsData
                    .filter((item) => item.fsci >= 75)
                    .slice(0, 5)
                    .map((item, index) => (
                      <div
                        key={item.name}
                        className="flex items-center justify-between p-3 bg-green-50 rounded border border-green-200"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="w-6 h-6 bg-green-600 text-white text-xs font-bold rounded-full flex items-center justify-center">
                            {index + 1}
                          </div>
                          <div>
                            <span className="text-sm font-medium">
                              {item.name}
                            </span>
                            <p className="text-xs text-gray-600">
                              {item.production.toFixed(1)} K tons production
                            </p>
                          </div>
                        </div>
                        <span className="text-sm font-bold text-green-700">
                          {item.fsci.toFixed(1)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              <div>
                <h5 className="text-sm font-medium text-orange-700 mb-3">
                  Needs Improvement (FSCI &lt; 60)
                </h5>
                <div className="space-y-2">
                  {componentsData
                    .filter((item) => item.fsci < 60)
                    .slice(0, 5)
                    .map((item, index) => (
                      <div
                        key={item.name}
                        className="flex items-center justify-between p-3 bg-orange-50 rounded border border-orange-200"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="w-6 h-6 bg-orange-600 text-white text-xs font-bold rounded-full flex items-center justify-center">
                            {index + 1}
                          </div>
                          <div>
                            <span className="text-sm font-medium">
                              {item.name}
                            </span>
                            <p className="text-xs text-gray-600">
                              {item.production.toFixed(1)} K tons production
                            </p>
                          </div>
                        </div>
                        <span className="text-sm font-bold text-orange-700">
                          {item.fsci.toFixed(1)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Details Tab */}
          <TabsContent value="details" className="space-y-6">
            {selectedRegionData ? (
              <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                <h4 className="text-sm font-medium mb-3 flex items-center">
                  <Icons.mapPin className="h-4 w-4 mr-2" />
                  Detailed Analysis: {selectedRegionData.name}
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="text-center">
                    <div className="text-lg font-bold text-blue-700">
                      {selectedRegionData.fsci.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-600">FSCI Score</div>
                    <Progress
                      value={selectedRegionData.fsci}
                      className="mt-1"
                    />
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-purple-600">
                      {selectedRegionData.production.toFixed(1)}
                    </div>
                    <div className="text-xs text-gray-600">
                      Production (K tons)
                    </div>
                    <Progress
                      value={Math.min(100, selectedRegionData.production * 10)}
                      className="mt-1"
                    />
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-green-600">
                      {selectedRegionData.correlation.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-600">
                      Climate Correlation
                    </div>
                    <Progress
                      value={selectedRegionData.correlation * 100}
                      className="mt-1"
                    />
                  </div>
                </div>
                <div className="text-sm space-y-1">
                  <p>
                    <strong>Performance Level:</strong>{" "}
                    {selectedRegionData.performanceLabel}
                  </p>
                  <p>
                    <strong>Classification:</strong>{" "}
                    {selectedRegionData.classification}
                  </p>
                  <p>
                    <strong>Production Efficiency:</strong>{" "}
                    {selectedRegionData.efficiency.toFixed(2)}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                <Icons.mapPin className="h-8 w-8 mx-auto mb-2" />
                <p>Select a region to view detailed analysis</p>
                <p className="text-sm mt-1">
                  Component breakdown not available (requires PCI/PSI/CRS data)
                </p>
              </div>
            )}

            {/* Complete Data Table */}
            <div>
              <h4 className="text-sm font-medium mb-3 flex items-center">
                <Icons.table className="h-4 w-4 mr-2" />
                Complete FSCI Performance Data
              </h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Region
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        FSCI Score
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Production (K tons)
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Climate Correlation
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Performance
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Efficiency
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {componentsData.map((item) => (
                      <tr
                        key={item.name}
                        className={`hover:bg-gray-50 ${
                          selectedRegion === item.name ? "bg-blue-50" : ""
                        }`}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {item.name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-blue-600">
                          {item.fsci.toFixed(1)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-purple-600">
                          {item.production.toFixed(1)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">
                          {item.correlation.toFixed(2)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              item.classification === "excellent"
                                ? "bg-green-100 text-green-800"
                                : item.classification === "good"
                                ? "bg-blue-100 text-blue-800"
                                : item.classification === "fair"
                                ? "bg-yellow-100 text-yellow-800"
                                : "bg-red-100 text-red-800"
                            }`}
                          >
                            {item.performanceLabel}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {item.efficiency.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
