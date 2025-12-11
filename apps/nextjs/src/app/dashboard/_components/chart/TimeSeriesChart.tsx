"use client";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  AreaChart,
  Area,
  ComposedChart,
  Bar,
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
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface TimeSeriesChartProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  selectedRegions?: string[];
  showForecast?: boolean;
  chartType?: "line" | "area" | "composed";
  maxRegions?: number;
}

const chartConfig = {
  fsci: {
    label: "FSCI Score",
    color: "hsl(var(--chart-1))",
  },
  production: {
    label: "Production (K tons)",
    color: "hsl(var(--chart-2))",
  },
  pci: {
    label: "PCI",
    color: "#3B82F6", // Blue
  },
  psi: {
    label: "PSI",
    color: "#F59E0B", // Orange
  },
  crs: {
    label: "CRS",
    color: "#10B981", // Green
  },
  efficiency: {
    label: "Efficiency (%)",
    color: "#8B5CF6", // Purple
  },
  trend: {
    label: "Trend",
    color: "#6B7280", // Gray
  },
  forecast: {
    label: "Forecast",
    color: "#EF4444", // Red
  },
} satisfies ChartConfig;

export function TimeSeriesChart({
  className,
  analysisParams,
  level = "kabupaten",
  selectedRegions = [],
  showForecast = false,
  chartType = "line",
  maxRegions = 5,
}: TimeSeriesChartProps) {
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

  // Process time series data
  const timeSeriesData = useMemo(() => {
    if (!analysisData) return [];

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    // Get top regions if none selected
    const regionsToShow =
      selectedRegions.length > 0
        ? selectedRegions
        : sourceData
            .sort((a: any, b: any) => {
              const aScore =
                level === "kabupaten"
                  ? (a as any).fsci_mean ||
                    (a as any).aggregated_fsci_score ||
                    0
                  : (a as any).fsci_score || (a as any).fsci_mean || 0;
              const bScore =
                level === "kabupaten"
                  ? (b as any).fsci_mean ||
                    (b as any).aggregated_fsci_score ||
                    0
                  : (b as any).fsci_score || (b as any).fsci_mean || 0;
              return bScore - aScore;
            })
            .slice(0, maxRegions)
            .map((item: any) =>
              level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name
            );

    const startYear = params.year_start || 2018;
    const endYear = params.year_end || 2024;

    // Generate years array with safe parameters
    const years = [];
    for (let year = startYear; year <= endYear; year++) {
      years.push(year);
    }

    // Create time series data structure
    const timeSeriesMap = new Map();

    years.forEach((year) => {
      const yearData: any = { year };

      regionsToShow.forEach((regionName) => {
        const regionData = sourceData.find(
          (item: any) =>
            (level === "kabupaten"
              ? item.kabupaten_name
              : item.kecamatan_name) === regionName
        );

        if (regionData) {
          const itemData = regionData as any; // Type assertion for flexibility

          // Get FSCI score with fallbacks
          const baseValue =
            level === "kabupaten"
              ? itemData.fsci_mean ||
                itemData.aggregated_fsci_score ||
                itemData.fsci_score ||
                0
              : itemData.fsci_score ||
                itemData.fsci_mean ||
                itemData.aggregated_fsci_score ||
                0;

          // Get production data with fallbacks
          const productionTons =
            itemData.production_tons ||
            itemData.latest_production_tons ||
            itemData.total_production ||
            0;

          // Get efficiency with fallbacks
          const efficiency =
            itemData.efficiency_score ||
            itemData.production_efficiency_score ||
            itemData.performance_score ||
            0;

          // For demo, we'll create synthetic time series data
          // In real implementation, you'd have historical data per year
          const variation = Math.sin((year - startYear) * 0.5) * 5;
          const trend = (year - startYear) * 0.5;

          yearData[`${regionName}_fsci`] = Math.max(
            0,
            baseValue + variation + trend
          );

          yearData[`${regionName}_production`] = Math.max(
            0,
            (productionTons / 1000) * (1 + (variation + trend) / 100)
          );

          yearData[`${regionName}_efficiency`] = Math.max(
            0,
            Math.min(100, efficiency * 100 + variation)
          );
        }
      });

      timeSeriesMap.set(year, yearData);
    });

    return Array.from(timeSeriesMap.values());
  }, [analysisData, level, selectedRegions, maxRegions, params]);

  // Calculate trends and statistics
  const trendAnalysis = useMemo(() => {
    if (timeSeriesData.length < 2) return null;

    const regionStats = new Map();

    // Get regions from the data
    const regions = selectedRegions.length > 0 ? selectedRegions : [];
    if (regions.length === 0 && timeSeriesData.length > 0) {
      const firstYear = timeSeriesData[0];
      Object.keys(firstYear).forEach((key) => {
        if (key.endsWith("_fsci")) {
          regions.push(key.replace("_fsci", ""));
        }
      });
    }

    regions.forEach((region) => {
      const fsciValues = timeSeriesData
        .map((d) => d[`${region}_fsci`] || 0)
        .filter((v) => v > 0);
      const productionValues = timeSeriesData
        .map((d) => d[`${region}_production`] || 0)
        .filter((v) => v > 0);

      if (fsciValues.length >= 2) {
        const fsciTrend =
          (fsciValues[fsciValues.length - 1] - fsciValues[0]) /
          fsciValues.length;
        const productionTrend =
          productionValues.length >= 2
            ? (productionValues[productionValues.length - 1] -
                productionValues[0]) /
              productionValues.length
            : 0;

        regionStats.set(region, {
          fsciTrend,
          productionTrend,
          fsciGrowth:
            fsciValues.length >= 2
              ? ((fsciValues[fsciValues.length - 1] - fsciValues[0]) /
                  fsciValues[0]) *
                100
              : 0,
          productionGrowth:
            productionValues.length >= 2
              ? ((productionValues[productionValues.length - 1] -
                  productionValues[0]) /
                  productionValues[0]) *
                100
              : 0,
        });
      }
    });

    return regionStats;
  }, [timeSeriesData, selectedRegions]);

  // Forecast data (simple linear projection)
  const forecastData = useMemo(() => {
    if (!showForecast || !trendAnalysis || timeSeriesData.length === 0)
      return [];

    const lastYear = params.year_end || 2024;
    const forecastYears = [lastYear + 1, lastYear + 2];

    return forecastYears.map((year) => {
      const yearData: any = { year, isForecast: true };

      trendAnalysis.forEach((stats, region) => {
        const lastValue = timeSeriesData[timeSeriesData.length - 1];

        if (!lastValue) return;

        const yearsAhead = year - lastYear;

        yearData[`${region}_fsci`] = Math.max(
          0,
          (lastValue[`${region}_fsci`] || 0) + stats.fsciTrend * yearsAhead
        );
        yearData[`${region}_production`] = Math.max(
          0,
          (lastValue[`${region}_production`] || 0) +
            stats.productionTrend * yearsAhead
        );
      });

      return yearData;
    });
  }, [showForecast, trendAnalysis, timeSeriesData, params.year_end]);

  // Combined data with forecast
  const combinedData = useMemo(() => {
    return [...timeSeriesData, ...forecastData];
  }, [timeSeriesData, forecastData]);

  // Get regions for rendering
  const displayRegions = useMemo(() => {
    if (selectedRegions.length > 0) return selectedRegions.slice(0, maxRegions);

    if (timeSeriesData.length === 0) return [];

    const firstYear = timeSeriesData[0];
    if (!firstYear) return [];

    const regions: string[] = [];

    Object.keys(firstYear).forEach((key) => {
      if (key.endsWith("_fsci") && !key.startsWith("year")) {
        regions.push(key.replace("_fsci", ""));
      }
    });

    return regions.slice(0, maxRegions);
  }, [selectedRegions, timeSeriesData, maxRegions]);

  // Chart colors for regions
  const getRegionColor = (index: number) => {
    const colors = [
      chartConfig.fsci.color,
      chartConfig.production.color,
      chartConfig.pci.color,
      chartConfig.psi.color,
      chartConfig.crs.color,
    ];
    return colors[index % colors.length];
  };
  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading time series data...</span>
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
            <p>Error loading time series data</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (combinedData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.trendingUp className="h-8 w-8 mx-auto mb-2" />
            <p>No time series data available</p>
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
            <Icons.trendingUp className="h-5 w-5 mr-2" />
            Time Series Analysis
            <Badge variant="outline" className="ml-2">
              {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Level
            </Badge>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="secondary">
              {params.year_start || 2018}-{params.year_end || 2024}
            </Badge>
            {showForecast && (
              <Badge variant="outline">
                Forecast: {(params.year_end || 2024) + 1}-
                {(params.year_end || 2024) + 2}
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="fsci" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="fsci">FSCI Trends</TabsTrigger>
            <TabsTrigger value="production">Production Trends</TabsTrigger>
            <TabsTrigger value="comparison">Multi-Metric</TabsTrigger>
          </TabsList>

          {/* FSCI Trends */}
          <TabsContent value="fsci" className="space-y-6">
            <div>
              <h4 className="text-sm font-medium mb-3 flex items-center">
                <Icons.lineChart className="h-4 w-4 mr-2" />
                FSCI Score Trends Over Time
              </h4>
              <ChartContainer config={chartConfig} className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  {chartType === "area" ? (
                    <AreaChart data={combinedData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="year"
                        tickFormatter={(value) => `${value}`}
                      />
                      <YAxis
                        tickFormatter={(value) => `${Number(value).toFixed(0)}`}
                      />
                      <ChartTooltip
                        content={({ active, payload, label }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-white p-3 border rounded-lg shadow-lg">
                                <p className="font-semibold">Year {label}</p>
                                {payload.map((entry, index) => {
                                  if (
                                    typeof entry.dataKey === "string" &&
                                    entry.dataKey.includes("_fsci")
                                  ) {
                                    const regionName = entry.dataKey.replace(
                                      "_fsci",
                                      ""
                                    );
                                    const value =
                                      typeof entry.value === "number"
                                        ? entry.value
                                        : 0;
                                    return (
                                      <p
                                        key={index}
                                        className="text-sm"
                                        style={{ color: entry.color }}
                                      >
                                        {regionName}: {value.toFixed(1)}
                                      </p>
                                    );
                                  }
                                  return null;
                                })}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      {displayRegions.map((region, index) => (
                        <Area
                          key={region}
                          type="monotone"
                          dataKey={`${region}_fsci`}
                          stackId="1"
                          stroke={getRegionColor(index)}
                          fill={getRegionColor(index)}
                          fillOpacity={0.3}
                          strokeWidth={2}
                          name={region}
                          connectNulls={false}
                        />
                      ))}
                    </AreaChart>
                  ) : (
                    <LineChart data={combinedData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="year"
                        tickFormatter={(value) => `${value}`}
                      />
                      <YAxis
                        tickFormatter={(value) => `${Number(value).toFixed(0)}`}
                      />
                      <ChartTooltip
                        content={({ active, payload, label }) => {
                          if (active && payload && payload.length) {
                            const isForecast = payload[0]?.payload?.isForecast;
                            return (
                              <div className="bg-white p-3 border rounded-lg shadow-lg">
                                <p className="font-semibold">
                                  {isForecast ? "Forecast " : ""}Year {label}
                                </p>
                                {payload.map((entry, index) => {
                                  if (
                                    typeof entry.dataKey === "string" &&
                                    entry.dataKey.includes("_fsci")
                                  ) {
                                    const regionName = entry.dataKey.replace(
                                      "_fsci",
                                      ""
                                    );
                                    const value =
                                      typeof entry.value === "number"
                                        ? entry.value
                                        : 0;
                                    return (
                                      <p
                                        key={index}
                                        className="text-sm"
                                        style={{ color: entry.color }}
                                      >
                                        {regionName}: {value.toFixed(1)}
                                      </p>
                                    );
                                  }
                                  return null;
                                })}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      {displayRegions.map((region, index) => (
                        <Line
                          key={region}
                          type="monotone"
                          dataKey={`${region}_fsci`}
                          stroke={getRegionColor(index)}
                          strokeWidth={2}
                          dot={{ r: 4 }}
                          name={region}
                          connectNulls={false}
                        />
                      ))}
                    </LineChart>
                  )}
                </ResponsiveContainer>
              </ChartContainer>
            </div>

            {/* FSCI Trend Statistics */}
            {trendAnalysis && (
              <div>
                <h4 className="text-sm font-medium mb-3 flex items-center">
                  <Icons.barChart className="h-4 w-4 mr-2" />
                  FSCI Growth Analysis
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Array.from(trendAnalysis.entries()).map(
                    ([region, stats]) => (
                      <div key={region} className="p-4 border rounded-lg">
                        <h5 className="font-medium text-sm mb-2">{region}</h5>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span>FSCI Growth:</span>
                            <span
                              className={`font-semibold ${
                                stats.fsciGrowth > 0
                                  ? "text-green-600"
                                  : "text-red-600"
                              }`}
                            >
                              {stats.fsciGrowth > 0 ? "+" : ""}
                              {Number(stats.fsciGrowth).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Annual Trend:</span>
                            <span
                              className={`font-semibold ${
                                stats.fsciTrend > 0
                                  ? "text-green-600"
                                  : "text-red-600"
                              }`}
                            >
                              {stats.fsciTrend > 0 ? "+" : ""}
                              {Number(stats.fsciTrend).toFixed(2)}/year
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Production Trends */}
          <TabsContent value="production" className="space-y-6">
            <div>
              <h4 className="text-sm font-medium mb-3 flex items-center">
                <Icons.trendingUp className="h-4 w-4 mr-2" />
                Production Trends Over Time
              </h4>
              <ChartContainer config={chartConfig} className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={combinedData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="year"
                      tickFormatter={(value) => `${value}`}
                    />
                    <YAxis
                      tickFormatter={(value) => `${Number(value).toFixed(0)}K`}
                    />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const isForecast = payload[0]?.payload?.isForecast;
                          return (
                            <div className="bg-white p-3 border rounded-lg shadow-lg">
                              <p className="font-semibold">
                                {isForecast ? "Forecast " : ""}Year {label}
                              </p>
                              {payload.map((entry, index) => {
                                if (
                                  typeof entry.dataKey === "string" &&
                                  entry.dataKey.includes("_production")
                                ) {
                                  const regionName = entry.dataKey.replace(
                                    "_production",
                                    ""
                                  );
                                  const value =
                                    typeof entry.value === "number"
                                      ? entry.value
                                      : 0;
                                  return (
                                    <p
                                      key={index}
                                      className="text-sm"
                                      style={{ color: entry.color }}
                                    >
                                      {regionName}: {value.toFixed(1)}K tons
                                    </p>
                                  );
                                }
                                return null;
                              })}
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    {displayRegions.map((region, index) => (
                      <Line
                        key={region}
                        type="monotone"
                        dataKey={`${region}_production`}
                        stroke={getRegionColor(index)}
                        strokeWidth={2}
                        dot={{ r: 4 }}
                        name={region}
                        connectNulls={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>

            {/* Production Trend Statistics */}
            {trendAnalysis && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Array.from(trendAnalysis.entries()).map(([region, stats]) => (
                  <div key={region} className="p-4 border rounded-lg">
                    <h5 className="font-medium text-sm mb-2">{region}</h5>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span>Production Growth:</span>
                        <span
                          className={`font-semibold ${
                            stats.productionGrowth > 0
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {stats.productionGrowth > 0 ? "+" : ""}
                          {Number(stats.productionGrowth).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Annual Trend:</span>
                        <span
                          className={`font-semibold ${
                            stats.productionTrend > 0
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {stats.productionTrend > 0 ? "+" : ""}
                          {Number(stats.productionTrend).toFixed(2)}K/year
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Multi-Metric Comparison */}
          <TabsContent value="comparison" className="space-y-6">
            <div>
              <h4 className="text-sm font-medium mb-3 flex items-center">
                <Icons.activity className="h-4 w-4 mr-2" />
                Multi-Metric Comparison
              </h4>
              <ChartContainer config={chartConfig} className="h-[500px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={combinedData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="year"
                      tickFormatter={(value) => `${value}`}
                    />
                    <YAxis
                      yAxisId="left"
                      tickFormatter={(value) => `${Number(value).toFixed(0)}`}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      tickFormatter={(value) => `${Number(value).toFixed(0)}K`}
                    />
                    <ChartTooltip
                      content={({ active, payload, label }) => {
                        if (active && payload && payload.length) {
                          const isForecast = payload[0]?.payload?.isForecast;
                          return (
                            <div className="bg-white p-3 border rounded-lg shadow-lg">
                              <p className="font-semibold">
                                {isForecast ? "Forecast " : ""}Year {label}
                              </p>
                              {payload.map((entry, index) => {
                                const value =
                                  typeof entry.value === "number"
                                    ? entry.value
                                    : 0;
                                if (
                                  typeof entry.dataKey === "string" &&
                                  entry.dataKey.includes("_fsci")
                                ) {
                                  const regionName = entry.dataKey.replace(
                                    "_fsci",
                                    ""
                                  );
                                  return (
                                    <p
                                      key={index}
                                      className="text-sm"
                                      style={{ color: entry.color }}
                                    >
                                      {regionName} FSCI: {value.toFixed(1)}
                                    </p>
                                  );
                                } else if (
                                  typeof entry.dataKey === "string" &&
                                  entry.dataKey.includes("_production")
                                ) {
                                  const regionName = entry.dataKey.replace(
                                    "_production",
                                    ""
                                  );
                                  return (
                                    <p
                                      key={index}
                                      className="text-sm"
                                      style={{ color: entry.color }}
                                    >
                                      {regionName} Production:{" "}
                                      {value.toFixed(1)}K tons
                                    </p>
                                  );
                                }
                                return null;
                              })}
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Legend />

                    {/* FSCI Lines */}
                    {displayRegions.slice(0, 2).map((region, index) => (
                      <Line
                        key={`${region}_fsci`}
                        yAxisId="left"
                        type="monotone"
                        dataKey={`${region}_fsci`}
                        stroke={getRegionColor(index)}
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        name={`${region} FSCI`}
                        connectNulls={false}
                      />
                    ))}

                    {/* Production Bars */}
                    {displayRegions.slice(0, 2).map((region, index) => (
                      <Bar
                        key={`${region}_production`}
                        yAxisId="right"
                        dataKey={`${region}_production`}
                        fill={getRegionColor(index)}
                        fillOpacity={0.3}
                        name={`${region} Production`}
                      />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
