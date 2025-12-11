"use client";
import { Suspense, useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { AlertDescription, Alert } from "@/components/ui/alert";

// Dashboard components
import { MetricsCards } from "../_components/MetricsCards";
import { RankingTable } from "../_components/RankingTable";
import { PolicyPanel } from "../_components/PolicyPanel";
import { InvestmentPriority } from "../_components/InvestmentPriority";

// Chart components
import {
  TimeSeriesChart,
  CorrelationScatter,
  EfficiencyMatrix,
  FSCIComponents,
} from "../_components/chart";

// Icons
import { Icons } from "../_components/icons";

// Hook
import { useDashboardData } from "@/hooks/use-dashboard-data";

// Types
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

// Loading components
function DashboardSkeleton() {
  return (
    <div className="space-y-6 p-6">
      {/* Header Skeleton */}
      <div className="space-y-2">
        <div className="h-8 bg-gray-200 rounded w-1/3 animate-pulse" />
        <div className="h-4 bg-gray-200 rounded w-1/2 animate-pulse" />
      </div>

      {/* Metrics Skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-32 bg-gray-200 rounded animate-pulse" />
        ))}
      </div>

      {/* Charts Skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-96 bg-gray-200 rounded animate-pulse" />
        ))}
      </div>

      {/* Tables Skeleton */}
      <div className="space-y-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-64 bg-gray-200 rounded animate-pulse" />
        ))}
      </div>
    </div>
  );
}

function ErrorDisplay({
  error,
  onRetry,
}: {
  error: Error;
  onRetry: () => void;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <Card className="max-w-md w-full">
        <CardHeader>
          <CardTitle className="flex items-center text-red-600">
            <Icons.alertTriangle className="h-5 w-5 mr-2" />
            Dashboard Error
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertDescription>
              Failed to load dashboard data. Please check your connection and
              try again.
            </AlertDescription>
          </Alert>

          <div className="text-sm text-gray-600">
            <details>
              <summary className="cursor-pointer hover:text-gray-800">
                Error Details
              </summary>
              <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto">
                {error.message}
              </pre>
            </details>
          </div>

          <div className="flex gap-3">
            <Button onClick={onRetry} className="flex-1">
              <Icons.refresh className="h-4 w-4 mr-2" />
              Retry
            </Button>
            <Button
              variant="outline"
              onClick={() => window.location.reload()}
              className="flex-1"
            >
              <Icons.refresh className="h-4 w-4 mr-2" />
              Reload Page
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Main Dashboard Page Component
export default function FoodSecurityDashboard() {
  // State for filters
  const [analysisParams, setAnalysisParams] = useState<
    Partial<TwoLevelAnalysisParams>
  >({
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    districts: "all",
  });

  const [refreshKey, setRefreshKey] = useState(0);
  const [exportFormat, setExportFormat] = useState<"json" | "csv">("json");

  // Main dashboard data hook
  const {
    data,
    metrics,
    chartData,
    rankings,
    filters,
    activeLevel,
    isLoading,
    error,
    isFetching,
    isReady,
    refetch,
    switchLevel,
    exportData,
  } = useDashboardData(analysisParams, {
    level: "kabupaten",
    autoRefresh: false,
  });

  // Handle parameter updates
  const updateFilters = (newParams: Partial<TwoLevelAnalysisParams>) => {
    setAnalysisParams((prev) => ({ ...prev, ...newParams }));
    setRefreshKey((prev) => prev + 1);
  };

  // Handle data export
  const handleExport = () => {
    const exportedData = exportData(exportFormat);
    if (exportedData) {
      const blob = new Blob([exportedData], {
        type: exportFormat === "json" ? "application/json" : "text/csv",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `food-security-dashboard-${Date.now()}.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Handle manual refresh
  const handleRefresh = () => {
    refetch();
    setRefreshKey((prev) => prev + 1);
  };

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    if (!isReady || !metrics) return null;

    const criticalPercentage =
      (metrics.criticalRegions / metrics.totalRegions) * 100 || 0;
    const highPerformancePercentage =
      (metrics.highPerformanceRegions / metrics.totalRegions) * 100 || 0;
    const investmentPerRegion =
      metrics.investmentNeeded / metrics.totalRegions || 0;

    return {
      criticalPercentage,
      highPerformancePercentage,
      investmentPerRegion,
    };
  }, [isReady, metrics]);

  // Error handling
  if (error) {
    return <ErrorDisplay error={error} onRetry={handleRefresh} />;
  }

  // Loading state
  if (isLoading) {
    return <DashboardSkeleton />;
  }

  // No data state
  if (!isReady) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <Card>
          <CardContent className="p-8 text-center">
            <Icons.database className="h-12 w-12 mx-auto mb-4 text-gray-400" />
            <h3 className="text-lg font-semibold mb-2">No Data Available</h3>
            <p className="text-gray-600 mb-4">
              No food security data found for the selected parameters.
            </p>
            <Button onClick={handleRefresh}>
              <Icons.refresh className="h-4 w-4 mr-2" />
              Retry Loading
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header Section */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
            <div className="flex-1">
              <div className="flex items-center space-x-3">
                <div className="h-12 w-12 bg-green-600 rounded-lg flex items-center justify-center">
                  <Icons.leaf className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">
                    Food Security Intelligence Dashboard
                  </h1>
                  <p className="text-gray-600">
                    Comprehensive analysis of agricultural food security across
                    Indonesian regions
                  </p>
                </div>
              </div>

              {/* Summary Status */}
              {/* Summary Status */}
              {summaryStats && (
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <div className="text-sm text-gray-600">
                    <span className="font-medium">{metrics.totalRegions}</span>{" "}
                    regions analyzed
                  </div>

                  <Separator orientation="vertical" className="h-4" />

                  <div className="text-sm text-gray-600">
                    Last updated: {new Date().toLocaleString()}
                  </div>
                </div>
              )}
            </div>

            {/* Controls */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              {/* Level Switch */}
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-700">
                  Level:
                </span>
                <Select
                  value={activeLevel}
                  onValueChange={(value) =>
                    switchLevel(value as "kabupaten" | "kecamatan")
                  }
                >
                  <SelectTrigger className="w-[120px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="kabupaten">Kabupaten</SelectItem>
                    <SelectItem value="kecamatan">Kecamatan</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Export */}
              <div className="flex items-center space-x-2">
                <Select
                  value={exportFormat}
                  onValueChange={(value) =>
                    setExportFormat(value as "json" | "csv")
                  }
                >
                  <SelectTrigger className="w-[80px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="json">JSON</SelectItem>
                    <SelectItem value="csv">CSV</SelectItem>
                  </SelectContent>
                </Select>

                <Button onClick={handleExport} variant="outline">
                  <Icons.download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>

              {/* Refresh */}
              <Button
                onClick={handleRefresh}
                variant="outline"
                disabled={isFetching}
              >
                <Icons.refresh
                  className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`}
                />
                {isFetching ? "Loading..." : "Refresh"}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Metrics Overview */}
        <Suspense
          fallback={<div className="h-32 bg-gray-200 rounded animate-pulse" />}
        >
          <MetricsCards analysisParams={analysisParams} level={activeLevel} />
        </Suspense>

        {/* Dashboard Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-1 lg:grid-cols-4">
            <TabsTrigger value="overview">Overview & Charts</TabsTrigger>
            <TabsTrigger value="rankings">Rankings & Performance</TabsTrigger>
            <TabsTrigger value="policy">Policy & Investment</TabsTrigger>
            <TabsTrigger value="analysis">Advanced Analysis</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <TimeSeriesChart
                  analysisParams={analysisParams}
                  level={activeLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <CorrelationScatter
                  analysisParams={analysisParams}
                  level={activeLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <EfficiencyMatrix
                  analysisParams={analysisParams}
                  level={activeLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <FSCIComponents
                  analysisParams={analysisParams}
                  level={activeLevel}
                />
              </Suspense>
            </div>
          </TabsContent>

          {/* Rankings Tab */}
          <TabsContent value="rankings" className="space-y-6">
            <Suspense
              fallback={
                <div className="h-64 bg-gray-200 rounded animate-pulse" />
              }
            >
              <RankingTable
                analysisParams={analysisParams}
                level={activeLevel}
              />
            </Suspense>
          </TabsContent>

          {/* Policy Tab */}
          <TabsContent value="policy" className="space-y-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Suspense
                fallback={
                  <div className="h-64 bg-gray-200 rounded animate-pulse" />
                }
              >
                <PolicyPanel
                  analysisParams={analysisParams}
                  level={activeLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-64 bg-gray-200 rounded animate-pulse" />
                }
              >
                <InvestmentPriority
                  analysisParams={analysisParams}
                  level={activeLevel}
                />
              </Suspense>
            </div>
          </TabsContent>

          {/* Advanced Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              {/* Combined Analysis View */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Icons.cpu className="h-5 w-5 mr-2" />
                    Comprehensive Analysis Dashboard
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* All components in one view */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">
                        Regional Performance
                      </h3>
                      <RankingTable
                        analysisParams={analysisParams}
                        level={activeLevel}
                      />
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">
                        Investment Priorities
                      </h3>
                      <InvestmentPriority
                        analysisParams={analysisParams}
                        level={activeLevel}
                      />
                    </div>
                  </div>

                  <Separator />

                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">
                        Policy Recommendations
                      </h3>
                      <PolicyPanel
                        analysisParams={analysisParams}
                        level={activeLevel}
                      />
                    </div>

                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">
                        Key Performance Metrics
                      </h3>
                      <MetricsCards
                        analysisParams={analysisParams}
                        level={activeLevel}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
