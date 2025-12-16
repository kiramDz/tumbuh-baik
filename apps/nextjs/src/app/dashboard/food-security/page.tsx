"use client";
import { Suspense, useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  KabupatenAnalysis,
  KecamatanAnalysis,
} from "@/lib/fetch/spatial.map.fetch";
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

// Chart components
import {
  TimeSeriesChart,
  CorrelationScatter,
  EfficiencyMatrix,
  FSCIComponents,
} from "../_components/chart";

// FSCI Spatial Components
import {
  FSCIMap,
  FSCIFilters,
  FSCILegend,
  FSCIMetadataPanel,
  type FSCIStats,
} from "../_components/spatial";

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

// Spatial Loading Skeleton
function SpatialLoadingSkeleton() {
  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-3 space-y-4">
        <div className="h-96 bg-gray-200 rounded animate-pulse" />
        <div className="h-64 bg-gray-200 rounded animate-pulse" />
      </div>
      <div className="col-span-12 lg:col-span-6">
        <div className="h-[600px] bg-gray-200 rounded animate-pulse" />
      </div>
      <div className="col-span-12 lg:col-span-3">
        <div className="h-96 bg-gray-200 rounded animate-pulse" />
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

  // Spatial state
  const [selectedRegion, setSelectedRegion] = useState<any>(null);
  const [spatialLevel, setSpatialLevel] = useState<"kabupaten" | "kecamatan">(
    "kabupaten"
  );
  const [selectedFSCIRange, setSelectedFSCIRange] = useState<
    "excellent" | "good" | "fair" | "poor" | null
  >(null);

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
    level: spatialLevel,
    autoRefresh: false,
  });

  // ✅ FIX: Use pre-extracted data from DashboardData
  const fsciStats = useMemo((): FSCIStats | undefined => {
    if (!isReady || !data) return undefined;

    // ✅ Option 1: Use pre-extracted kabupatenData / kecamatanData
    const sourceData =
      spatialLevel === "kabupaten" ? data.kabupatenData : data.kecamatanData;

    if (sourceData.length === 0) return undefined;

    // Calculate FSCI classification counts with proper typing
    const counts = sourceData.reduce(
      (
        acc: { excellent: number; good: number; fair: number; poor: number },
        item: any
      ) => {
        const fsciScore =
          spatialLevel === "kabupaten"
            ? item.aggregated_fsci_score
            : item.fsci_score;

        if (!fsciScore) return acc;

        if (fsciScore >= 75) acc.excellent++;
        else if (fsciScore >= 60) acc.good++;
        else if (fsciScore >= 45) acc.fair++;
        else acc.poor++;

        return acc;
      },
      { excellent: 0, good: 0, fair: 0, poor: 0 }
    );

    // Calculate average FSCI with proper typing
    const totalFsci = sourceData.reduce((sum: number, item: any) => {
      const fsciScore =
        spatialLevel === "kabupaten"
          ? item.aggregated_fsci_score
          : item.fsci_score;
      return sum + (fsciScore || 0);
    }, 0);

    return {
      total_regions: sourceData.length,
      avg_fsci: sourceData.length > 0 ? totalFsci / sourceData.length : 0,
      excellent_count: counts.excellent,
      good_count: counts.good,
      fair_count: counts.fair,
      poor_count: counts.poor,
    };
  }, [isReady, data, spatialLevel]);

  const transformedSelectedRegion = useMemo(() => {
    if (!selectedRegion || !data) return null;

    const regionName =
      selectedRegion.properties?.NAME_2 || selectedRegion.properties?.NAME_3;
    if (!regionName) return null;

    const sourceData =
      spatialLevel === "kabupaten" ? data.kabupatenData : data.kecamatanData;

    const matchingRegion = sourceData.find((item: any) => {
      const itemName =
        spatialLevel === "kabupaten"
          ? item.kabupaten_name
          : item.kecamatan_name;
      return itemName === regionName;
    });

    if (!matchingRegion) return null;

    // Transform to FSCIMetadataPanel expected format with type guards
    if (spatialLevel === "kabupaten") {
      const kabupatenRegion = matchingRegion as KabupatenAnalysis;
      return {
        kabupaten_name: kabupatenRegion.kabupaten_name,
        bps_compatible_name: kabupatenRegion.bps_compatible_name,
        total_area_km2: kabupatenRegion.total_area_km2,
        analysis_level: "kabupaten" as const,

        // Aggregated FSCI Scores
        aggregated_fsci_score: kabupatenRegion.aggregated_fsci_score,
        aggregated_fsci_class: kabupatenRegion.aggregated_fsci_class,
        aggregated_pci_score: kabupatenRegion.aggregated_pci_score,
        aggregated_psi_score: kabupatenRegion.aggregated_psi_score,
        aggregated_crs_score: kabupatenRegion.aggregated_crs_score,

        // Constituent analyses
        constituent_kecamatan: kabupatenRegion.constituent_kecamatan || [],
        constituent_nasa_locations:
          kabupatenRegion.constituent_nasa_locations || [],

        // Note: kecamatan_analyses might not exist in the interface, so we'll use fallback
        kecamatan_analyses: [],

        // Validation Data - BPS integration
        bps_validation: {
          kabupaten_name: kabupatenRegion.kabupaten_name,
          latest_production_tons: kabupatenRegion.latest_production_tons,
          average_production_tons: kabupatenRegion.average_production_tons,
          production_trend: kabupatenRegion.production_trend,
          data_years_available: [],
          data_coverage_years: kabupatenRegion.data_coverage_years || 0,
        },

        climate_production_correlation:
          kabupatenRegion.climate_production_correlation,
        production_efficiency_score:
          kabupatenRegion.production_efficiency_score,

        // Rankings and Performance
        climate_potential_rank: kabupatenRegion.climate_potential_rank,
        actual_production_rank: kabupatenRegion.actual_production_rank,
        performance_gap_category: kabupatenRegion.performance_gap_category,

        //  Access metadata from rawData
        analysis_timestamp: data.rawData?.metadata?.analysis_timestamp,
        validation_notes: kabupatenRegion.validation_notes,
      };
    } else {
      // Assert this is KecamatanAnalysis
      const kecamatanRegion = matchingRegion as KecamatanAnalysis;
      return {
        kabupaten_name: kecamatanRegion.kabupaten_name,
        bps_compatible_name: kecamatanRegion.kabupaten_name, // Map to available field
        total_area_km2: kecamatanRegion.area_km2,
        analysis_level: "kecamatan" as const,

        aggregated_fsci_score: kecamatanRegion.fsci_score || 0,
        aggregated_fsci_class: kecamatanRegion.fsci_class || "fair",
        aggregated_pci_score: kecamatanRegion.pci_score || 0,
        aggregated_psi_score: kecamatanRegion.psi_score || 0,
        aggregated_crs_score: kecamatanRegion.crs_score || 0,

        constituent_kecamatan: [kecamatanRegion.kecamatan_name],
        constituent_nasa_locations: [kecamatanRegion.nasa_location_name],

        kecamatan_analyses: [],

        bps_validation: undefined,
        climate_production_correlation: 0,
        production_efficiency_score: 0,

        climate_potential_rank: 0,
        actual_production_rank: 0,
        performance_gap_category: "aligned" as const,

        analysis_timestamp: data.rawData?.metadata?.analysis_timestamp,
        validation_notes: "Kecamatan-level analysis based on NASA climate data",
      };
    }
  }, [selectedRegion, data, spatialLevel]);

  // Spatial event handlers
  const handleFeatureClick = (feature: any) => {
    setSelectedRegion(feature);
  };

  const handleSpatialLevelChange = (level: "kabupaten" | "kecamatan") => {
    setSpatialLevel(level);
    setSelectedRegion(null);
    switchLevel(level);
  };

  const handleFSCIRangeFilter = (
    range: "excellent" | "good" | "fair" | "poor" | null
  ) => {
    setSelectedFSCIRange(range);
  };

  const handleExportSpatialData = (region: any) => {
    const exportData = JSON.stringify(region, null, 2);
    const blob = new Blob([exportData], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fsci-region-${region.kabupaten_name.replace(
      /\s+/g,
      "-"
    )}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

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
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header Section */}
        <div className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="animate-pulse">
              <div className="h-8 bg-gray-200 rounded w-1/3 mb-2" />
              <div className="h-4 bg-gray-200 rounded w-1/2" />
            </div>
          </div>
        </div>

        {/* Content Loading */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <DashboardSkeleton />
        </div>
      </div>
    );
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
                    Food Security Analysis Dashboard
                  </h1>
                  <p className="text-gray-600">
                    Spatial analysis of agricultural food security with FSCI
                    methodology
                  </p>
                </div>
              </div>

              {/* Summary Status */}
              {summaryStats && (
                <div className="mt-4 flex flex-wrap items-center gap-3">
                  <div className="text-sm text-gray-600">
                    <span className="font-medium">{metrics.totalRegions}</span>{" "}
                    regions analyzed
                  </div>

                  <Separator orientation="vertical" className="h-4" />

                  <div className="text-sm text-gray-600">
                    Analysis Level:{" "}
                    <span className="font-medium capitalize">
                      {spatialLevel}
                    </span>
                  </div>

                  <Separator orientation="vertical" className="h-4" />

                  <div className="text-sm text-gray-600">
                    Last updated: {new Date().toLocaleString()}
                  </div>
                </div>
              )}
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
          <MetricsCards analysisParams={analysisParams} level={spatialLevel} />
        </Suspense>

        {/* Dashboard Tabs */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-1 lg:grid-cols-3">
            <TabsTrigger value="overview">Overview & Charts</TabsTrigger>
            <TabsTrigger value="rankings">Rankings & Performance</TabsTrigger>
            <TabsTrigger value="spatial">Spatial Analysis</TabsTrigger>
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
                  level={spatialLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <CorrelationScatter
                  analysisParams={analysisParams}
                  level={spatialLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <EfficiencyMatrix
                  analysisParams={analysisParams}
                  level={spatialLevel}
                />
              </Suspense>

              <Suspense
                fallback={
                  <div className="h-96 bg-gray-200 rounded animate-pulse" />
                }
              >
                <FSCIComponents
                  analysisParams={analysisParams}
                  level={spatialLevel}
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
                level={spatialLevel}
              />
            </Suspense>
          </TabsContent>

          <TabsContent value="spatial" className="space-y-6">
            <div className="grid grid-cols-12 gap-6">
              {/* FSCI Filters Sidebar */}
              <div className="col-span-12 lg:col-span-3">
                <div className="space-y-4">
                  <Suspense
                    fallback={
                      <div className="h-96 bg-gray-200 rounded animate-pulse" />
                    }
                  >
                    <FSCIFilters
                      analysisParams={analysisParams}
                      level={spatialLevel}
                      onParamsChange={(params) => {
                        setAnalysisParams(params);
                        setRefreshKey((prev) => prev + 1);
                      }}
                      onLevelChange={handleSpatialLevelChange}
                      onReset={() => {
                        setAnalysisParams({
                          year_start: 2018,
                          year_end: 2024,
                          bps_start_year: 2018,
                          bps_end_year: 2024,
                          season: "all",
                          aggregation: "mean",
                          districts: "all",
                        });
                        setRefreshKey((prev) => prev + 1);
                      }}
                      isLoading={isFetching}
                    />
                  </Suspense>

                  <Suspense
                    fallback={
                      <div className="h-64 bg-gray-200 rounded animate-pulse" />
                    }
                  >
                    <FSCILegend
                      stats={fsciStats}
                      selectedRange={selectedFSCIRange}
                      onRangeSelect={handleFSCIRangeFilter}
                      compact={true}
                      interactive={true}
                      isLoading={isFetching}
                    />
                  </Suspense>
                </div>
              </div>

              {/* Main FSCI Map */}
              <div className="col-span-12 lg:col-span-6">
                <Card>
                  <CardContent className="p-0">
                    <div className="h-[600px] w-full">
                      <Suspense
                        fallback={
                          <div className="h-full bg-gray-100 flex items-center justify-center">
                            <div className="text-center">
                              <Icons.spinner className="h-8 w-8 animate-spin text-gray-400 mx-auto mb-4" />
                              <p className="text-gray-600">
                                Loading FSCI map...
                              </p>
                            </div>
                          </div>
                        }
                      >
                        <FSCIMap
                          analysisParams={analysisParams}
                          level={spatialLevel}
                          onFeatureClick={handleFeatureClick}
                          selectedRegion={selectedRegion?.properties?.NAME_2}
                          className="rounded-lg overflow-hidden"
                        />
                      </Suspense>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Regional Metadata Panel */}
              <div className="col-span-12 lg:col-span-3">
                <Suspense
                  fallback={
                    <div className="h-96 bg-gray-200 rounded animate-pulse" />
                  }
                >
                  <FSCIMetadataPanel
                    selectedRegion={transformedSelectedRegion}
                    level={spatialLevel}
                    mode="sidebar"
                    onClose={() => setSelectedRegion(null)}
                    onExport={handleExportSpatialData}
                    isLoading={isFetching}
                  />
                </Suspense>
              </div>
            </div>

            {/* Full-width Metadata Panel when region is selected */}
            {transformedSelectedRegion && (
              <Suspense
                fallback={
                  <div className="h-64 bg-gray-200 rounded animate-pulse" />
                }
              >
                <FSCIMetadataPanel
                  selectedRegion={transformedSelectedRegion}
                  level={spatialLevel}
                  mode="full"
                  showDetails={true}
                  showTrends={true}
                  onClose={() => setSelectedRegion(null)}
                  onExport={handleExportSpatialData}
                  isLoading={isFetching}
                />
              </Suspense>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
