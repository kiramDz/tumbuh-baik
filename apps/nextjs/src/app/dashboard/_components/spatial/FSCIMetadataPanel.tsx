"use client";
import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

interface KecamatanMetadata {
  nasa_location_name: string;
  kecamatan_name: string;
  area_km2: number;
  area_weight: number;
  fsci_analysis: {
    fsci_score: number;
    fsci_class: string;
    pci_score: number;
    psi_score: number;
    crs_score: number;
  };
}

interface BPSValidationData {
  kabupaten_name: string;
  latest_production_tons: number;
  average_production_tons: number;
  production_trend: string;
  data_years_available: number[];
  data_coverage_years: number;
}

interface RegionalMetadata {
  // Basic information
  kabupaten_name: string;
  bps_compatible_name: string;
  total_area_km2: number;
  analysis_level: "kabupaten" | "kecamatan";

  // Aggregated FSCI Scores
  aggregated_fsci_score: number;
  aggregated_fsci_class: string;
  aggregated_pci_score: number;
  aggregated_psi_score: number;
  aggregated_crs_score: number;

  // Constituent Analysis
  constituent_kecamatan: string[];
  constituent_nasa_locations: string[];
  kecamatan_analyses?: KecamatanMetadata[];

  // Validation Data
  bps_validation?: BPSValidationData;
  climate_production_correlation: number;
  production_efficiency_score: number;

  // Rankings and Performance
  climate_potential_rank: number;
  actual_production_rank: number;
  performance_gap_category: string;

  // Analysis Metadata
  analysis_timestamp?: string;
  validation_notes?: string;
}

export interface FSCIMetadataPanelProps {
  selectedRegion?: RegionalMetadata | null;
  level?: "kabupaten" | "kecamatan";
  mode?: "full" | "compact" | "sidebar";
  showDetails?: boolean;
  showTrends?: boolean;
  isLoading?: boolean;
  onClose?: () => void;
  onExport?: (region: RegionalMetadata) => void;
  className?: string;
}

export function FSCIMetadataPanel({
  selectedRegion,
  level = "kabupaten",
  mode = "full",
  showDetails = true,
  showTrends = true,
  isLoading = false,
  onClose,
  onExport,
  className,
}: FSCIMetadataPanelProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [expandedKecamatan, setExpandedKecamatan] = useState<string | null>(
    null
  );

  // Performance classification based on FSCI score
  const getPerformanceConfig = (score: number) => {
    if (score >= 75)
      return {
        level: "Excellent",
        color: "#059669",
        bgColor: "#D1FAE5",
        icon: Icons.trendingUp,
      };
    if (score >= 60)
      return {
        level: "Good",
        color: "#3B82F6",
        bgColor: "#DBEAFE",
        icon: Icons.checkCircle,
      };
    if (score >= 45)
      return {
        level: "Fair",
        color: "#F59E0B",
        bgColor: "#FEF3C7",
        icon: Icons.alertTriangle,
      };
    return {
      level: "Poor",
      color: "#DC2626",
      bgColor: "#FEE2E2",
      icon: Icons.alertCircle,
    };
  };

  // Calculate consistent statistics
  const constituentStats = useMemo(() => {
    if (!selectedRegion?.kecamatan_analyses) return null;

    const analyses = selectedRegion.kecamatan_analyses;
    const fsciScores = analyses.map((a) => a.fsci_analysis.fsci_score);

    return {
      count: analyses.length,
      fsci_range: {
        min: Math.min(...fsciScores),
        max: Math.max(...fsciScores),
        avg:
          fsciScores.reduce((sum, score) => sum + score, 0) / fsciScores.length,
      },
      area_distribution: analyses
        .map((a) => ({
          name: a.kecamatan_name,
          area_km2: a.area_km2,
          area_weight: a.area_weight,
          fsci_score: a.fsci_analysis.fsci_score,
        }))
        .sort((a, b) => b.area_weight - a.area_weight),
    };
  }, [selectedRegion]);

  // Production trend analysis
  const getProductionTrendConfig = (trend: string) => {
    switch (trend) {
      case "increasing":
        return {
          label: "Increasing",
          color: "#059669",
          icon: Icons.trendingUp,
        };
      case "decreasing":
        return {
          label: "Decreasing",
          color: "#DC2626",
          icon: Icons.trendingDown,
        };
      case "stable":
        return { label: "Stable", color: "#3B82F6", icon: Icons.minus };
      default:
        return { label: "Unknown", color: "#6B7280", icon: Icons.question };
    }
  };

  // Performance gap analysis
  const getPerformanceGapConfig = (category: string) => {
    switch (category) {
      case "overperforming":
        return {
          label: "Overperforming",
          description: "Production exceeds climate potential",
          color: "#059669",
          icon: Icons.trendingUp,
        };
      case "underperforming":
        return {
          label: "Underperforming",
          description: "Production below climate potential",
          color: "#DC2626",
          icon: Icons.trendingDown,
        };
      case "aligned":
        return {
          label: "Well Aligned",
          description: "Production matches climate potential",
          color: "#3B82F6",
          icon: Icons.checkCircle,
        };
      default:
        return {
          label: "Unknown",
          description: "Insufficient data for analysis",
          color: "#6B7280",
          icon: Icons.question,
        };
    }
  };
  // No selection state
  if (!selectedRegion) {
    return (
      <Card className={cn("w-full", className)}>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <div className="text-4xl mb-4">🗺️</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Select a Region
          </h3>
          <p className="text-sm text-gray-500 text-center max-w-sm">
            Click on a region in the map to view detailed FSCI analysis,
            production data, and climate correlation metrics.
          </p>
        </CardContent>
      </Card>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <Card className={cn("w-full", className)}>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Icons.spinner className="h-8 w-8 animate-spin text-gray-400 mx-auto mb-4" />
            <p className="text-sm text-gray-500">
              Loading regional metadata...
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const performanceConfig = getPerformanceConfig(
    selectedRegion.aggregated_fsci_score
  );
  const PerformanceIcon = performanceConfig.icon;

  // Compact mode rendering
  if (mode === "compact" || mode === "sidebar") {
    return (
      <Card className={cn("w-full", className)}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base truncate">
              {selectedRegion.kabupaten_name}
            </CardTitle>
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                <Icons.closeX className="h-4 w-4" />
              </Button>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {/* FSCI Score Display */}
          <div
            className="p-4 rounded-lg"
            style={{ backgroundColor: performanceConfig.bgColor }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-gray-600">FSCI Score</div>
                <div
                  className="text-2xl font-bold"
                  style={{ color: performanceConfig.color }}
                >
                  {selectedRegion.aggregated_fsci_score.toFixed(1)}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <PerformanceIcon
                  className="h-5 w-5"
                  style={{ color: performanceConfig.color }}
                />
                <Badge
                  variant="secondary"
                  style={{
                    backgroundColor: performanceConfig.color,
                    color: "white",
                  }}
                >
                  {performanceConfig.level}
                </Badge>
              </div>
            </div>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center">
              <div className="text-xs text-gray-500">Production Rank</div>
              <div className="text-lg font-semibold">
                #{selectedRegion.actual_production_rank}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-gray-500">Climate Rank</div>
              <div className="text-lg font-semibold">
                #{selectedRegion.climate_potential_rank}
              </div>
            </div>
          </div>

          {/* Production Data */}
          {selectedRegion.bps_validation && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-gray-700">
                Latest Production
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold">
                  {(
                    selectedRegion.bps_validation.latest_production_tons / 1000
                  ).toFixed(1)}
                  K tons
                </span>
                {(() => {
                  const trendConfig = getProductionTrendConfig(
                    selectedRegion.bps_validation.production_trend
                  );
                  const TrendIcon = trendConfig.icon;
                  return (
                    <div className="flex items-center space-x-1">
                      <TrendIcon
                        className="h-4 w-4"
                        style={{ color: trendConfig.color }}
                      />
                      <span
                        className="text-xs"
                        style={{ color: trendConfig.color }}
                      >
                        {trendConfig.label}
                      </span>
                    </div>
                  );
                })()}
              </div>
            </div>
          )}

          {/* Correlation */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-gray-700">
              Climate-Production Correlation
            </div>
            <div className="flex items-center space-x-2">
              <Progress
                value={selectedRegion.climate_production_correlation * 100}
                className="flex-1"
              />
              <span className="text-sm font-medium">
                {(selectedRegion.climate_production_correlation * 100).toFixed(
                  0
                )}
                %
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Full mode rendering
  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">
            {selectedRegion.kabupaten_name}
          </CardTitle>
          <div className="flex items-center space-x-2">
            {onExport && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onExport(selectedRegion)}
              >
                <Icons.download className="h-4 w-4 mr-2" />
                Export
              </Button>
            )}
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                <Icons.closeX className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
        <div className="text-sm text-gray-600">
          {selectedRegion.analysis_level === "kabupaten"
            ? "District-level Analysis"
            : "Sub-district-level Analysis"}{" "}
          • {selectedRegion.total_area_km2.toFixed(0)} km²
        </div>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="fsci">FSCI Analysis</TabsTrigger>
            <TabsTrigger value="production">Production</TabsTrigger>
            <TabsTrigger value="constituents">Components</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* FSCI Performance Card */}
            <div
              className="p-6 rounded-lg"
              style={{ backgroundColor: performanceConfig.bgColor }}
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <div className="text-sm text-gray-600 mb-1">
                    Overall FSCI Score
                  </div>
                  <div
                    className="text-4xl font-bold"
                    style={{ color: performanceConfig.color }}
                  >
                    {selectedRegion.aggregated_fsci_score.toFixed(1)}
                  </div>
                </div>
                <div className="text-center">
                  <PerformanceIcon
                    className="h-8 w-8 mx-auto mb-2"
                    style={{ color: performanceConfig.color }}
                  />
                  <Badge
                    variant="secondary"
                    style={{
                      backgroundColor: performanceConfig.color,
                      color: "white",
                    }}
                  >
                    {performanceConfig.level} Performance
                  </Badge>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mt-4">
                <div className="text-center">
                  <div className="text-sm text-gray-600">PCI Score</div>
                  <div className="text-xl font-semibold">
                    {selectedRegion.aggregated_pci_score.toFixed(1)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-600">PSI Score</div>
                  <div className="text-xl font-semibold">
                    {selectedRegion.aggregated_psi_score.toFixed(1)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-600">CRS Score</div>
                  <div className="text-xl font-semibold">
                    {selectedRegion.aggregated_crs_score.toFixed(1)}
                  </div>
                </div>
              </div>
            </div>

            {/* Rankings and Performance */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Climate Potential Ranking */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center">
                    <Icons.cloudSun className="h-5 w-5 mr-2" />
                    Climate Ranking
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-600">
                      #{selectedRegion.climate_potential_rank}
                    </div>
                    <div className="text-sm text-gray-600">
                      Climate Potential Rank
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Production Ranking */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center">
                    <Icons.barChart className="h-5 w-5 mr-2" />
                    Production Ranking
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600">
                      #{selectedRegion.actual_production_rank}
                    </div>
                    <div className="text-sm text-gray-600">
                      Actual Production Rank
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Performance Gap Analysis */}
            {(() => {
              const gapConfig = getPerformanceGapConfig(
                selectedRegion.performance_gap_category
              );
              const GapIcon = gapConfig.icon;
              return (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center">
                      <Icons.target className="h-5 w-5 mr-2" />
                      Performance Gap Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center space-x-3">
                      <GapIcon
                        className="h-6 w-6"
                        style={{ color: gapConfig.color }}
                      />
                      <div>
                        <div
                          className="font-semibold"
                          style={{ color: gapConfig.color }}
                        >
                          {gapConfig.label}
                        </div>
                        <div className="text-sm text-gray-600">
                          {gapConfig.description}
                        </div>
                      </div>
                    </div>
                    <div className="mt-4">
                      <div className="text-sm text-gray-600 mb-1">
                        Production Efficiency Score
                      </div>
                      <Progress
                        value={selectedRegion.production_efficiency_score}
                        className="h-2"
                      />
                      <div className="text-xs text-gray-500 mt-1">
                        {selectedRegion.production_efficiency_score.toFixed(1)}%
                        efficiency
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })()}
          </TabsContent>

          {/* FSCI Analysis Tab */}
          <TabsContent value="fsci" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* PCI Breakdown */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base text-center">
                    Production Capacity Index
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-2">
                    {selectedRegion.aggregated_pci_score.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-600">
                    Climate suitability and land quality assessment
                  </div>
                </CardContent>
              </Card>

              {/* PSI Breakdown */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base text-center">
                    Production Stability Index
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <div className="text-3xl font-bold text-green-600 mb-2">
                    {selectedRegion.aggregated_psi_score.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-600">
                    Temporal stability and trend consistency
                  </div>
                </CardContent>
              </Card>

              {/* CRS Breakdown */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base text-center">
                    Climate Resilience Score
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <div className="text-3xl font-bold text-orange-600 mb-2">
                    {selectedRegion.aggregated_crs_score.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-600">
                    Adaptation capacity and extreme weather resilience
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Climate-Production Correlation */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center">
                  <Icons.activity className="h-5 w-5 mr-2" />
                  Climate-Production Correlation Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-gray-600">
                        Correlation Strength
                      </span>
                      <span className="text-lg font-semibold">
                        {(
                          selectedRegion.climate_production_correlation * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <Progress
                      value={
                        selectedRegion.climate_production_correlation * 100
                      }
                      className="h-3"
                    />
                  </div>

                  <div className="text-sm text-gray-600">
                    <strong>Interpretation:</strong>{" "}
                    {selectedRegion.climate_production_correlation > 0.7
                      ? "Strong positive correlation between climate potential and actual production."
                      : selectedRegion.climate_production_correlation > 0.4
                      ? "Moderate correlation suggests room for optimization."
                      : "Low correlation indicates potential inefficiencies or external factors."}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Production Tab */}
          <TabsContent value="production" className="space-y-4">
            {selectedRegion.bps_validation ? (
              <>
                {/* Production Overview */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">
                        Latest Production (2024)
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-green-600 mb-2">
                        {(
                          selectedRegion.bps_validation.latest_production_tons /
                          1000
                        ).toFixed(1)}
                        K
                      </div>
                      <div className="text-sm text-gray-600">
                        tons of rice production
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base">
                        Average Production
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-blue-600 mb-2">
                        {(
                          selectedRegion.bps_validation
                            .average_production_tons / 1000
                        ).toFixed(1)}
                        K
                      </div>
                      <div className="text-sm text-gray-600">
                        tons historical average
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Production Trend */}
                {(() => {
                  const trendConfig = getProductionTrendConfig(
                    selectedRegion.bps_validation.production_trend
                  );
                  const TrendIcon = trendConfig.icon;
                  return (
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base flex items-center">
                          <Icons.trendingUp className="h-5 w-5 mr-2" />
                          Production Trend Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center space-x-3 mb-4">
                          <TrendIcon
                            className="h-6 w-6"
                            style={{ color: trendConfig.color }}
                          />
                          <div>
                            <div
                              className="font-semibold"
                              style={{ color: trendConfig.color }}
                            >
                              {trendConfig.label} Trend
                            </div>
                            <div className="text-sm text-gray-600">
                              Based on{" "}
                              {
                                selectedRegion.bps_validation
                                  .data_coverage_years
                              }{" "}
                              years of BPS data
                            </div>
                          </div>
                        </div>

                        <div className="text-sm text-gray-600">
                          <strong>Data Coverage:</strong>{" "}
                          {selectedRegion.bps_validation.data_years_available.join(
                            ", "
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })()}

                {/* Production Efficiency */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center">
                      <Icons.target className="h-5 w-5 mr-2" />
                      Production Efficiency Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-600">
                            Efficiency Score
                          </span>
                          <span className="text-lg font-semibold">
                            {selectedRegion.production_efficiency_score.toFixed(
                              1
                            )}
                            %
                          </span>
                        </div>
                        <Progress
                          value={selectedRegion.production_efficiency_score}
                          className="h-3"
                        />
                      </div>

                      <div className="text-sm text-gray-600">
                        <strong>Analysis:</strong>{" "}
                        {selectedRegion.production_efficiency_score > 100
                          ? "Production exceeds climate-based expectations, indicating excellent agricultural practices."
                          : selectedRegion.production_efficiency_score > 80
                          ? "Good production efficiency relative to climate potential."
                          : "Production below climate potential, suggesting opportunities for improvement."}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card>
                <CardContent className="text-center py-8">
                  <Icons.alertTriangle className="h-8 w-8 text-yellow-500 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    No Production Data Available
                  </h3>
                  <p className="text-sm text-gray-600">
                    BPS production validation data is not available for this
                    region.
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Constituents Tab */}
          <TabsContent value="constituents" className="space-y-4">
            {constituentStats && (
              <>
                {/* Constituent Overview */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base text-center">
                        Total Kecamatan
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-center">
                      <div className="text-3xl font-bold text-blue-600">
                        {constituentStats.count}
                      </div>
                      <div className="text-sm text-gray-600">
                        constituent sub-districts
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base text-center">
                        FSCI Range
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-center">
                      <div className="text-lg font-bold text-gray-900">
                        {constituentStats.fsci_range.min.toFixed(1)} -{" "}
                        {constituentStats.fsci_range.max.toFixed(1)}
                      </div>
                      <div className="text-sm text-gray-600">
                        average: {constituentStats.fsci_range.avg.toFixed(1)}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-base text-center">
                        Area Coverage
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="text-center">
                      <div className="text-lg font-bold text-green-600">
                        {selectedRegion.total_area_km2.toFixed(0)} km²
                      </div>
                      <div className="text-sm text-gray-600">total area</div>
                    </CardContent>
                  </Card>
                </div>

                {/* Detailed Kecamatan List */}
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center">
                      <Icons.list className="h-5 w-5 mr-2" />
                      Constituent Kecamatan Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ScrollArea className="h-64">
                      <div className="space-y-2">
                        {constituentStats.area_distribution.map(
                          (kecamatan, index) => {
                            const performanceConfig = getPerformanceConfig(
                              kecamatan.fsci_score
                            );
                            return (
                              <Collapsible
                                key={index}
                                open={expandedKecamatan === kecamatan.name}
                                onOpenChange={(open) =>
                                  setExpandedKecamatan(
                                    open ? kecamatan.name : null
                                  )
                                }
                              >
                                <CollapsibleTrigger className="w-full">
                                  <div className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 transition-colors">
                                    <div className="flex items-center space-x-3">
                                      <div
                                        className="w-3 h-3 rounded-full"
                                        style={{
                                          backgroundColor:
                                            performanceConfig.color,
                                        }}
                                      />
                                      <div className="text-left">
                                        <div className="font-medium">
                                          {kecamatan.name}
                                        </div>
                                        <div className="text-sm text-gray-600">
                                          {kecamatan.area_km2.toFixed(1)} km² •{" "}
                                          {(
                                            kecamatan.area_weight * 100
                                          ).toFixed(1)}
                                          % weight
                                        </div>
                                      </div>
                                    </div>
                                    <div className="text-right">
                                      <div
                                        className="font-semibold"
                                        style={{
                                          color: performanceConfig.color,
                                        }}
                                      >
                                        {kecamatan.fsci_score.toFixed(1)}
                                      </div>
                                      <div className="text-xs text-gray-500">
                                        FSCI Score
                                      </div>
                                    </div>
                                  </div>
                                </CollapsibleTrigger>
                                <CollapsibleContent>
                                  <div className="p-4 bg-gray-50 rounded-b-lg">
                                    <div className="text-sm text-gray-600">
                                      <p>
                                        <strong>Performance:</strong>{" "}
                                        {performanceConfig.level}
                                      </p>
                                      <p>
                                        <strong>Contribution:</strong>{" "}
                                        {(kecamatan.area_weight * 100).toFixed(
                                          1
                                        )}
                                        % of total kabupaten area
                                      </p>
                                      <p>
                                        <strong>Area:</strong>{" "}
                                        {kecamatan.area_km2.toFixed(1)} km²
                                      </p>
                                    </div>
                                  </div>
                                </CollapsibleContent>
                              </Collapsible>
                            );
                          }
                        )}
                      </div>
                    </ScrollArea>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>

        {/* Analysis Notes */}
        {selectedRegion.validation_notes && (
          <>
            <Separator className="my-6" />
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="flex items-start space-x-3">
                <Icons.info className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <div className="font-medium text-blue-900 mb-1">
                    Analysis Notes
                  </div>
                  <div className="text-sm text-blue-800">
                    {selectedRegion.validation_notes}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Timestamp */}
        {selectedRegion.analysis_timestamp && (
          <div className="text-xs text-gray-500 text-center mt-4">
            Analysis performed on:{" "}
            {new Date(selectedRegion.analysis_timestamp).toLocaleString()}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
