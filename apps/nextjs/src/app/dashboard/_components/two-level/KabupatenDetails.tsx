"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Icons } from "@/app/dashboard/_components/icons";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useTwoLevelAnalysis } from "@/hooks/use-twoLevelAnalysis";

interface KabupatenDetailsProps {
  className?: string;
  kabupatenName?: string;
  onClose?: () => void;
  onViewKecamatan?: (kabupatenName: string) => void;
}

export function KabupatenDetails({
  className,
  kabupatenName,
  onClose,
  onViewKecamatan,
}: KabupatenDetailsProps) {
  const { analysisData, selectedKabupaten, loading, error, selectKabupaten } =
    useTwoLevelAnalysis();

  // Use provided kabupatenName or selected from hook
  const activeKabupaten = kabupatenName || selectedKabupaten;

  // Get kabupaten data
  const kabupatenData = useMemo(() => {
    if (!analysisData?.level_2_kabupaten_analysis?.data || !activeKabupaten) {
      return null;
    }

    return analysisData.level_2_kabupaten_analysis.data.find(
      (kabupaten) => kabupaten.kabupaten_name === activeKabupaten
    );
  }, [analysisData, activeKabupaten]);

  // Get related kecamatan data
  const relatedKecamatan = useMemo(() => {
    if (!analysisData?.level_1_kecamatan_analysis?.data || !activeKabupaten) {
      return [];
    }

    return analysisData.level_1_kecamatan_analysis.data.filter(
      (kecamatan) => kecamatan.kabupaten_name === activeKabupaten
    );
  }, [analysisData, activeKabupaten]);

  // Calculate aggregate statistics
  const aggregateStats = useMemo(() => {
    if (relatedKecamatan.length === 0) return null;

    const fsciScores = relatedKecamatan.map((k) => k.fsci_score || 0);
    const pciScores = relatedKecamatan.map((k) => k.pci_score || 0);
    const psiScores = relatedKecamatan.map((k) => k.psi_score || 0);
    const crsScores = relatedKecamatan.map((k) => k.crs_score || 0);

    return {
      totalKecamatan: relatedKecamatan.length,
      avgFSCI: fsciScores.reduce((a, b) => a + b, 0) / fsciScores.length,
      avgPCI: pciScores.reduce((a, b) => a + b, 0) / pciScores.length,
      avgPSI: psiScores.reduce((a, b) => a + b, 0) / psiScores.length,
      avgCRS: crsScores.reduce((a, b) => a + b, 0) / crsScores.length,
      minFSCI: Math.min(...fsciScores),
      maxFSCI: Math.max(...fsciScores),

      // FSCI classification distribution
      lumbungPrimer: relatedKecamatan.filter((k) => (k.fsci_score || 0) >= 80)
        .length,
      lumbungSekunder: relatedKecamatan.filter(
        (k) => (k.fsci_score || 0) >= 60 && (k.fsci_score || 0) < 80
      ).length,
      lumbungTersier: relatedKecamatan.filter(
        (k) => (k.fsci_score || 0) >= 40 && (k.fsci_score || 0) < 60
      ).length,
      belowThreshold: relatedKecamatan.filter((k) => (k.fsci_score || 0) < 40)
        .length,
    };
  }, [relatedKecamatan]);

  // Performance indicators
  const getPerformanceColor = (category: string) => {
    switch (category) {
      case "overperforming":
        return "text-green-700 bg-green-100 border-green-300";
      case "aligned":
        return "text-blue-700 bg-blue-100 border-blue-300";
      case "underperforming":
        return "text-orange-700 bg-orange-100 border-orange-300";
      default:
        return "text-gray-700 bg-gray-100 border-gray-300";
    }
  };

  const getPerformanceIcon = (category: string) => {
    switch (category) {
      case "overperforming":
        return <Icons.trendingUp className="h-4 w-4" />;
      case "aligned":
        return <Icons.target className="h-4 w-4" />;
      case "underperforming":
        return <Icons.trendingDown className="h-4 w-4" />;
      default:
        return <Icons.arrowUpDown className="h-4 w-4" />;
    }
  };

  const getFSCIClassification = (score: number) => {
    if (score >= 80)
      return {
        label: "Lumbung Pangan Primer",
        color: "bg-green-500",
        textColor: "text-green-700",
      };
    if (score >= 60)
      return {
        label: "Lumbung Pangan Sekunder",
        color: "bg-yellow-500",
        textColor: "text-yellow-700",
      };
    if (score >= 40)
      return {
        label: "Lumbung Pangan Tersier",
        color: "bg-red-500",
        textColor: "text-red-700",
      };
    return {
      label: "Below Threshold",
      color: "bg-gray-500",
      textColor: "text-gray-700",
    };
  };

  const getCorrelationInterpretation = (correlation: number) => {
    const abs = Math.abs(correlation);
    if (abs >= 0.7) return { strength: "Strong", color: "text-green-600" };
    if (abs >= 0.5) return { strength: "Moderate", color: "text-blue-600" };
    if (abs >= 0.3) return { strength: "Weak", color: "text-orange-600" };
    return { strength: "Very Weak", color: "text-red-600" };
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading kabupaten details...</span>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (error) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            <Icons.alertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p>Error loading kabupaten details</p>
            <p className="text-sm text-gray-600 mt-1">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (!activeKabupaten) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.mapPin className="h-8 w-8 mx-auto mb-2" />
            <p className="font-medium">No Kabupaten Selected</p>
            <p className="text-sm mt-1">
              Select a kabupaten from the map or list to view detailed
              information
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (!kabupatenData) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.alertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p className="font-medium">Kabupaten Not Found</p>
            <p className="text-sm mt-1">
              No data available for {activeKabupaten}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  const fsciClassification = getFSCIClassification(
    kabupatenData.aggregated_fsci_score
  );
  const correlationInfo = getCorrelationInterpretation(
    kabupatenData.climate_production_correlation
  );
  return (
    <Card className={className}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center text-xl">
            <Icons.mapPin className="h-5 w-5 mr-2 text-blue-600" />
            {kabupatenData.kabupaten_name}
          </CardTitle>
          <div className="flex items-center space-x-2">
            {onViewKecamatan && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onViewKecamatan(kabupatenData.kabupaten_name)}
              >
                <Icons.users className="h-4 w-4 mr-1" />
                View Kecamatan
              </Button>
            )}
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                ×
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Performance Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* FSCI Score */}
          <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
            <div className="text-2xl font-bold text-blue-800">
              {kabupatenData.aggregated_fsci_score.toFixed(1)}
            </div>
            <div className="text-sm text-blue-600 mb-2">FSCI Score</div>
            <Badge className={`${fsciClassification.color} text-white text-xs`}>
              {fsciClassification.label}
            </Badge>
          </div>

          {/* Production */}
          <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
            <div className="text-2xl font-bold text-green-800">
              {(kabupatenData.latest_production_tons / 1000).toFixed(0)}K
            </div>
            <div className="text-sm text-green-600 mb-2">Production (tons)</div>
            <div className="text-xs text-green-700">
              Rank #{kabupatenData.actual_production_rank || "N/A"}
            </div>
          </div>

          {/* Performance Gap */}
          <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
            <div
              className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getPerformanceColor(
                kabupatenData.performance_gap_category
              )}`}
            >
              {getPerformanceIcon(kabupatenData.performance_gap_category)}
              <span className="ml-1 capitalize">
                {kabupatenData.performance_gap_category}
              </span>
            </div>
            <div className="text-xs text-purple-600 mt-2">
              Performance vs Potential
            </div>
          </div>
        </div>

        <Separator />

        {/* Detailed Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Climate-Production Analysis */}
          <div className="space-y-4">
            <h3 className="font-semibold flex items-center">
              <Icons.barChart className="h-4 w-4 mr-2" />
              Climate-Production Analysis
            </h3>

            <div className="space-y-3">
              {/* Correlation */}
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">
                  Climate-Production Correlation
                </span>
                <div className="flex items-center space-x-2">
                  <span className="font-medium">
                    {kabupatenData.climate_production_correlation.toFixed(3)}
                  </span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Badge
                          variant="outline"
                          className={correlationInfo.color}
                        >
                          {correlationInfo.strength}
                        </Badge>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          Correlation between climate suitability and actual
                          production
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              {/* Efficiency Score */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">
                    Production Efficiency
                  </span>
                  <span className="font-medium">
                    {(kabupatenData.production_efficiency_score * 100).toFixed(
                      1
                    )}
                    %
                  </span>
                </div>
                <Progress
                  value={kabupatenData.production_efficiency_score * 100}
                  className="h-2"
                />
                <div className="text-xs text-gray-500">
                  Actual production vs climate potential
                </div>
              </div>

              {/* Rankings */}
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-lg font-semibold text-gray-800">
                    #{kabupatenData.climate_potential_rank}
                  </div>
                  <div className="text-xs text-gray-600">Climate Rank</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-lg font-semibold text-gray-800">
                    #{kabupatenData.actual_production_rank || "N/A"}
                  </div>
                  <div className="text-xs text-gray-600">Production Rank</div>
                </div>
              </div>
            </div>
          </div>

          {/* FSCI Components Breakdown */}
          {aggregateStats && (
            <div className="space-y-4">
              <h3 className="font-semibold flex items-center">
                <Icons.zap className="h-4 w-4 mr-2" />
                FSCI Components (Average)
              </h3>

              <div className="space-y-3">
                {/* PCI */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <Icons.zap className="h-4 w-4 mr-1 text-blue-600" />
                      <span className="text-sm">
                        PCI (Precipitation Climate Index)
                      </span>
                    </div>
                    <span className="font-medium">
                      {aggregateStats.avgPCI.toFixed(1)}
                    </span>
                  </div>
                  <Progress value={aggregateStats.avgPCI} className="h-2" />
                </div>

                {/* PSI */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <Icons.thermometerSun className="h-4 w-4 mr-1 text-orange-600" />
                      <span className="text-sm">
                        PSI (Precipitation Stress Index)
                      </span>
                    </div>
                    <span className="font-medium">
                      {aggregateStats.avgPSI.toFixed(1)}
                    </span>
                  </div>
                  <Progress value={aggregateStats.avgPSI} className="h-2" />
                </div>

                {/* CRS */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <Icons.droplets className="h-4 w-4 mr-1 text-teal-600" />
                      <span className="text-sm">CRS (Climate Risk Score)</span>
                    </div>
                    <span className="font-medium">
                      {aggregateStats.avgCRS.toFixed(1)}
                    </span>
                  </div>
                  <Progress value={aggregateStats.avgCRS} className="h-2" />
                </div>
              </div>
            </div>
          )}
        </div>

        <Separator />

        {/* Kecamatan Summary */}
        {aggregateStats && (
          <div className="space-y-4">
            <h3 className="font-semibold flex items-center">
              <Icons.users className="h-4 w-4 mr-2" />
              Constituent Kecamatan Summary
            </h3>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="text-2xl font-bold text-green-700">
                  {aggregateStats.lumbungPrimer}
                </div>
                <div className="text-xs text-green-600">Lumbung Primer</div>
                <div className="text-xs text-gray-500">(≥80 FSCI)</div>
              </div>

              <div className="text-center p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                <div className="text-2xl font-bold text-yellow-700">
                  {aggregateStats.lumbungSekunder}
                </div>
                <div className="text-xs text-yellow-600">Lumbung Sekunder</div>
                <div className="text-xs text-gray-500">(60-79 FSCI)</div>
              </div>

              <div className="text-center p-3 bg-red-50 rounded-lg border border-red-200">
                <div className="text-2xl font-bold text-red-700">
                  {aggregateStats.lumbungTersier}
                </div>
                <div className="text-xs text-red-600">Lumbung Tersier</div>
                <div className="text-xs text-gray-500">(40-59 FSCI)</div>
              </div>

              <div className="text-center p-3 bg-gray-50 rounded-lg border border-gray-200">
                <div className="text-2xl font-bold text-gray-700">
                  {aggregateStats.belowThreshold}
                </div>
                <div className="text-xs text-gray-600">Below Threshold</div>
                <div className="text-xs text-gray-500">(&lt;40 FSCI)</div>
              </div>
            </div>

            {/* FSCI Range */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-lg font-semibold text-gray-800">
                    {aggregateStats.totalKecamatan}
                  </div>
                  <div className="text-xs text-gray-600">Total Kecamatan</div>
                </div>
                <div>
                  <div className="text-lg font-semibold text-red-600">
                    {aggregateStats.minFSCI.toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-600">Lowest FSCI</div>
                </div>
                <div>
                  <div className="text-lg font-semibold text-green-600">
                    {aggregateStats.maxFSCI.toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-600">Highest FSCI</div>
                </div>
              </div>
            </div>
          </div>
        )}

        <Separator />

        {/* Investment Insights */}
        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <h3 className="font-semibold flex items-center mb-3 text-blue-800">
            <Icons.award className="h-4 w-4 mr-2" />
            Investment Insights
          </h3>

          <div className="space-y-2 text-sm">
            {kabupatenData.performance_gap_category === "underperforming" && (
              <div className="flex items-start space-x-2">
                <Icons.alertTriangle className="h-4 w-4 text-orange-600 mt-0.5" />
                <div>
                  <span className="font-medium text-orange-800">
                    High Potential, Low Production:
                  </span>
                  <span className="text-gray-700 ml-1">
                    This kabupaten has strong climate suitability but
                    underperforms in actual production. Consider infrastructure
                    investment and farming technique improvements.
                  </span>
                </div>
              </div>
            )}

            {kabupatenData.performance_gap_category === "overperforming" && (
              <div className="flex items-start space-x-2">
                <Icons.trendingUp className="h-4 w-4 text-green-600 mt-0.5" />
                <div>
                  <span className="font-medium text-green-800">
                    High Performance:
                  </span>
                  <span className="text-gray-700 ml-1">
                    This kabupaten exceeds expectations based on climate
                    potential. Study best practices here for replication
                    elsewhere.
                  </span>
                </div>
              </div>
            )}

            {kabupatenData.performance_gap_category === "aligned" && (
              <div className="flex items-start space-x-2">
                <Icons.target className="h-4 w-4 text-blue-600 mt-0.5" />
                <div>
                  <span className="font-medium text-blue-800">
                    Balanced Performance:
                  </span>
                  <span className="text-gray-700 ml-1">
                    Production aligns well with climate potential. Focus on
                    maintaining current practices and gradual improvements.
                  </span>
                </div>
              </div>
            )}

            <div className="mt-3 pt-3 border-t border-blue-200">
              <span className="font-medium text-blue-800">
                Climate-Production Correlation:
              </span>
              <span className="text-gray-700 ml-1">
                {correlationInfo.strength} correlation (
                {kabupatenData.climate_production_correlation > 0
                  ? "positive"
                  : "negative"}
                ) suggests that climate factors{" "}
                {kabupatenData.climate_production_correlation > 0
                  ? "support"
                  : "may hinder"}
                production outcomes in this region.
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-2 pt-2">
          {onViewKecamatan && (
            <Button
              variant="outline"
              onClick={() => onViewKecamatan(kabupatenData.kabupaten_name)}
              className="flex-1 min-w-[200px]"
            >
              <Icons.users className="h-4 w-4 mr-2" />
              View {aggregateStats?.totalKecamatan || 0} Kecamatan Details
            </Button>
          )}

          <Button
            variant="ghost"
            onClick={() => {
              // Could open external resources or detailed reports
              console.log(
                "View detailed report for",
                kabupatenData.kabupaten_name
              );
            }}
            className="flex-1 min-w-[150px]"
          >
            <Icons.externalLink className="h-4 w-4 mr-2" />
            Detailed Report
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
export type { KabupatenDetailsProps };
