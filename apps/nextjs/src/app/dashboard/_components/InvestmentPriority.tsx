"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface InvestmentPriorityProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  showMatrix?: boolean;
  showBudgetAllocation?: boolean;
  maxPriorities?: number;
}

interface BudgetAllocation {
  total_investment: number;
  critical_regions: PriorityRegion[];
  high_priority_regions: PriorityRegion[];
  total_beneficiaries: number;
  average_roi: number;
}

interface PriorityRegion {
  id: string;
  name: string;
  fsci_score: number;
  pci_score: number;
  psi_score: number;
  crs_score: number;
  production_potential: number;
  current_production: number;
  population: number;
  priority_score: number; // 0-100
  priority_category: "critical" | "high" | "medium" | "low";
  investment_need: number; // USD
  expected_roi: number; // Return on Investment %
  risk_level: "low" | "medium" | "high";
  intervention_type: string[];
  time_to_impact: number; // months
  beneficiaries: number;
}

interface PriorityMatrix {
  quadrant:
    | "high_impact_low_cost"
    | "high_impact_high_cost"
    | "low_impact_low_cost"
    | "low_impact_high_cost";
  regions: PriorityRegion[];
  total_investment: number;
  expected_impact: number;
  priority_level: number;
}

export function InvestmentPriority({
  className,
  analysisParams,
  level = "kabupaten",
  showMatrix = true,
  showBudgetAllocation = true,
  maxPriorities = 20,
}: InvestmentPriorityProps) {
  const [selectedMetric, setSelectedMetric] = useState<
    "fsci" | "roi" | "impact" | "urgency"
  >("fsci");
  const [selectedQuadrant, setSelectedQuadrant] = useState<string>("all");
  const [sortBy, setSortBy] = useState<
    "priority" | "roi" | "investment" | "beneficiaries"
  >("priority");

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

  const { priorityRegions, priorityMatrix, budgetAllocation } = useMemo((): {
    priorityRegions: PriorityRegion[];
    priorityMatrix: PriorityMatrix[];
    budgetAllocation: BudgetAllocation;
  } => {
    // Default return object with proper typing
    const defaultReturn = {
      priorityRegions: [],
      priorityMatrix: [],
      budgetAllocation: {
        total_investment: 0,
        critical_regions: [],
        high_priority_regions: [],
        total_beneficiaries: 0,
        average_roi: 0,
      } as BudgetAllocation,
    };

    if (!analysisData) return defaultReturn;

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    if (sourceData.length === 0) return defaultReturn;

    // Helper function to safely get values
    const getValue = (item: any, ...keys: string[]): number => {
      for (const key of keys) {
        const value = item[key];
        if (typeof value === "number" && !isNaN(value) && value >= 0)
          return value;
      }
      return 0;
    };

    // Process regions and calculate priority scores
    const priorityRegions: PriorityRegion[] = sourceData
      .map((item: any, index: number) => {
        const fsci =
          level === "kabupaten"
            ? getValue(item, "aggregated_fsci_score", "fsci_score", "fsci_mean")
            : getValue(
                item,
                "fsci_score",
                "fsci_mean",
                "aggregated_fsci_score"
              );

        const pci =
          level === "kabupaten"
            ? getValue(item, "aggregated_pci_score", "pci_score", "pci_mean")
            : getValue(item, "pci_score", "pci_mean");

        const psi =
          level === "kabupaten"
            ? getValue(item, "aggregated_psi_score", "psi_score", "psi_mean")
            : getValue(item, "psi_score", "psi_mean");

        const crs =
          level === "kabupaten"
            ? getValue(item, "aggregated_crs_score", "crs_score", "crs_mean")
            : getValue(item, "crs_score", "crs_mean");

        const currentProduction =
          level === "kabupaten"
            ? getValue(
                item,
                "latest_production_tons",
                "average_production_tons",
                "total_production"
              )
            : 0;

        // Calculate priority metrics
        const urgencyScore = 100 - fsci; // Lower FSCI = Higher urgency
        const potentialImpact = (100 - fsci) * (currentProduction / 1000); // Urgency * Production scale
        const climateChallenges = 100 - pci + (100 - psi) + (100 - crs); // Sum of climate deficits

        // Priority calculation (0-100 scale)
        const priorityScore = Math.min(
          100,
          urgencyScore * 0.4 +
            (climateChallenges / 3) * 0.3 +
            (potentialImpact / 10) * 0.3
        );

        // Determine priority category
        const getPriorityCategory = (
          score: number
        ): PriorityRegion["priority_category"] => {
          if (score >= 80) return "critical";
          if (score >= 65) return "high";
          if (score >= 45) return "medium";
          return "low";
        };

        // Calculate investment needs (based on deficits and scale)
        const baseInvestment = 1000000; // 1M USD base
        const scaleFactor = Math.max(1, currentProduction / 5000); // Scale by production
        const deficitFactor = (100 - fsci) / 20; // Scale by FSCI deficit
        const investmentNeed = Math.round(
          baseInvestment * scaleFactor * deficitFactor
        );

        // Calculate expected ROI (higher for regions with good potential but poor current performance)
        const productionPotential = Math.min(100, (pci + psi + crs) / 3);
        const performanceGap = Math.max(0, productionPotential - fsci);
        const expectedROI = Math.min(
          300,
          Math.max(50, performanceGap * 3 + 50)
        );

        // Determine risk level
        const getRiskLevel = (
          fsciScore: number,
          climateAvg: number
        ): PriorityRegion["risk_level"] => {
          if (fsciScore < 40 || climateAvg < 45) return "high";
          if (fsciScore < 60 || climateAvg < 60) return "medium";
          return "low";
        };

        // Determine intervention types needed
        const interventionTypes: string[] = [];
        if (pci < 60) interventionTypes.push("Water Management");
        if (psi < 60) interventionTypes.push("Climate Adaptation");
        if (crs < 60) interventionTypes.push("Resilience Building");
        if (fsci < 50) interventionTypes.push("Emergency Support");
        if (currentProduction < 5000)
          interventionTypes.push("Productivity Enhancement");

        // Estimate time to impact (months)
        const timeToImpact =
          priorityScore >= 80
            ? 6
            : priorityScore >= 65
            ? 12
            : priorityScore >= 45
            ? 18
            : 24;

        // Estimate beneficiaries (farmers and families)
        const estimatedFarmers = Math.round(
          currentProduction / 2 + Math.random() * 1000
        );
        const beneficiaries = estimatedFarmers * 4; // Average family size

        return {
          id: item.id || `${level}_${index}`,
          name:
            level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
          fsci_score: fsci,
          pci_score: pci,
          psi_score: psi,
          crs_score: crs,
          production_potential: productionPotential,
          current_production: currentProduction,
          population: beneficiaries,
          priority_score: priorityScore,
          priority_category: getPriorityCategory(priorityScore),
          investment_need: investmentNeed,
          expected_roi: expectedROI,
          risk_level: getRiskLevel(fsci, (pci + psi + crs) / 3),
          intervention_type: interventionTypes,
          time_to_impact: timeToImpact,
          beneficiaries: beneficiaries,
        };
      })
      .filter((region) => region.name && region.fsci_score > 0);

    // Create priority matrix (Impact vs Cost quadrants)
    const createPriorityMatrix = (): PriorityMatrix[] => {
      const medianInvestment =
        priorityRegions.map((r) => r.investment_need).sort((a, b) => a - b)[
          Math.floor(priorityRegions.length / 2)
        ] || 5000000;

      const medianImpact =
        priorityRegions.map((r) => r.priority_score).sort((a, b) => a - b)[
          Math.floor(priorityRegions.length / 2)
        ] || 50;

      return [
        {
          quadrant: "high_impact_low_cost" as const,
          regions: priorityRegions.filter(
            (r) =>
              r.priority_score >= medianImpact &&
              r.investment_need <= medianInvestment
          ),
          total_investment: 0,
          expected_impact: 0,
          priority_level: 1,
        },
        {
          quadrant: "high_impact_high_cost" as const,
          regions: priorityRegions.filter(
            (r) =>
              r.priority_score >= medianImpact &&
              r.investment_need > medianInvestment
          ),
          total_investment: 0,
          expected_impact: 0,
          priority_level: 2,
        },
        {
          quadrant: "low_impact_low_cost" as const,
          regions: priorityRegions.filter(
            (r) =>
              r.priority_score < medianImpact &&
              r.investment_need <= medianInvestment
          ),
          total_investment: 0,
          expected_impact: 0,
          priority_level: 3,
        },
        {
          quadrant: "low_impact_high_cost" as const,
          regions: priorityRegions.filter(
            (r) =>
              r.priority_score < medianImpact &&
              r.investment_need > medianInvestment
          ),
          total_investment: 0,
          expected_impact: 0,
          priority_level: 4,
        },
      ].map((quadrant) => ({
        ...quadrant,
        total_investment: quadrant.regions.reduce(
          (sum, r) => sum + r.investment_need,
          0
        ),
        expected_impact:
          quadrant.regions.length > 0
            ? quadrant.regions.reduce((sum, r) => sum + r.priority_score, 0) /
              quadrant.regions.length
            : 0,
      }));
    };

    const priorityMatrix = createPriorityMatrix();

    // Budget allocation summary with proper typing
    const budgetAllocation: BudgetAllocation = {
      total_investment: priorityRegions.reduce(
        (sum, r) => sum + r.investment_need,
        0
      ),
      critical_regions: priorityRegions.filter(
        (r) => r.priority_category === "critical"
      ),
      high_priority_regions: priorityRegions.filter(
        (r) => r.priority_category === "high"
      ),
      total_beneficiaries: priorityRegions.reduce(
        (sum, r) => sum + r.beneficiaries,
        0
      ),
      average_roi:
        priorityRegions.length > 0
          ? priorityRegions.reduce((sum, r) => sum + r.expected_roi, 0) /
            priorityRegions.length
          : 0,
    };

    return { priorityRegions, priorityMatrix, budgetAllocation };
  }, [analysisData, level]);

  // Filter and sort regions
  const filteredRegions = useMemo(() => {
    let filtered = [...priorityRegions];

    if (selectedQuadrant !== "all") {
      const quadrant = priorityMatrix.find(
        (q) => q.quadrant === selectedQuadrant
      );
      if (quadrant) {
        const quadrantRegionIds = quadrant.regions.map((r) => r.id);
        filtered = filtered.filter((r) => quadrantRegionIds.includes(r.id));
      }
    }

    // Sort regions
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "priority":
          return b.priority_score - a.priority_score;
        case "roi":
          return b.expected_roi - a.expected_roi;
        case "investment":
          return b.investment_need - a.investment_need;
        case "beneficiaries":
          return b.beneficiaries - a.beneficiaries;
        default:
          return b.priority_score - a.priority_score;
      }
    });

    return filtered.slice(0, maxPriorities);
  }, [
    priorityRegions,
    priorityMatrix,
    selectedQuadrant,
    sortBy,
    maxPriorities,
  ]);

  // Get priority badge variant
  const getPriorityBadgeVariant = (
    category: PriorityRegion["priority_category"]
  ) => {
    switch (category) {
      case "critical":
        return "destructive";
      case "high":
        return "default";
      case "medium":
        return "secondary";
      case "low":
        return "outline";
      default:
        return "outline";
    }
  };

  // Get risk badge variant
  const getRiskBadgeVariant = (risk: PriorityRegion["risk_level"]) => {
    switch (risk) {
      case "high":
        return "destructive";
      case "medium":
        return "outline";
      case "low":
        return "secondary";
      default:
        return "outline";
    }
  };

  // Format currency
  const formatCurrency = (amount: number) => {
    if (amount >= 1000000) {
      return `$${(amount / 1000000).toFixed(1)}M`;
    }
    return `$${(amount / 1000).toFixed(0)}K`;
  };

  // Format number
  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  // Loading state
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-1/3 animate-pulse"></div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {[...Array(4)].map((_, index) => (
                <div
                  key={index}
                  className="h-32 bg-gray-200 rounded animate-pulse"
                ></div>
              ))}
            </div>
            <div className="space-y-3">
              {[...Array(5)].map((_, index) => (
                <div
                  key={index}
                  className="h-16 bg-gray-200 rounded animate-pulse"
                ></div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error || !analysisData) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-red-600">
            <Icons.alertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p>Error loading investment priority data</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (priorityRegions.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.target className="h-8 w-8 mx-auto mb-2" />
            <p>No investment priority data available</p>
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
            <Icons.target className="h-5 w-5 mr-2" />
            Investment Priority Analysis
            <Badge variant="outline" className="ml-2">
              {filteredRegions.length} Priority Regions
            </Badge>
          </div>
        </CardTitle>

        {/* Summary Dashboard */}
        {showBudgetAllocation && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-4">
            <div className="p-4 bg-red-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-red-600">Critical Priority</p>
                  <p className="text-2xl font-bold text-red-800">
                    {budgetAllocation.critical_regions?.length || 0}
                  </p>
                  <p className="text-xs text-red-600">
                    regions need immediate action
                  </p>
                </div>
                <Icons.alertTriangle className="h-8 w-8 text-red-600" />
              </div>
            </div>

            <div className="p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-600">Total Investment</p>
                  <p className="text-2xl font-bold text-blue-800">
                    {formatCurrency(budgetAllocation.total_investment)}
                  </p>
                  <p className="text-xs text-blue-600">
                    estimated funding required
                  </p>
                </div>
                <Icons.dollarSign className="h-8 w-8 text-blue-600" />
              </div>
            </div>

            <div className="p-4 bg-green-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-600">Expected ROI</p>
                  <p className="text-2xl font-bold text-green-800">
                    {budgetAllocation.average_roi.toFixed(0)}%
                  </p>
                  <p className="text-xs text-green-600">
                    average return on investment
                  </p>
                </div>
                <Icons.trendingUp className="h-8 w-8 text-green-600" />
              </div>
            </div>

            <div className="p-4 bg-purple-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-purple-600">Beneficiaries</p>
                  <p className="text-2xl font-bold text-purple-800">
                    {formatNumber(budgetAllocation.total_beneficiaries)}
                  </p>
                  <p className="text-xs text-purple-600">
                    people directly impacted
                  </p>
                </div>
                <Icons.users className="h-8 w-8 text-purple-600" />
              </div>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="priority_list" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="priority_list">Priority Rankings</TabsTrigger>
            <TabsTrigger value="matrix_view">Investment Matrix</TabsTrigger>
            <TabsTrigger value="portfolio">Portfolio Analysis</TabsTrigger>
          </TabsList>

          {/* Priority Rankings Tab */}
          <TabsContent value="priority_list" className="space-y-6">
            {/* Controls */}
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">Sort by:</span>
                <Select
                  value={sortBy}
                  onValueChange={(value) => setSortBy(value as typeof sortBy)}
                >
                  <SelectTrigger className="w-[150px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="priority">Priority Score</SelectItem>
                    <SelectItem value="roi">Expected ROI</SelectItem>
                    <SelectItem value="investment">Investment Need</SelectItem>
                    <SelectItem value="beneficiaries">Beneficiaries</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator orientation="vertical" className="h-6" />

              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">Quadrant:</span>
                <Select
                  value={selectedQuadrant}
                  onValueChange={setSelectedQuadrant}
                >
                  <SelectTrigger className="w-[180px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Regions</SelectItem>
                    <SelectItem value="high_impact_low_cost">
                      High Impact, Low Cost
                    </SelectItem>
                    <SelectItem value="high_impact_high_cost">
                      High Impact, High Cost
                    </SelectItem>
                    <SelectItem value="low_impact_low_cost">
                      Low Impact, Low Cost
                    </SelectItem>
                    <SelectItem value="low_impact_high_cost">
                      Low Impact, High Cost
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Priority List */}
            <div className="space-y-4">
              {filteredRegions.map((region, index) => (
                <Card key={region.id} className="border-l-4 border-l-blue-500">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-start space-x-4">
                        <div className="flex-shrink-0">
                          <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-lg font-bold">
                            {index + 1}
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="text-lg font-semibold">
                              {region.name}
                            </h3>
                            <Badge
                              variant={getPriorityBadgeVariant(
                                region.priority_category
                              )}
                            >
                              {region.priority_category} Priority
                            </Badge>
                            <Badge
                              variant={getRiskBadgeVariant(region.risk_level)}
                            >
                              {region.risk_level} Risk
                            </Badge>
                          </div>

                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">FSCI Score:</span>
                              <div className="font-semibold">
                                {region.fsci_score.toFixed(1)}
                              </div>
                            </div>
                            <div>
                              <span className="text-gray-600">
                                Investment Need:
                              </span>
                              <div className="font-semibold">
                                {formatCurrency(region.investment_need)}
                              </div>
                            </div>
                            <div>
                              <span className="text-gray-600">
                                Expected ROI:
                              </span>
                              <div className="font-semibold text-green-600">
                                {region.expected_roi}%
                              </div>
                            </div>
                            <div>
                              <span className="text-gray-600">
                                Beneficiaries:
                              </span>
                              <div className="font-semibold">
                                {formatNumber(region.beneficiaries)}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="text-sm text-gray-600">
                          Priority Score
                        </div>
                        <div className="text-2xl font-bold text-blue-600">
                          {region.priority_score.toFixed(0)}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      {/* Climate Scores */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">
                          Climate Indicators:
                        </h4>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-xs">PCI</span>
                            <span className="text-xs font-medium">
                              {region.pci_score.toFixed(1)}
                            </span>
                          </div>
                          <Progress value={region.pci_score} className="h-1" />
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-xs">PSI</span>
                            <span className="text-xs font-medium">
                              {region.psi_score.toFixed(1)}
                            </span>
                          </div>
                          <Progress value={region.psi_score} className="h-1" />
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between items-center">
                            <span className="text-xs">CRS</span>
                            <span className="text-xs font-medium">
                              {region.crs_score.toFixed(1)}
                            </span>
                          </div>
                          <Progress value={region.crs_score} className="h-1" />
                        </div>
                      </div>

                      {/* Intervention Types */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">
                          Required Interventions:
                        </h4>
                        <div className="flex flex-wrap gap-1">
                          {region.intervention_type.map((intervention, idx) => (
                            <Badge
                              key={idx}
                              variant="outline"
                              className="text-xs"
                            >
                              {intervention}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Timeline & Impact */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium">Implementation:</h4>
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between">
                            <span className="text-gray-600">
                              Time to Impact:
                            </span>
                            <span className="font-medium">
                              {region.time_to_impact} months
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">
                              Production Potential:
                            </span>
                            <span className="font-medium">
                              {region.production_potential.toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Priority Score Visualization */}
                    <div className="pt-2">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Investment Priority</span>
                        <span>{region.priority_score.toFixed(0)}/100</span>
                      </div>
                      <Progress value={region.priority_score} className="h-2" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Investment Matrix Tab */}
          <TabsContent value="matrix_view" className="space-y-6">
            {showMatrix && (
              <div>
                <h3 className="text-lg font-semibold mb-4">
                  Investment Impact vs Cost Matrix
                </h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {priorityMatrix.map((quadrant) => (
                    <Card
                      key={quadrant.quadrant}
                      className={`${
                        quadrant.priority_level === 1
                          ? "border-green-500 bg-green-50"
                          : quadrant.priority_level === 2
                          ? "border-blue-500 bg-blue-50"
                          : quadrant.priority_level === 3
                          ? "border-yellow-500 bg-yellow-50"
                          : "border-red-500 bg-red-50"
                      }`}
                    >
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base">
                          {quadrant.quadrant === "high_impact_low_cost" &&
                            "🎯 High Impact, Low Cost"}
                          {quadrant.quadrant === "high_impact_high_cost" &&
                            "💰 High Impact, High Cost"}
                          {quadrant.quadrant === "low_impact_low_cost" &&
                            "⚡ Low Impact, Low Cost"}
                          {quadrant.quadrant === "low_impact_high_cost" &&
                            "⚠️ Low Impact, High Cost"}

                          <Badge variant="outline" className="ml-2">
                            Priority {quadrant.priority_level}
                          </Badge>
                        </CardTitle>
                      </CardHeader>

                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-600">Regions:</span>
                            <div className="text-xl font-bold">
                              {quadrant.regions.length}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">
                              Total Investment:
                            </span>
                            <div className="text-xl font-bold">
                              {formatCurrency(quadrant.total_investment)}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">
                              Avg Impact Score:
                            </span>
                            <div className="text-xl font-bold">
                              {quadrant.expected_impact.toFixed(0)}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">
                              Avg Investment:
                            </span>
                            <div className="text-xl font-bold">
                              {quadrant.regions.length > 0
                                ? formatCurrency(
                                    quadrant.total_investment /
                                      quadrant.regions.length
                                  )
                                : "$0"}
                            </div>
                          </div>
                        </div>

                        {quadrant.regions.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium mb-2">
                              Top Regions:
                            </h4>
                            <div className="space-y-1">
                              {quadrant.regions
                                .slice(0, 3)
                                .map((region, index) => (
                                  <div
                                    key={region.id}
                                    className="flex justify-between text-sm"
                                  >
                                    <span>{region.name}</span>
                                    <span className="font-medium">
                                      {region.priority_score.toFixed(0)}
                                    </span>
                                  </div>
                                ))}
                              {quadrant.regions.length > 3 && (
                                <div className="text-xs text-gray-500">
                                  +{quadrant.regions.length - 3} more regions
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        <Progress
                          value={quadrant.expected_impact}
                          className="h-2"
                        />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>

          {/* Portfolio Analysis Tab */}
          <TabsContent value="portfolio" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Risk-Return Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    Risk-Return Profile
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {["low", "medium", "high"].map((riskLevel) => {
                      const riskRegions = priorityRegions.filter(
                        (r) => r.risk_level === riskLevel
                      );
                      const avgROI =
                        riskRegions.length > 0
                          ? riskRegions.reduce(
                              (sum, r) => sum + r.expected_roi,
                              0
                            ) / riskRegions.length
                          : 0;
                      const totalInvestment = riskRegions.reduce(
                        (sum, r) => sum + r.investment_need,
                        0
                      );

                      return (
                        <div key={riskLevel} className="p-3 border rounded-lg">
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-medium capitalize">
                              {riskLevel} Risk
                            </span>
                            <Badge
                              variant={getRiskBadgeVariant(
                                riskLevel as PriorityRegion["risk_level"]
                              )}
                            >
                              {riskRegions.length} regions
                            </Badge>
                          </div>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">Avg ROI:</span>
                              <div className="font-bold text-green-600">
                                {avgROI.toFixed(0)}%
                              </div>
                            </div>
                            <div>
                              <span className="text-gray-600">Investment:</span>
                              <div className="font-bold">
                                {formatCurrency(totalInvestment)}
                              </div>
                            </div>
                          </div>
                          <Progress value={avgROI / 3} className="h-2 mt-2" />
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Investment Timeline */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    Implementation Timeline
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {[
                      {
                        range: "0-6 months",
                        regions: priorityRegions.filter(
                          (r) => r.time_to_impact <= 6
                        ),
                      },
                      {
                        range: "6-12 months",
                        regions: priorityRegions.filter(
                          (r) => r.time_to_impact > 6 && r.time_to_impact <= 12
                        ),
                      },
                      {
                        range: "12-18 months",
                        regions: priorityRegions.filter(
                          (r) => r.time_to_impact > 12 && r.time_to_impact <= 18
                        ),
                      },
                      {
                        range: "18+ months",
                        regions: priorityRegions.filter(
                          (r) => r.time_to_impact > 18
                        ),
                      },
                    ].map((timeframe) => (
                      <div
                        key={timeframe.range}
                        className="p-3 border rounded-lg"
                      >
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium">{timeframe.range}</span>
                          <Badge variant="outline">
                            {timeframe.regions.length} regions
                          </Badge>
                        </div>
                        <div className="text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">
                              Total Investment:
                            </span>
                            <span className="font-medium">
                              {formatCurrency(
                                timeframe.regions.reduce(
                                  (sum, r) => sum + r.investment_need,
                                  0
                                )
                              )}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">
                              Total Beneficiaries:
                            </span>
                            <span className="font-medium">
                              {formatNumber(
                                timeframe.regions.reduce(
                                  (sum, r) => sum + r.beneficiaries,
                                  0
                                )
                              )}
                            </span>
                          </div>
                        </div>
                        <Progress
                          value={
                            (timeframe.regions.length /
                              priorityRegions.length) *
                            100
                          }
                          className="h-2 mt-2"
                        />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Strategic Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Strategic Portfolio Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-800 mb-2">
                      🎯 Phase 1: Quick Wins (0-12 months)
                    </h4>
                    <p className="text-sm text-blue-700 mb-3">
                      Focus on{" "}
                      {priorityMatrix.find(
                        (q) => q.quadrant === "high_impact_low_cost"
                      )?.regions.length || 0}{" "}
                      high-impact, low-cost regions to demonstrate rapid results
                      and build momentum.
                    </p>
                    <div className="text-sm">
                      <strong>Investment: </strong>
                      {formatCurrency(
                        priorityMatrix.find(
                          (q) => q.quadrant === "high_impact_low_cost"
                        )?.total_investment || 0
                      )}
                    </div>
                  </div>

                  <div className="p-4 bg-green-50 rounded-lg">
                    <h4 className="font-medium text-green-800 mb-2">
                      💰 Phase 2: Major Impact (12-24 months)
                    </h4>
                    <p className="text-sm text-green-700 mb-3">
                      Implement high-impact, high-cost interventions in critical
                      regions with secured funding and proven implementation
                      capacity.
                    </p>
                    <div className="text-sm">
                      <strong>Investment: </strong>
                      {formatCurrency(
                        priorityMatrix.find(
                          (q) => q.quadrant === "high_impact_high_cost"
                        )?.total_investment || 0
                      )}
                    </div>
                  </div>

                  <div className="p-4 bg-yellow-50 rounded-lg">
                    <h4 className="font-medium text-yellow-800 mb-2">
                      ⚡ Phase 3: Optimization (24+ months)
                    </h4>
                    <p className="text-sm text-yellow-700 mb-3">
                      Address remaining regions with cost-effective
                      interventions and consolidate gains from previous phases.
                    </p>
                    <div className="text-sm">
                      <strong>Total Remaining Investment: </strong>
                      {formatCurrency(
                        (priorityMatrix.find(
                          (q) => q.quadrant === "low_impact_low_cost"
                        )?.total_investment || 0) +
                          (priorityMatrix.find(
                            (q) => q.quadrant === "low_impact_high_cost"
                          )?.total_investment || 0)
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
