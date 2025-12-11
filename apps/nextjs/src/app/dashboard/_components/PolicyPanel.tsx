"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface PolicyPanelProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  maxRecommendations?: number;
  showBudgetEstimate?: boolean;
  showTimeframes?: boolean;
}

interface PolicyRecommendation {
  id: string;
  title: string;
  description: string;
  category:
    | "infrastructure"
    | "climate_adaptation"
    | "capacity_building"
    | "technology"
    | "market_access";
  priority: "critical" | "high" | "medium" | "low";
  regions: string[];
  impact_areas: string[];
  estimated_cost: number;
  timeframe: "immediate" | "short_term" | "medium_term" | "long_term";
  expected_impact: number; // 0-100%
  success_indicators: string[];
  implementation_steps: string[];
}

interface RegionAnalysis {
  name: string;
  fsci_score: number;
  main_challenges: string[];
  opportunities: string[];
  recommended_policies: string[];
  investment_priority: number; // 1-10 scale
}

export function PolicyPanel({
  className,
  analysisParams,
  level = "kabupaten",
  maxRecommendations = 10,
  showBudgetEstimate = true,
  showTimeframes = true,
}: PolicyPanelProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [selectedPriority, setSelectedPriority] = useState<string>("all");

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

  // Process data and generate policy recommendations
  const { regionAnalyses, policyRecommendations } = useMemo(() => {
    if (!analysisData) return { regionAnalyses: [], policyRecommendations: [] };

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    if (sourceData.length === 0)
      return { regionAnalyses: [], policyRecommendations: [] };

    // Update property access based on actual backend response
    const getValue = (item: any, ...keys: string[]): number => {
      for (const key of keys) {
        const value = item[key];
        if (typeof value === "number" && !isNaN(value) && value >= 0)
          return value;
      }
      return 0;
    };

    // Analyze regions and identify challenges/opportunities
    const regionAnalyses: RegionAnalysis[] = sourceData
      .map((item: any) => {
        const fsci =
          level === "kabupaten"
            ? getValue(item, "aggregated_fsci_score", "fsci_score")
            : getValue(item, "fsci_score", "aggregated_fsci_score");

        const pci = getValue(item, "pci_score", "pci_mean");
        const psi = getValue(item, "psi_score", "psi_mean");
        const crs = getValue(item, "crs_score", "crs_mean");
        const production =
          level === "kabupaten"
            ? getValue(
                item,
                "latest_production_tons",
                "average_production_tons"
              )
            : getValue(
                item,
                "production_tons",
                "latest_production_tons",
                "total_production"
              );

        // Identify main challenges based on lowest scores
        const scores = [
          { name: "Climate Precipitation", value: pci },
          { name: "Temperature Suitability", value: psi },
          { name: "Crop Resilience", value: crs },
          { name: "Overall Food Security", value: fsci },
        ];

        const mainChallenges = scores
          .filter((s) => s.value < 60)
          .sort((a, b) => a.value - b.value)
          .slice(0, 3)
          .map((s) => s.name);

        // Identify opportunities (high scores that can be leveraged)
        const opportunities = scores
          .filter((s) => s.value >= 70)
          .sort((a, b) => b.value - a.value)
          .slice(0, 2)
          .map((s) => s.name);

        // Generate region-specific recommendations
        const recommendedPolicies: string[] = [];

        if (pci < 50)
          recommendedPolicies.push("Water Management Infrastructure");
        if (psi < 50)
          recommendedPolicies.push("Climate-Resilient Crop Varieties");
        if (crs < 50)
          recommendedPolicies.push("Agricultural Extension Services");
        if (production < 1000)
          recommendedPolicies.push("Productivity Enhancement Programs");
        if (fsci < 60)
          recommendedPolicies.push("Integrated Food Security Initiative");

        return {
          name:
            level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
          fsci_score: fsci,
          main_challenges: mainChallenges,
          opportunities: opportunities,
          recommended_policies: recommendedPolicies,
          investment_priority:
            Math.round((100 - fsci) / 10) + Math.min(3, mainChallenges.length),
        };
      })
      .filter((analysis) => analysis.name && analysis.fsci_score > 0);

    // Generate comprehensive policy recommendations
    const policyRecommendations: PolicyRecommendation[] = [
      {
        id: "water_infrastructure",
        title: "Integrated Water Management Infrastructure",
        description:
          "Develop comprehensive irrigation systems, water storage facilities, and flood management infrastructure to address precipitation variability and improve agricultural resilience.",
        category: "infrastructure",
        priority: "critical",
        regions: regionAnalyses
          .filter((r) => r.main_challenges.includes("Climate Precipitation"))
          .map((r) => r.name)
          .slice(0, 5),
        impact_areas: ["Water Security", "Crop Yield", "Climate Adaptation"],
        estimated_cost: 50000000, // 50M USD
        timeframe: "medium_term",
        expected_impact: 85,
        success_indicators: [
          "Increase in irrigated agricultural area by 40%",
          "Reduction in crop loss due to drought by 60%",
          "Improvement in PCI scores by 25 points average",
        ],
        implementation_steps: [
          "Conduct detailed water resource assessment",
          "Design integrated water management systems",
          "Secure funding and regulatory approvals",
          "Phase 1: Priority regions infrastructure development",
          "Phase 2: System integration and optimization",
          "Monitoring and adaptive management",
        ],
      },
      {
        id: "climate_resilient_crops",
        title: "Climate-Resilient Crop Development Program",
        description:
          "Research, develop, and distribute climate-adapted crop varieties that can withstand temperature extremes and changing precipitation patterns.",
        category: "climate_adaptation",
        priority: "high",
        regions: regionAnalyses
          .filter((r) => r.main_challenges.includes("Temperature Suitability"))
          .map((r) => r.name)
          .slice(0, 6),
        impact_areas: [
          "Temperature Adaptation",
          "Yield Stability",
          "Food Security",
        ],
        estimated_cost: 25000000, // 25M USD
        timeframe: "long_term",
        expected_impact: 75,
        success_indicators: [
          "Development of 10+ climate-resilient varieties",
          "Adoption rate of 70% among target farmers",
          "PSI score improvement by 20 points average",
        ],
        implementation_steps: [
          "Establish research partnerships with agricultural institutions",
          "Conduct climate impact assessments",
          "Develop breeding and selection programs",
          "Field testing and validation",
          "Farmer training and distribution programs",
          "Monitoring and continuous improvement",
        ],
      },
      {
        id: "extension_services",
        title: "Enhanced Agricultural Extension Services",
        description:
          "Strengthen agricultural extension services with modern technology, training programs, and farmer support networks to improve agricultural practices and resilience.",
        category: "capacity_building",
        priority: "high",
        regions: regionAnalyses
          .filter((r) => r.main_challenges.includes("Crop Resilience"))
          .map((r) => r.name)
          .slice(0, 8),
        impact_areas: [
          "Knowledge Transfer",
          "Best Practices",
          "Farmer Capacity",
        ],
        estimated_cost: 15000000, // 15M USD
        timeframe: "short_term",
        expected_impact: 70,
        success_indicators: [
          "Train 5000+ farmers in climate-smart agriculture",
          "Establish 50+ farmer field schools",
          "CRS score improvement by 15 points average",
        ],
        implementation_steps: [
          "Assess current extension service capacity",
          "Develop training curricula and materials",
          "Train extension agents and master farmers",
          "Establish farmer field schools",
          "Deploy digital extension platforms",
          "Regular monitoring and feedback collection",
        ],
      },
      {
        id: "precision_agriculture",
        title: "Precision Agriculture Technology Adoption",
        description:
          "Implement precision agriculture technologies including GPS-guided machinery, soil sensors, and data analytics to optimize resource use and increase productivity.",
        category: "technology",
        priority: "medium",
        regions: regionAnalyses
          .filter((r) => r.opportunities.length > 0)
          .map((r) => r.name)
          .slice(0, 4),
        impact_areas: ["Resource Efficiency", "Productivity", "Sustainability"],
        estimated_cost: 30000000, // 30M USD
        timeframe: "medium_term",
        expected_impact: 80,
        success_indicators: [
          "Adopt precision agriculture in 1000+ farms",
          "Reduce input costs by 20% while maintaining yields",
          "Increase overall productivity by 25%",
        ],
        implementation_steps: [
          "Technology assessment and selection",
          "Pilot project implementation",
          "Farmer training and certification programs",
          "Equipment financing and subsidy programs",
          "Data platform development",
          "Scale-up and continuous support",
        ],
      },
      {
        id: "market_linkages",
        title: "Agricultural Market Linkage Development",
        description:
          "Strengthen market connections for farmers through cooperative development, value chain improvement, and digital marketing platforms to ensure fair pricing and market access.",
        category: "market_access",
        priority: "medium",
        regions: regionAnalyses.slice(0, 10).map((r) => r.name),
        impact_areas: ["Market Access", "Price Stability", "Income Generation"],
        estimated_cost: 20000000, // 20M USD
        timeframe: "short_term",
        expected_impact: 65,
        success_indicators: [
          "Establish 20+ farmer cooperatives",
          "Increase farmer income by 30% average",
          "Reduce post-harvest losses by 40%",
        ],
        implementation_steps: [
          "Market analysis and value chain mapping",
          "Cooperative formation and strengthening",
          "Digital platform development",
          "Storage and processing facility development",
          "Buyer-seller linkage programs",
          "Monitoring and impact assessment",
        ],
      },
      {
        id: "early_warning_system",
        title: "Climate Early Warning System",
        description:
          "Develop and deploy comprehensive climate monitoring and early warning systems to help farmers make informed decisions and prepare for climate risks.",
        category: "climate_adaptation",
        priority: "high",
        regions: regionAnalyses.map((r) => r.name),
        impact_areas: [
          "Risk Management",
          "Decision Support",
          "Climate Preparedness",
        ],
        estimated_cost: 10000000, // 10M USD
        timeframe: "immediate",
        expected_impact: 60,
        success_indicators: [
          "Deploy 100+ weather monitoring stations",
          "Reach 80% of farmers with early warnings",
          "Reduce climate-related crop losses by 50%",
        ],
        implementation_steps: [
          "Install weather monitoring infrastructure",
          "Develop forecasting and alert systems",
          "Create farmer communication networks",
          "Train farmers on system usage",
          "Integrate with existing agricultural services",
          "Regular system updates and maintenance",
        ],
      },
    ];

    return { regionAnalyses, policyRecommendations };
  }, [analysisData, level]);

  // Filter recommendations based on selected filters
  const filteredRecommendations = useMemo(() => {
    let filtered = [...policyRecommendations];

    if (selectedCategory !== "all") {
      filtered = filtered.filter((rec) => rec.category === selectedCategory);
    }

    if (selectedPriority !== "all") {
      filtered = filtered.filter((rec) => rec.priority === selectedPriority);
    }

    return filtered.slice(0, maxRecommendations);
  }, [
    policyRecommendations,
    selectedCategory,
    selectedPriority,
    maxRecommendations,
  ]);

  // Calculate total budget and impact estimates
  const budgetSummary = useMemo(() => {
    const totalCost = filteredRecommendations.reduce(
      (sum, rec) => sum + rec.estimated_cost,
      0
    );
    const averageImpact =
      filteredRecommendations.length > 0
        ? filteredRecommendations.reduce(
            (sum, rec) => sum + rec.expected_impact,
            0
          ) / filteredRecommendations.length
        : 0;

    const timeframeDistribution = {
      immediate: filteredRecommendations.filter(
        (r) => r.timeframe === "immediate"
      ).length,
      short_term: filteredRecommendations.filter(
        (r) => r.timeframe === "short_term"
      ).length,
      medium_term: filteredRecommendations.filter(
        (r) => r.timeframe === "medium_term"
      ).length,
      long_term: filteredRecommendations.filter(
        (r) => r.timeframe === "long_term"
      ).length,
    };

    return { totalCost, averageImpact, timeframeDistribution };
  }, [filteredRecommendations]);

  // Get priority badge variant
  const getPriorityBadgeVariant = (
    priority: PolicyRecommendation["priority"]
  ) => {
    switch (priority) {
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

  // Get category icon
  const getCategoryIcon = (category: PolicyRecommendation["category"]) => {
    switch (category) {
      case "infrastructure":
        return <Icons.building className="h-4 w-4" />;
      case "climate_adaptation":
        return <Icons.cloudRain className="h-4 w-4" />;
      case "capacity_building":
        return <Icons.users className="h-4 w-4" />;
      case "technology":
        return <Icons.cpu className="h-4 w-4" />;
      case "market_access":
        return <Icons.trendingUp className="h-4 w-4" />;
      default:
        return <Icons.clipBoard className="h-4 w-4" />;
    }
  };

  // Get timeframe badge color
  const getTimeframeBadgeVariant = (
    timeframe: PolicyRecommendation["timeframe"]
  ) => {
    switch (timeframe) {
      case "immediate":
        return "destructive";
      case "short_term":
        return "default";
      case "medium_term":
        return "secondary";
      case "long_term":
        return "outline";
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

  // Loading state
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-1/3 animate-pulse"></div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {[...Array(3)].map((_, index) => (
              <div key={index} className="p-4 border rounded-lg animate-pulse">
                <div className="h-5 bg-gray-200 rounded w-2/3 mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
              </div>
            ))}
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
            <p>Error loading policy recommendations</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (regionAnalyses.length === 0 || policyRecommendations.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.clipBoard className="h-8 w-8 mx-auto mb-2" />
            <p>No policy recommendations available</p>
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
            <Icons.clipBoard className="h-5 w-5 mr-2" />
            Policy Recommendations & Investment Analysis
            <Badge variant="outline" className="ml-2">
              {filteredRecommendations.length} Recommendations
            </Badge>
          </div>
        </CardTitle>

        {/* Summary Stats */}
        {showBudgetEstimate && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <div className="p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-600">Total Investment</p>
                  <p className="text-xl font-bold text-blue-800">
                    {formatCurrency(budgetSummary.totalCost)}
                  </p>
                </div>
                <Icons.dollarSign className="h-8 w-8 text-blue-600" />
              </div>
            </div>

            <div className="p-3 bg-green-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-green-600">Expected Impact</p>
                  <p className="text-xl font-bold text-green-800">
                    {budgetSummary.averageImpact.toFixed(0)}%
                  </p>
                </div>
                <Icons.target className="h-8 w-8 text-green-600" />
              </div>
            </div>

            <div className="p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-purple-600">Target Regions</p>
                  <p className="text-xl font-bold text-purple-800">
                    {regionAnalyses.length}
                  </p>
                </div>
                <Icons.mapPin className="h-8 w-8 text-purple-600" />
              </div>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="recommendations" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="recommendations">
              Policy Recommendations
            </TabsTrigger>
            <TabsTrigger value="regional">Regional Analysis</TabsTrigger>
            <TabsTrigger value="implementation">
              Implementation Plan
            </TabsTrigger>
          </TabsList>

          {/* Policy Recommendations Tab */}
          <TabsContent value="recommendations" className="space-y-6">
            {/* Filters */}
            <div className="flex flex-wrap gap-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">Category:</span>
                <div className="flex space-x-1">
                  <Button
                    variant={selectedCategory === "all" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedCategory("all")}
                  >
                    All
                  </Button>
                  <Button
                    variant={
                      selectedCategory === "infrastructure"
                        ? "default"
                        : "outline"
                    }
                    size="sm"
                    onClick={() => setSelectedCategory("infrastructure")}
                  >
                    Infrastructure
                  </Button>
                  <Button
                    variant={
                      selectedCategory === "climate_adaptation"
                        ? "default"
                        : "outline"
                    }
                    size="sm"
                    onClick={() => setSelectedCategory("climate_adaptation")}
                  >
                    Climate
                  </Button>
                  <Button
                    variant={
                      selectedCategory === "technology" ? "default" : "outline"
                    }
                    size="sm"
                    onClick={() => setSelectedCategory("technology")}
                  >
                    Technology
                  </Button>
                </div>
              </div>

              <Separator orientation="vertical" className="h-6" />

              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">Priority:</span>
                <div className="flex space-x-1">
                  <Button
                    variant={selectedPriority === "all" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedPriority("all")}
                  >
                    All
                  </Button>
                  <Button
                    variant={
                      selectedPriority === "critical"
                        ? "destructive"
                        : "outline"
                    }
                    size="sm"
                    onClick={() => setSelectedPriority("critical")}
                  >
                    Critical
                  </Button>
                  <Button
                    variant={
                      selectedPriority === "high" ? "default" : "outline"
                    }
                    size="sm"
                    onClick={() => setSelectedPriority("high")}
                  >
                    High
                  </Button>
                </div>
              </div>
            </div>

            {/* Recommendations List */}
            <div className="space-y-6">
              {filteredRecommendations.map((recommendation) => (
                <Card
                  key={recommendation.id}
                  className="border-l-4 border-l-blue-500"
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        {getCategoryIcon(recommendation.category)}
                        <div>
                          <CardTitle className="text-lg">
                            {recommendation.title}
                          </CardTitle>
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge
                              variant={getPriorityBadgeVariant(
                                recommendation.priority
                              )}
                            >
                              {recommendation.priority} Priority
                            </Badge>
                            {showTimeframes && (
                              <Badge
                                variant={getTimeframeBadgeVariant(
                                  recommendation.timeframe
                                )}
                              >
                                {recommendation.timeframe.replace("_", " ")}
                              </Badge>
                            )}
                            {showBudgetEstimate && (
                              <Badge variant="outline">
                                {formatCurrency(recommendation.estimated_cost)}
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-gray-600">
                          Expected Impact
                        </div>
                        <div className="text-xl font-bold text-green-600">
                          {recommendation.expected_impact}%
                        </div>
                      </div>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <p className="text-gray-700">
                      {recommendation.description}
                    </p>

                    {/* Impact Areas */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">
                        Impact Areas:
                      </h4>
                      <div className="flex flex-wrap gap-1">
                        {recommendation.impact_areas.map((area) => (
                          <Badge
                            key={area}
                            variant="secondary"
                            className="text-xs"
                          >
                            {area}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Target Regions */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">
                        Target Regions ({recommendation.regions.length}):
                      </h4>
                      <div className="flex flex-wrap gap-1">
                        {recommendation.regions.slice(0, 5).map((region) => (
                          <Badge
                            key={region}
                            variant="outline"
                            className="text-xs"
                          >
                            {region}
                          </Badge>
                        ))}
                        {recommendation.regions.length > 5 && (
                          <Badge variant="outline" className="text-xs">
                            +{recommendation.regions.length - 5} more
                          </Badge>
                        )}
                      </div>
                    </div>

                    {/* Success Indicators */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">
                        Success Indicators:
                      </h4>
                      <ul className="text-sm space-y-1">
                        {recommendation.success_indicators.map(
                          (indicator, index) => (
                            <li
                              key={index}
                              className="flex items-center space-x-2"
                            >
                              <Icons.checkCircle className="h-3 w-3 text-green-600 flex-shrink-0" />
                              <span>{indicator}</span>
                            </li>
                          )
                        )}
                      </ul>
                    </div>

                    {/* Progress Visualization */}
                    <div className="pt-2">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Implementation Readiness</span>
                        <span>{recommendation.expected_impact}%</span>
                      </div>
                      <Progress
                        value={recommendation.expected_impact}
                        className="h-2"
                      />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {filteredRecommendations.length === 0 &&
              policyRecommendations.length > 0 && (
                <div className="text-center py-8 text-gray-500">
                  <Icons.filter className="h-8 w-8 mx-auto mb-2" />
                  <p>No recommendations match the selected filters</p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setSelectedCategory("all");
                      setSelectedPriority("all");
                    }}
                    className="mt-2"
                  >
                    Clear Filters
                  </Button>
                </div>
              )}
          </TabsContent>

          {/* Regional Analysis Tab */}
          <TabsContent value="regional" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {regionAnalyses.slice(0, 10).map((region) => (
                <Card key={region.name}>
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center justify-between">
                      <span className="text-base">{region.name}</span>
                      <div className="flex items-center space-x-2">
                        <Badge
                          variant={
                            region.fsci_score >= 70
                              ? "default"
                              : region.fsci_score >= 50
                              ? "secondary"
                              : "destructive"
                          }
                        >
                          FSCI: {region.fsci_score.toFixed(1)}
                        </Badge>
                        <Badge variant="outline">
                          Priority: {region.investment_priority}/10
                        </Badge>
                      </div>
                    </CardTitle>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Main Challenges */}
                    {region.main_challenges.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium mb-2 flex items-center">
                          <Icons.alertCircle className="h-4 w-4 mr-1 text-red-600" />
                          Main Challenges:
                        </h4>
                        <ul className="text-sm space-y-1">
                          {region.main_challenges.map((challenge, index) => (
                            <li
                              key={index}
                              className="flex items-center space-x-2"
                            >
                              <div className="w-2 h-2 bg-red-400 rounded-full flex-shrink-0"></div>
                              <span>{challenge}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Opportunities */}
                    {region.opportunities.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium mb-2 flex items-center">
                          <Icons.lightBulb className="h-4 w-4 mr-1 text-green-600" />
                          Opportunities:
                        </h4>
                        <ul className="text-sm space-y-1">
                          {region.opportunities.map((opportunity, index) => (
                            <li
                              key={index}
                              className="flex items-center space-x-2"
                            >
                              <div className="w-2 h-2 bg-green-400 rounded-full flex-shrink-0"></div>
                              <span>{opportunity}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Recommended Policies */}
                    <div>
                      <h4 className="text-sm font-medium mb-2 flex items-center">
                        <Icons.clipBoard className="h-4 w-4 mr-1 text-blue-600" />
                        Recommended Policies:
                      </h4>
                      <div className="flex flex-wrap gap-1">
                        {region.recommended_policies.map((policy, index) => (
                          <Badge
                            key={index}
                            variant="outline"
                            className="text-xs"
                          >
                            {policy}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Investment Priority Visualization */}
                    <div>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span>Investment Priority</span>
                        <span>{region.investment_priority}/10</span>
                      </div>
                      <Progress
                        value={region.investment_priority * 10}
                        className="h-2"
                      />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Implementation Plan Tab */}
          <TabsContent value="implementation" className="space-y-6">
            {showTimeframes && (
              <div>
                <h3 className="text-lg font-semibold mb-4">
                  Implementation Timeline
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  {Object.entries(budgetSummary.timeframeDistribution).map(
                    ([timeframe, count]) => (
                      <div key={timeframe} className="p-4 border rounded-lg">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {count}
                          </div>
                          <div className="text-sm capitalize text-gray-600">
                            {timeframe.replace("_", " ")}
                          </div>
                        </div>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {/* Critical Path Recommendations */}
            <div>
              <h3 className="text-lg font-semibold mb-4">
                Priority Implementation Sequence
              </h3>
              <div className="space-y-4">
                {policyRecommendations
                  .filter(
                    (rec) =>
                      rec.priority === "critical" || rec.priority === "high"
                  )
                  .slice(0, 4)
                  .map((recommendation, index) => (
                    <Card key={recommendation.id}>
                      <CardContent className="p-4">
                        <div className="flex items-start space-x-4">
                          <div className="flex-shrink-0">
                            <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                              {index + 1}
                            </div>
                          </div>
                          <div className="flex-1">
                            <h4 className="font-medium">
                              {recommendation.title}
                            </h4>
                            <p className="text-sm text-gray-600 mb-3">
                              {recommendation.description}
                            </p>

                            <div className="space-y-2">
                              <h5 className="text-sm font-medium">
                                Key Implementation Steps:
                              </h5>
                              <ul className="text-sm space-y-1">
                                {recommendation.implementation_steps
                                  .slice(0, 3)
                                  .map((step, stepIndex) => (
                                    <li
                                      key={stepIndex}
                                      className="flex items-center space-x-2"
                                    >
                                      <Icons.next className="h-3 w-3 text-gray-400 flex-shrink-0" />
                                      <span>{step}</span>
                                    </li>
                                  ))}
                              </ul>
                            </div>

                            <div className="flex items-center justify-between mt-3">
                              <div className="flex items-center space-x-2">
                                <Badge
                                  variant={getPriorityBadgeVariant(
                                    recommendation.priority
                                  )}
                                >
                                  {recommendation.priority}
                                </Badge>
                                <Badge variant="outline">
                                  {formatCurrency(
                                    recommendation.estimated_cost
                                  )}
                                </Badge>
                              </div>
                              <div className="text-sm text-gray-600">
                                {recommendation.timeframe.replace("_", " ")}
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
              </div>
            </div>

            {/* Implementation Alerts */}
            <div className="space-y-4">
              <Alert>
                <Icons.info className="h-4 w-4" />
                <AlertTitle>Implementation Readiness</AlertTitle>
                <AlertDescription>
                  Based on current FSCI analysis,{" "}
                  {
                    regionAnalyses.filter((r) => r.investment_priority >= 7)
                      .length
                  }{" "}
                  regions require immediate attention with critical priority
                  interventions. Consider phased implementation starting with
                  highest priority regions to maximize impact.
                </AlertDescription>
              </Alert>

              {budgetSummary.totalCost > 100000000 && (
                <Alert>
                  <Icons.dollarSign className="h-4 w-4" />
                  <AlertTitle>Budget Consideration</AlertTitle>
                  <AlertDescription>
                    Total estimated investment of{" "}
                    {formatCurrency(budgetSummary.totalCost)} requires strategic
                    funding approach. Consider partnership opportunities with
                    international development agencies, climate funds, and
                    private sector investment.
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
