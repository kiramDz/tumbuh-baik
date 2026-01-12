"use client";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Icons } from "@/app/dashboard/_components/icons";
import { getTwoLevelAnalysis } from "@/lib/fetch/spatial.map.fetch";
import type { FSIAnalysisParams } from "@/lib/fetch/spatial.map.fetch"; // ✅ Updated import

export interface RankingTableProps {
  className?: string;
  analysisParams?: FSIAnalysisParams; // ✅ Updated from TwoLevelAnalysisParams
  level?: "kabupaten" | "kecamatan";
  maxItems?: number;
  showFilters?: boolean;
  showExport?: boolean;
}

interface RankingItem {
  id: string;
  name: string;
  fsi_score: number;
  padi_production_tons: number;
  climate_correlation: number;
  rank: number;
  trend: "up" | "down" | "stable";
  fsi_category:
    | "Sangat Tinggi"
    | "Tinggi"
    | "Sedang"
    | "Rendah"
    | "Sangat Rendah"; // ✅ Added "Sangat Rendah"
  production_growth?: number;
}

type SortField = "fsi_score" | "padi_production_tons" | "climate_correlation"; // ✅ Updated fields
type SortDirection = "asc" | "desc";

export function RankingTable({
  className,
  analysisParams,
  level = "kabupaten",
  maxItems = 50,
  showFilters = true,
  showExport = true,
}: RankingTableProps) {
  // State management
  const [sortField, setSortField] = useState<SortField>("fsi_score"); // ✅ Updated default sort
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [searchTerm, setSearchTerm] = useState("");
  const [fsiFilter, setFsiFilter] = useState<string>("all"); // ✅ Added FSI category filter

  // ✅ Updated default parameters to FSI format
  const defaultParams: FSIAnalysisParams = {
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    districts: "all",
    analysis_level: "both",
    include_bps_data: true,
  };

  const params = analysisParams || defaultParams;

  const {
    data: analysisData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["fsi-two-level-ranking", params], // ✅ Updated query key
    queryFn: () => getTwoLevelAnalysis(params),
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });

  // ✅ Process and transform data for FSI ranking table
  const rankingData = useMemo((): RankingItem[] => {
    if (!analysisData) return [];

    const sourceData =
      level === "kabupaten"
        ? analysisData.level_2_kabupaten_analysis?.data || []
        : analysisData.level_1_kecamatan_analysis?.data || [];

    if (sourceData.length === 0) return [];

    // Helper function to safely get values
    const getValue = (item: any, ...keys: string[]): number => {
      for (const key of keys) {
        const value = item[key];
        if (typeof value === "number" && !isNaN(value)) return value;
      }
      return 0;
    };

    // ✅ Helper function to extract latest padi production from BPS structure
    const getLatestPadiProduction = (item: any): number => {
      // Check for direct production fields first
      const directProduction = getValue(
        item,
        "latest_production_tons",
        "production_tons"
      );
      if (directProduction > 0) return directProduction;

      // ✅ Extract from BPS padi_ton structure: {"2018": 375153.85, "2019": 396467.64, ...}
      if (item.padi_ton && typeof item.padi_ton === "object") {
        const years = Object.keys(item.padi_ton)
          .map((y) => parseInt(y))
          .filter((y) => !isNaN(y))
          .sort((a, b) => b - a); // Get latest year first

        if (years.length > 0) {
          const latestYear = years[0];
          const production = item.padi_ton[latestYear.toString()];
          return typeof production === "number" ? production : 0;
        }
      }

      return 0;
    };

    // ✅ Helper function to calculate production growth from BPS data
    const calculateProductionGrowth = (item: any): number => {
      if (!item.padi_ton || typeof item.padi_ton !== "object") return 0;

      const years = Object.keys(item.padi_ton)
        .map((y) => parseInt(y))
        .filter((y) => !isNaN(y))
        .sort((a, b) => a - b); // Sort ascending for growth calculation

      if (years.length < 2) return 0;

      const firstYear = years[0];
      const lastYear = years[years.length - 1];
      const firstValue = item.padi_ton[firstYear.toString()];
      const lastValue = item.padi_ton[lastYear.toString()];

      if (firstValue > 0 && lastValue > 0) {
        return ((lastValue - firstValue) / firstValue) * 100;
      }

      return 0;
    };

    // ✅ Helper function to get FSI category
    const getFsiCategory = (score: number): RankingItem["fsi_category"] => {
      if (score >= 72.6) return "Sangat Tinggi"; // Lhoksukon tier (Top producer)
      if (score >= 67.0) {
        // Further classification based on production ranking
        if (score >= 69.4) return "Rendah"; // Bireuen tier (Lower production despite good climate)
        if (score >= 67.6) return "Sedang"; // Aceh Besar tier (Medium production)
        return "Tinggi"; // Pidie tier (High production with moderate climate)
      }
      return "Sangat Rendah"; // Aceh Jaya tier (Lowest production)
    };

    const items: RankingItem[] = sourceData
      .map((item: any) => {
        // ✅ Extract FSI score (updated from FSCI)
        const fsi =
          level === "kabupaten"
            ? getValue(item, "aggregated_fsi_score", "fsi_score", "fsi_mean")
            : getValue(item, "fsi_score", "fsi_mean", "aggregated_fsi_score");

        // ✅ Extract padi production using BPS structure
        const padiProduction = getLatestPadiProduction(item);

        // Extract correlation
        const correlation = getValue(
          item,
          "climate_production_correlation",
          "correlation",
          "climate_correlation"
        );

        // Calculate production growth
        const productionGrowth = calculateProductionGrowth(item);

        // Simulate trend based on FSI score and production growth
        const getTrend = (): RankingItem["trend"] => {
          if (productionGrowth > 5) return "up";
          if (productionGrowth < -5) return "down";
          if (fsi > 75) return "up";
          if (fsi < 50) return "down";
          return "stable";
        };

        return {
          id: item.id || `${level}_${Math.random()}`,
          name:
            level === "kabupaten"
              ? item.kabupaten_name || item.name
              : item.kecamatan_name || item.name,
          fsi_score: fsi, // ✅ Updated from fsci_score
          padi_production_tons: padiProduction, // ✅ Updated from production_tons
          climate_correlation: correlation,
          rank: 0, // Will be set after sorting
          trend: getTrend(),
          fsi_category: getFsiCategory(fsi), // ✅ Added FSI classification
          production_growth: productionGrowth, // ✅ Added growth calculation
        };
      })
      .filter((item) => item.name && item.fsi_score > 0);

    // Sort items by FSI score and assign ranks
    const sortedItems = [...items].sort((a, b) => b.fsi_score - a.fsi_score);
    sortedItems.forEach((item, index) => {
      item.rank = index + 1;
    });

    return sortedItems;
  }, [analysisData, level]);

  // Apply filtering and sorting
  const filteredAndSortedData = useMemo(() => {
    let filtered = [...rankingData];

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter((item) =>
        item.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // ✅ Apply FSI category filter
    if (fsiFilter !== "all") {
      filtered = filtered.filter((item) => item.fsi_category === fsiFilter);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      const aValue = a[sortField];
      const bValue = b[sortField];
      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
      }
      return 0;
    });

    // Re-assign ranks after sorting and filtering
    filtered.forEach((item, index) => {
      item.rank = index + 1;
    });

    // Limit results
    return filtered.slice(0, maxItems);
  }, [rankingData, searchTerm, fsiFilter, sortField, sortDirection, maxItems]);

  // Handle sorting
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  // ✅ Export functionality - Updated for FSI and padi production
  const handleExport = () => {
    const csvContent = [
      // Header
      [
        "Rank",
        "Name",
        "FSI Score", // ✅ Updated from FSCI
        "FSI Category", // ✅ Added category
        "Padi Production (tons)", // ✅ Updated from Wheat Production
        "Production Growth (%)", // ✅ Added growth metric
        "Climate Correlation",
        "Trend",
      ].join(","),
      // Data rows
      ...filteredAndSortedData.map((item) =>
        [
          item.rank,
          `"${item.name}"`, // Quote names to handle commas
          item.fsi_score.toFixed(1),
          item.fsi_category,
          item.padi_production_tons.toFixed(0),
          item.production_growth?.toFixed(1) || "0",
          item.climate_correlation.toFixed(3),
          item.trend,
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `fsi_${level}_ranking_${
      new Date().toISOString().split("T")[0]
    }.csv`; // ✅ Updated filename
    link.click();
  };

  // Loading State
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="h-6 bg-gray-200 rounded w-1/3 animate-pulse"></div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[...Array(10)].map((_, index) => (
              <div key={index} className="flex space-x-4 animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-8"></div>
                <div className="h-4 bg-gray-200 rounded w-32"></div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
                <div className="h-4 bg-gray-200 rounded w-20"></div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
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
            <p>Error loading FSI ranking data</p>{" "}
            {/* ✅ Updated error message */}
          </div>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (rankingData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.list className="h-8 w-8 mx-auto mb-2" />
            <p>No FSI ranking data available</p> {/* ✅ Updated message */}
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
            <Icons.trophy className="h-5 w-5 mr-2" />
            {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} FSI Rankings{" "}
            {/* ✅ Updated title */}
            <Badge variant="outline" className="ml-2">
              {filteredAndSortedData.length} of {rankingData.length}
            </Badge>
          </div>
          {showExport && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleExport}
              className="flex items-center space-x-2"
            >
              <Icons.download className="h-4 w-4" />
              <span>Export CSV</span>
            </Button>
          )}
        </CardTitle>

        {showFilters && (
          <div className="flex flex-wrap gap-4 mt-4">
            {/* Search */}
            <div className="flex-1 min-w-[200px]">
              <Input
                placeholder={`Search ${level}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>

            {/* ✅ FSI Category Filter */}
            <div className="min-w-[150px]">
              <select
                value={fsiFilter}
                onChange={(e) => setFsiFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="all">All Categories</option>
                <option value="Sangat Tinggi">Sangat Tinggi (≥72.6)</option>
                <option value="Tinggi">Tinggi (67.0 - Pidie)</option>
                <option value="Sedang">Sedang (67.6 - Aceh Besar)</option>
                <option value="Rendah">Rendah (69.4 - Bireuen)</option>
                <option value="Sangat Rendah">
                  Sangat Rendah (&lt;65.5)
                </option>{" "}
                {/* ✅ Fixed HTML entity */}
              </select>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[60px]">Rank</TableHead>
                <TableHead className="min-w-[200px]">
                  {level === "kabupaten" ? "Kabupaten" : "Kecamatan"}
                </TableHead>
                <TableHead
                  className="text-center cursor-pointer"
                  onClick={() => handleSort("fsi_score")} // ✅ Updated sort field
                >
                  <div className="flex items-center justify-center space-x-1">
                    <span>FSI Score</span> {/* ✅ Updated from FSCI */}
                    {sortField === "fsi_score" && (
                      <Icons.arrowUpDown className="h-4 w-4" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="text-center cursor-pointer"
                  onClick={() => handleSort("padi_production_tons")} // ✅ Updated sort field
                >
                  <div className="flex items-center justify-center space-x-1">
                    <span>Padi Production</span>{" "}
                    {/* ✅ Updated from Wheat Production */}
                    {sortField === "padi_production_tons" && (
                      <Icons.arrowUpDown className="h-4 w-4" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="text-center cursor-pointer"
                  onClick={() => handleSort("climate_correlation")}
                >
                  <div className="flex items-center justify-center space-x-1">
                    <span>Climate Correlation</span>
                    {sortField === "climate_correlation" && (
                      <Icons.arrowUpDown className="h-4 w-4" />
                    )}
                  </div>
                </TableHead>
                <TableHead className="text-center">Category</TableHead>{" "}
                {/* ✅ Added FSI category column */}
                <TableHead className="text-center">Trend</TableHead>
              </TableRow>
            </TableHeader>

            <TableBody>
              {filteredAndSortedData.map((item) => (
                <TableRow key={item.id} className="hover:bg-gray-50">
                  {/* Rank */}
                  <TableCell className="text-center font-medium">
                    <div className="flex items-center justify-center">
                      {item.rank <= 3 && (
                        <Icons.medal
                          className={`h-4 w-4 mr-1 ${
                            item.rank === 1
                              ? "text-yellow-500"
                              : item.rank === 2
                              ? "text-gray-400"
                              : "text-amber-600"
                          }`}
                        />
                      )}
                      {item.rank}
                    </div>
                  </TableCell>

                  {/* Name */}
                  <TableCell className="font-medium">
                    <div className="flex items-center space-x-2">
                      <span>{item.name}</span>
                    </div>
                  </TableCell>

                  {/* ✅ FSI Score (updated from FSCI) */}
                  <TableCell className="text-center">
                    <div className="space-y-1">
                      <div className="font-semibold">
                        {item.fsi_score.toFixed(1)}
                      </div>
                      <Progress
                        value={item.fsi_score}
                        className="h-1 w-16 mx-auto"
                      />
                    </div>
                  </TableCell>

                  {/* ✅ Padi Production (updated from Wheat) */}
                  <TableCell className="text-center">
                    <div className="text-sm">
                      <div className="font-medium">
                        {(item.padi_production_tons / 1000).toFixed(1)}K
                      </div>
                      <div className="text-gray-500">tons padi</div>{" "}
                      {/* ✅ Updated label */}
                      {item.production_growth !== undefined && (
                        <div
                          className={`text-xs ${
                            item.production_growth > 0
                              ? "text-green-600"
                              : item.production_growth < 0
                              ? "text-red-600"
                              : "text-gray-500"
                          }`}
                        >
                          {item.production_growth > 0 ? "+" : ""}
                          {item.production_growth.toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </TableCell>

                  {/* Climate Correlation */}
                  <TableCell className="text-center">
                    <div className="space-y-1">
                      <div className="font-medium">
                        {item.climate_correlation.toFixed(3)}
                      </div>
                      <Progress
                        value={item.climate_correlation * 100}
                        className="h-1 w-12 mx-auto"
                      />
                    </div>
                  </TableCell>

                  {/* ✅ FSI Category */}
                  <TableCell className="text-center">
                    <Badge
                      variant={
                        item.fsi_category === "Sangat Tinggi"
                          ? "default"
                          : item.fsi_category === "Tinggi"
                          ? "secondary"
                          : item.fsi_category === "Sedang"
                          ? "outline"
                          : "destructive"
                      }
                      className={`text-xs ${
                        item.fsi_category === "Sangat Tinggi"
                          ? "bg-green-100 text-green-800"
                          : item.fsi_category === "Tinggi"
                          ? "bg-blue-100 text-blue-800"
                          : item.fsi_category === "Sedang"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-red-100 text-red-800"
                      }`}
                    >
                      {item.fsi_category}
                    </Badge>
                  </TableCell>

                  {/* Trend */}
                  <TableCell className="text-center">
                    <div className="flex items-center justify-center">
                      {item.trend === "up" && (
                        <Icons.trendingUp className="h-4 w-4 text-green-600" />
                      )}
                      {item.trend === "down" && (
                        <Icons.trendingDown className="h-4 w-4 text-red-600" />
                      )}
                      {item.trend === "stable" && (
                        <Icons.minus className="h-4 w-4 text-gray-500" />
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {filteredAndSortedData.length === 0 && rankingData.length > 0 && (
          <div className="text-center py-8 text-gray-500">
            <Icons.search className="h-8 w-8 mx-auto mb-2" />
            <p>No results found for current filters</p>
            <div className="flex justify-center space-x-2 mt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSearchTerm("")}
              >
                Clear Search
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setFsiFilter("all")}
              >
                Clear Filters
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
