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
import type { TwoLevelAnalysisParams } from "@/lib/fetch/spatial.map.fetch";

export interface RankingTableProps {
  className?: string;
  analysisParams?: TwoLevelAnalysisParams;
  level?: "kabupaten" | "kecamatan";
  maxItems?: number;
  showFilters?: boolean;
  showExport?: boolean;
}

interface RankingItem {
  id: string;
  name: string;
  fsci_score: number;
  production_tons: number;
  climate_correlation: number;
  rank: number;
  trend: "up" | "down" | "stable";
}

type SortField = "fsci_score" | "production_tons" | "climate_correlation";
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
  const [sortField, setSortField] = useState<SortField>("fsci_score");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [searchTerm, setSearchTerm] = useState("");

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

  // Process and transform data for ranking table
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

    const items: RankingItem[] = sourceData
      .map((item: any) => {
        const fsci =
          level === "kabupaten"
            ? getValue(item, "aggregated_fsci_score", "fsci_score", "fsci_mean")
            : getValue(
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

        // Simulate trend (in real app, calculate from historical data)
        const getTrend = (): RankingItem["trend"] => {
          const random = Math.random();
          if (random > 0.6) return "up";
          if (random < 0.4) return "down";
          return "stable";
        };

        return {
          id: item.id || `${level}_${Math.random()}`,
          name:
            level === "kabupaten" ? item.kabupaten_name : item.kecamatan_name,
          fsci_score: fsci,
          production_tons: production,
          climate_correlation: correlation,
          rank: 0, // Will be set after sorting
          trend: getTrend(),
        };
      })
      .filter((item) => item.name && item.fsci_score > 0);

    // Sort items and assign ranks
    const sortedItems = [...items].sort((a, b) => b.fsci_score - a.fsci_score);
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
  }, [rankingData, searchTerm, sortField, sortDirection, maxItems]);

  // Handle sorting
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  // Export functionality - Updated to only include available columns
  const handleExport = () => {
    const csvContent = [
      // Header
      [
        "Rank",
        "Name",
        "FSCI Score",
        "Production (tons)",
        "Climate Correlation",
      ].join(","),
      // Data rows
      ...filteredAndSortedData.map((item) =>
        [
          item.rank,
          item.name,
          item.fsci_score.toFixed(1),
          item.production_tons.toFixed(0),
          item.climate_correlation.toFixed(3),
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${level}_ranking_${
      new Date().toISOString().split("T")[0]
    }.csv`;
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
            <p>Error loading ranking data</p>
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
            <p>No ranking data available</p>
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
            {level === "kabupaten" ? "Kabupaten" : "Kecamatan"} Rankings
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
            {/* Search - Only filter now */}
            <div className="flex-1 min-w-[200px]">
              <Input
                placeholder={`Search ${level}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
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
                  onClick={() => handleSort("fsci_score")}
                >
                  <div className="flex items-center justify-center space-x-1">
                    <span>FSCI Score</span>
                    {sortField === "fsci_score" && (
                      <Icons.arrowUpDown className="h-4 w-4" />
                    )}
                  </div>
                </TableHead>

                <TableHead
                  className="text-center cursor-pointer"
                  onClick={() => handleSort("production_tons")}
                >
                  <div className="flex items-center justify-center space-x-1">
                    <span>Wheat Production</span>
                    {sortField === "production_tons" && (
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

                  {/* FSCI Score */}
                  <TableCell className="text-center">
                    <div className="space-y-1">
                      <div className="font-semibold">
                        {item.fsci_score.toFixed(1)}
                      </div>
                      <Progress
                        value={item.fsci_score}
                        className="h-1 w-16 mx-auto"
                      />
                    </div>
                  </TableCell>

                  {/* Wheat Production */}
                  <TableCell className="text-center">
                    <div className="text-sm">
                      <div className="font-medium">
                        {(item.production_tons / 1000).toFixed(1)}K
                      </div>
                      <div className="text-gray-500">tons</div>
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
            <p>No results found for current search</p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSearchTerm("")}
              className="mt-2"
            >
              Clear Search
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
