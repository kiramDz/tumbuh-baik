"use client";

import { useMemo, useState, useCallback } from "react";
import { useTwoLevelAnalysis } from "@/hooks/use-twoLevelAnalysis";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Icons } from "@/app/dashboard/_components/icons";
import type { KecamatanAnalysis } from "@/lib/fetch/spatial.map.fetch";

export interface KecamatanListProps {
  className?: string;
  kabupatenFilter?: string | null;
  onKecamatanSelect?: (kecamatan: string, kabupaten: string) => void;
  onViewOnMap?: (kecamatan: string) => void;
  showKabupatenColumn?: boolean;
  maxItems?: number;
}

type SortField =
  | "fsci_score"
  | "pci_score"
  | "psi_score"
  | "crs_score"
  | "kecamatan_name"
  | "kabupaten_name";

type SortDirection = "asc" | "desc";

export function KecamatanList({
  className,
  kabupatenFilter,
  onKecamatanSelect,
  onViewOnMap,
  showKabupatenColumn = true,
  maxItems,
}: KecamatanListProps) {
  const {
    analysisData,
    selectedKecamatan,
    selectedKabupaten,
    selectKecamatan,
    loading,
    error,
  } = useTwoLevelAnalysis();

  // Local state
  const [searchTerm, setSearchTerm] = useState("");
  const [fsciFilter, setFsciFilter] = useState<string>("all");
  const [sortField, setSortField] = useState<SortField>("fsci_score");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  // Get kecamatan data
  const kecamatanData = useMemo(() => {
    return analysisData?.level_1_kecamatan_analysis?.data || [];
  }, [analysisData]);

  // Filter and sort data
  const filteredAndSortedData = useMemo(() => {
    let filtered = kecamatanData;
    // Apply kabupaten filter
    if (kabupatenFilter) {
      filtered = filtered.filter(
        (kecamatan) => kecamatan.kabupaten_name === kabupatenFilter
      );
    }
    // Apply search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      filtered = filtered.filter(
        (kecamatan) =>
          kecamatan.kecamatan_name.toLowerCase().includes(searchLower) ||
          kecamatan.kabupaten_name.toLowerCase().includes(searchLower)
      );
    }

    // Apply FSCI classification filter
    if (fsciFilter !== "all") {
      filtered = filtered.filter((kecamatan) => {
        const score = kecamatan.fsci_score || 0;
        switch (fsciFilter) {
          case "primer":
            return score >= 80;
          case "sekunder":
            return score >= 60 && score < 80;
          case "tersier":
            return score >= 40 && score < 60;
          case "below":
            return score < 40;
          default:
            return true;
        }
      });
    }

    // Sort data
    const sorted = [...filtered].sort((a, b) => {
      let aValue: any;
      let bValue: any;

      switch (sortField) {
        case "fsci_score":
          aValue = a.fsci_score || 0;
          bValue = b.fsci_score || 0;
          break;
        case "pci_score":
          aValue = a.pci_score || 0;
          bValue = b.pci_score || 0;
          break;
        case "psi_score":
          aValue = a.psi_score || 0;
          bValue = b.psi_score || 0;
          break;
        case "crs_score":
          aValue = a.crs_score || 0;
          bValue = b.crs_score || 0;
          break;
        case "kecamatan_name":
          aValue = a.kecamatan_name;
          bValue = b.kecamatan_name;
          break;
        case "kabupaten_name":
          aValue = a.kabupaten_name;
          bValue = b.kabupaten_name;
          break;
        default:
          return 0;
      }

      if (typeof aValue === "string" && typeof bValue === "string") {
        return sortDirection === "asc"
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      } else {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
      }
    });

    // Apply max items limit
    return maxItems ? sorted.slice(0, maxItems) : sorted;
  }, [
    kecamatanData,
    kabupatenFilter,
    searchTerm,
    fsciFilter,
    sortField,
    sortDirection,
    maxItems,
  ]);

  // Summary statistics
  const summaryStats = useMemo(() => {
    const data = filteredAndSortedData;
    if (data.length === 0) return null;

    const fsciScores = data.map((k) => k.fsci_score || 0);
    const pciScores = data.map((k) => k.pci_score || 0);
    const psiScores = data.map((k) => k.psi_score || 0);
    const crsScores = data.map((k) => k.crs_score || 0);

    return {
      total: data.length,
      avgFSCI: fsciScores.reduce((a, b) => a + b, 0) / fsciScores.length,
      avgPCI: pciScores.reduce((a, b) => a + b, 0) / pciScores.length,
      avgPSI: psiScores.reduce((a, b) => a + b, 0) / psiScores.length,
      avgCRS: crsScores.reduce((a, b) => a + b, 0) / crsScores.length,
      lumbungPrimer: data.filter((k) => (k.fsci_score || 0) >= 80).length,
      lumbungSekunder: data.filter(
        (k) => (k.fsci_score || 0) >= 60 && (k.fsci_score || 0) < 80
      ).length,
      lumbungTersier: data.filter(
        (k) => (k.fsci_score || 0) >= 40 && (k.fsci_score || 0) < 60
      ).length,
      belowThreshold: data.filter((k) => (k.fsci_score || 0) < 40).length,
    };
  }, [filteredAndSortedData]);

  // Helper function
  const getFSCIClassification = (score: number) => {
    if (score >= 80)
      return {
        label: "Lumbung Pangan Primer",
        color: "bg-green-500 text-white",
        textColor: "text-green-700",
      };
    if (score >= 60)
      return {
        label: "Lumbung Pangan Sekunder",
        color: "bg-yellow-500 text-white",
        textColor: "text-yellow-700",
      };
    if (score >= 40)
      return {
        label: "Lumbung Pangan Tersier",
        color: "bg-red-500 text-white",
        textColor: "text-red-700",
      };
    return {
      label: "Below Threshold",
      color: "bg-gray-500 text-white",
      textColor: "text-gray-700",
    };
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const handleKecamatanClick = useCallback(
    (kecamatan: KecamatanAnalysis) => {
      selectKecamatan(kecamatan.kecamatan_name);
      onKecamatanSelect?.(kecamatan.kecamatan_name, kecamatan.kabupaten_name);
    },
    [selectKecamatan, onKecamatanSelect]
  );

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span>Loading kecamatan data...</span>
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
            <p>Error loading kecamatan data</p>
            <p className="text-sm text-gray-600 mt-1">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }
  if (kecamatanData.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="text-center text-gray-600">
            <Icons.users className="h-8 w-8 mx-auto mb-2" />
            <p className="font-medium">No Kecamatan Data Available</p>
            <p className="text-sm mt-1">
              Run a two-level analysis to view kecamatan details
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className={className}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center justify-between text-lg">
          <div className="flex items-center">
            <Icons.users className="h-5 w-5 mr-2" />
            Kecamatan Analysis
            {kabupatenFilter && (
              <Badge variant="outline" className="ml-2">
                {kabupatenFilter}
              </Badge>
            )}
          </div>
          <div className="text-sm text-gray-600">
            {summaryStats?.total || 0} kecamatan
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Summary Statistics */}
        {summaryStats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {summaryStats.lumbungPrimer}
              </div>
              <div className="text-xs text-gray-600">Lumbung Primer</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {summaryStats.lumbungSekunder}
              </div>
              <div className="text-xs text-gray-600">Lumbung Sekunder</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {summaryStats.lumbungTersier}
              </div>
              <div className="text-xs text-gray-600">Lumbung Tersier</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">
                {summaryStats.belowThreshold}
              </div>
              <div className="text-xs text-gray-600">Below Threshold</div>
            </div>
          </div>
        )}

        {/* Filters and Controls */}
        <div className="flex flex-wrap gap-4">
          {/* Search */}
          <div className="flex-1 min-w-[200px]">
            <div className="relative">
              <Icons.search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <Input
                placeholder="Search kecamatan or kabupaten..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>

          {/* FSCI Classification Filter */}
          <Select value={fsciFilter} onValueChange={setFsciFilter}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter by FSCI" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Classifications</SelectItem>
              <SelectItem value="primer">Lumbung Primer (≥80)</SelectItem>
              <SelectItem value="sekunder">Lumbung Sekunder (60-79)</SelectItem>
              <SelectItem value="tersier">Lumbung Tersier (40-59)</SelectItem>
              <SelectItem value="below">Below Threshold (&lt;40)</SelectItem>
            </SelectContent>
          </Select>

          {/* Sort Field */}
          <Select
            value={sortField}
            onValueChange={(value: SortField) => setSortField(value)}
          >
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Sort By" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="fsci_score">FSCI Score</SelectItem>
              <SelectItem value="pci_score">PCI Score</SelectItem>
              <SelectItem value="psi_score">PSI Score</SelectItem>
              <SelectItem value="crs_score">CRS Score</SelectItem>
              <SelectItem value="kecamatan_name">Name</SelectItem>
              {showKabupatenColumn && (
                <SelectItem value="kabupaten_name">Kabupaten</SelectItem>
              )}
            </SelectContent>
          </Select>

          {/* Sort Direction */}
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setSortDirection(sortDirection === "asc" ? "desc" : "asc")
            }
          >
            <Icons.arrowUpDown className="h-4 w-4 mr-1" />
            {sortDirection === "asc" ? "↑" : "↓"}
          </Button>
        </div>

        {/* Results Count */}
        <div className="text-sm text-gray-600 flex items-center justify-between">
          <span>
            Showing {filteredAndSortedData.length} of {kecamatanData.length}{" "}
            kecamatan
            {maxItems && filteredAndSortedData.length >= maxItems && (
              <span className="text-amber-600 ml-1">
                (limited to {maxItems})
              </span>
            )}
          </span>
          {summaryStats && (
            <span>Avg FSCI: {summaryStats.avgFSCI.toFixed(1)}</span>
          )}
        </div>

        {/* Data Table */}
        <div className="border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("kecamatan_name")}
                  role="button"
                  tabIndex={0}
                  aria-sort={
                    sortField === "kecamatan_name"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <div className="flex items-center">
                    <Icons.mapPin className="h-3 w-3 mr-1" />
                    Kecamatan
                    {sortField === "kecamatan_name" && (
                      <Icons.arrowUpDown className="h-3 w-3 ml-1" />
                    )}
                  </div>
                </TableHead>
                {showKabupatenColumn && (
                  <TableHead
                    className="cursor-pointer hover:bg-gray-50"
                    onClick={() => handleSort("kabupaten_name")}
                    role="button"
                    tabIndex={0}
                    aria-sort={
                      sortField === "kabupaten_name"
                        ? sortDirection === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <div className="flex items-center">
                      Kabupaten
                      {sortField === "kabupaten_name" && (
                        <Icons.arrowUpDown className="h-3 w-3 ml-1" />
                      )}
                    </div>
                  </TableHead>
                )}
                <TableHead
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("fsci_score")}
                  role="button"
                  tabIndex={0}
                  aria-sort={
                    sortField === "fsci_score"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <div className="flex items-center">
                    <Icons.wheat className="h-4 w-4 mr-1" />
                    FSCI
                    {sortField === "fsci_score" && (
                      <Icons.arrowUpDown className="h-3 w-3 ml-1" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("pci_score")}
                  role="button"
                  tabIndex={0}
                  aria-sort={
                    sortField === "pci_score"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <div className="flex items-center">
                    <Icons.cloudRain className="h-4 w-4 mr-1 text-blue-600" />
                    PCI
                    {sortField === "pci_score" && (
                      <Icons.arrowUpDown className="h-3 w-3 ml-1" />
                    )}
                  </div>
                </TableHead>
                <TableHead
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("psi_score")}
                  role="button"
                  tabIndex={0}
                  aria-sort={
                    sortField === "psi_score"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <div className="flex items-center">
                    <Icons.thermometerSun className="h-4 w-4 mr-1 text-orange-600" />
                    PSI
                    {sortField === "psi_score" && (
                      <Icons.arrowUpDown className="h-3 w-3 ml-1" />
                    )}
                  </div>
                </TableHead>

                <TableHead
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => handleSort("crs_score")}
                  role="button"
                  tabIndex={0}
                  aria-sort={
                    sortField === "crs_score"
                      ? sortDirection === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <div className="flex items-center">
                    <Icons.droplets className="h-4 w-4 mr-1 text-teal-600" />
                    CRS
                    {sortField === "crs_score" && (
                      <Icons.arrowUpDown className="h-3 w-3 ml-1" />
                    )}
                  </div>
                </TableHead>
                <TableHead>Classification</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedData.map((kecamatan) => {
                const fsciClass = getFSCIClassification(
                  kecamatan.fsci_score || 0
                );
                const isSelected =
                  selectedKecamatan === kecamatan.kecamatan_name;

                return (
                  <TableRow
                    key={`${kecamatan.kabupaten_name}-${kecamatan.kecamatan_name}`}
                    className={`cursor-pointer hover:bg-gray-50 ${
                      isSelected
                        ? "bg-blue-50 border-l-4 border-l-blue-500"
                        : ""
                    }`}
                    onClick={() => handleKecamatanClick(kecamatan)}
                  >
                    <TableCell className="font-medium">
                      <div>
                        <div
                          className={
                            isSelected ? "text-blue-700 font-semibold" : ""
                          }
                        >
                          {kecamatan.kecamatan_name}
                        </div>
                      </div>
                    </TableCell>

                    {showKabupatenColumn && (
                      <TableCell className="text-sm text-gray-600">
                        {kecamatan.kabupaten_name}
                      </TableCell>
                    )}

                    <TableCell>
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">
                          {(kecamatan.fsci_score || 0).toFixed(1)}
                        </span>
                        <div className="flex-1">
                          <Progress
                            value={kecamatan.fsci_score || 0}
                            className="h-2 w-16"
                          />
                        </div>
                      </div>
                    </TableCell>

                    <TableCell>
                      <span className="text-sm">
                        {(kecamatan.pci_score || 0).toFixed(1)}
                      </span>
                    </TableCell>

                    <TableCell>
                      <span className="text-sm">
                        {(kecamatan.psi_score || 0).toFixed(1)}
                      </span>
                    </TableCell>

                    <TableCell>
                      <span className="text-sm">
                        {(kecamatan.crs_score || 0).toFixed(1)}
                      </span>
                    </TableCell>

                    <TableCell>
                      <Badge className={`${fsciClass.color} text-xs`}>
                        {fsciClass.label}
                      </Badge>
                    </TableCell>

                    <TableCell>
                      <div className="flex items-center space-x-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleKecamatanClick(kecamatan);
                          }}
                        >
                          <Icons.preview className="h-3 w-3" />
                        </Button>

                        {onViewOnMap && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              onViewOnMap(kecamatan.kecamatan_name);
                            }}
                          >
                            <Icons.mapPin className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
        {filteredAndSortedData.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <Icons.filter className="h-8 w-8 mx-auto mb-2" />
            <p> No Kecamatan match the current filters</p>
            <p className="text-sm"> Try adjusting your filter criteria</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
