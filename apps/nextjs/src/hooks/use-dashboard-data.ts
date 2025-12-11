"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getTwoLevelAnalysis,
  type TwoLevelAnalysisParams,
  type TwoLevelAnalysisResponse,
  type KabupatenAnalysis,
  type KecamatanAnalysis,
} from "@/lib/fetch/spatial.map.fetch";

// Hook configuration interface
export interface DashboardDataConfig {
  level?: "kabupaten" | "kecamatan";
  autoRefresh?: boolean;
  refreshInterval?: number;
  staleTime?: number;
  cacheTime?: number;
}

// Processed dashboard metrics interface
export interface DashboardMetrics {
  totalRegions: number;
  averageFSCI: number;
  averagePCI: number;
  averagePSI: number;
  averageCRS: number;
  totalProduction: number;
  criticalRegions: number;
  highPerformanceRegions: number;
  investmentNeeded: number;
  beneficiariesCount: number;
}

// Dashboard data interface
export interface DashboardData {
  rawData: TwoLevelAnalysisResponse | null;
  kabupatenData: KabupatenAnalysis[];
  kecamatanData: KecamatanAnalysis[];
  metrics: DashboardMetrics;
  chartData: {
    timeSeriesData: Array<{
      year: number;
      fsci: number;
      pci: number;
      psi: number;
      crs: number;
    }>;
    correlationData: Array<{
      x: number;
      y: number;
      name: string;
      category: string;
    }>;
    efficiencyData: Array<{
      region: string;
      efficiency: number;
      potential: number;
      gap: number;
    }>;
    fsciBreakdownData: Array<{
      region: string;
      pci: number;
      psi: number;
      crs: number;
      fsci: number;
    }>;
  };
  rankings: {
    topPerformers: Array<{ name: string; score: number; rank: number }>;
    criticalRegions: Array<{ name: string; score: number; priority: number }>;
    improvedRegions: Array<{
      name: string;
      improvement: number;
      current: number;
    }>;
  };
  filters: {
    availableYears: number[];
    availableSeasons: string[];
    availableDistricts: string[];
  };
}

// Main unified dashboard hook
export function useDashboardData(
  params?: Partial<TwoLevelAnalysisParams>,
  config?: DashboardDataConfig
) {
  // Default configuration
  const defaultConfig: DashboardDataConfig = {
    level: "kabupaten",
    autoRefresh: false,
    refreshInterval: 5 * 60 * 1000, // 5 minutes
    staleTime: 2 * 60 * 1000, // 2 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
  };

  const mergedConfig = { ...defaultConfig, ...config };

  // Default query parameters
  const defaultParams: TwoLevelAnalysisParams = {
    year_start: 2018,
    year_end: 2024,
    bps_start_year: 2018,
    bps_end_year: 2024,
    season: "all",
    aggregation: "mean",
    districts: "all",
  };

  const queryParams = { ...defaultParams, ...params };

  // State for dynamic filters
  const [activeLevel, setActiveLevel] = useState<"kabupaten" | "kecamatan">(
    mergedConfig.level || "kabupaten"
  );

  // Main data query
  const {
    data: rawData,
    isLoading,
    error,
    refetch,
    isRefetching,
    isFetching,
  } = useQuery({
    queryKey: ["dashboard-data", queryParams, activeLevel],
    queryFn: () => getTwoLevelAnalysis(queryParams),
    refetchOnWindowFocus: false,
    staleTime: mergedConfig.staleTime,
    gcTime: mergedConfig.cacheTime,
    refetchInterval: mergedConfig.autoRefresh
      ? mergedConfig.refreshInterval
      : false,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });

  // Process and transform raw data
  const processedData: DashboardData = useMemo(() => {
    if (!rawData) {
      return {
        rawData: null,
        kabupatenData: [],
        kecamatanData: [],
        metrics: {
          totalRegions: 0,
          averageFSCI: 0,
          averagePCI: 0,
          averagePSI: 0,
          averageCRS: 0,
          totalProduction: 0,
          criticalRegions: 0,
          highPerformanceRegions: 0,
          investmentNeeded: 0,
          beneficiariesCount: 0,
        },
        chartData: {
          timeSeriesData: [],
          correlationData: [],
          efficiencyData: [],
          fsciBreakdownData: [],
        },
        rankings: {
          topPerformers: [],
          criticalRegions: [],
          improvedRegions: [],
        },
        filters: {
          availableYears: [],
          availableSeasons: [],
          availableDistricts: [],
        },
      };
    }

    // Extract data arrays
    const kabupatenData = rawData.level_2_kabupaten_analysis?.data || [];
    const kecamatanData = rawData.level_1_kecamatan_analysis?.data || [];

    // Determine active dataset based on level
    const activeData =
      activeLevel === "kabupaten" ? kabupatenData : kecamatanData;

    // Helper function to safely get numeric values
    const getValue = (item: any, ...keys: string[]): number => {
      for (const key of keys) {
        const value = item[key];
        if (typeof value === "number" && !isNaN(value) && value >= 0)
          return value;
      }
      return 0;
    };

    // Calculate metrics
    const metrics: DashboardMetrics = {
      totalRegions: activeData.length,
      averageFSCI:
        activeData.length > 0
          ? activeData.reduce((sum, item) => {
              const fsci =
                activeLevel === "kabupaten"
                  ? getValue(
                      item,
                      "aggregated_fsci_score",
                      "fsci_score",
                      "fsci_mean"
                    )
                  : getValue(
                      item,
                      "fsci_score",
                      "fsci_mean",
                      "aggregated_fsci_score"
                    );
              return sum + fsci;
            }, 0) / activeData.length
          : 0,
      averagePCI:
        activeData.length > 0
          ? activeData.reduce((sum, item) => {
              const pci =
                activeLevel === "kabupaten"
                  ? getValue(
                      item,
                      "aggregated_pci_score",
                      "pci_score",
                      "pci_mean"
                    )
                  : getValue(item, "pci_score", "pci_mean");
              return sum + pci;
            }, 0) / activeData.length
          : 0,
      averagePSI:
        activeData.length > 0
          ? activeData.reduce((sum, item) => {
              const psi =
                activeLevel === "kabupaten"
                  ? getValue(
                      item,
                      "aggregated_psi_score",
                      "psi_score",
                      "psi_mean"
                    )
                  : getValue(item, "psi_score", "psi_mean");
              return sum + psi;
            }, 0) / activeData.length
          : 0,
      averageCRS:
        activeData.length > 0
          ? activeData.reduce((sum, item) => {
              const crs =
                activeLevel === "kabupaten"
                  ? getValue(
                      item,
                      "aggregated_crs_score",
                      "crs_score",
                      "crs_mean"
                    )
                  : getValue(item, "crs_score", "crs_mean");
              return sum + crs;
            }, 0) / activeData.length
          : 0,
      totalProduction: activeData.reduce((sum, item) => {
        const production =
          activeLevel === "kabupaten"
            ? getValue(
                item,
                "latest_production_tons",
                "average_production_tons",
                "total_production"
              )
            : 0; // Kecamatan level typically doesn't have production data
        return sum + production;
      }, 0),
      criticalRegions: activeData.filter((item) => {
        const fsci =
          activeLevel === "kabupaten"
            ? getValue(item, "aggregated_fsci_score", "fsci_score", "fsci_mean")
            : getValue(item, "fsci_score", "fsci_mean");
        return fsci < 50;
      }).length,
      highPerformanceRegions: activeData.filter((item) => {
        const fsci =
          activeLevel === "kabupaten"
            ? getValue(item, "aggregated_fsci_score", "fsci_score", "fsci_mean")
            : getValue(item, "fsci_score", "fsci_mean");
        return fsci >= 80;
      }).length,
      investmentNeeded: activeData.reduce((sum, item) => {
        const fsci =
          activeLevel === "kabupaten"
            ? getValue(item, "aggregated_fsci_score", "fsci_score", "fsci_mean")
            : getValue(item, "fsci_score", "fsci_mean");
        const production =
          activeLevel === "kabupaten"
            ? getValue(
                item,
                "latest_production_tons",
                "average_production_tons"
              )
            : 1000; // Default for kecamatan

        // Calculate investment need (basic formula)
        const urgency = Math.max(0, 100 - fsci);
        const scale = Math.max(1, production / 5000);
        return sum + urgency * scale * 50000; // $50K base investment per urgency point
      }, 0),
      beneficiariesCount: activeData.reduce((sum, item) => {
        const production =
          activeLevel === "kabupaten"
            ? getValue(
                item,
                "latest_production_tons",
                "average_production_tons"
              )
            : 500; // Default for kecamatan
        // Estimate beneficiaries based on production (farmers + families)
        const farmers = Math.round(production / 2);
        return sum + farmers * 4; // Average family size
      }, 0),
    };

    // Generate chart data
    const chartData = {
      // Time series data (simulated trend based on current values)
      timeSeriesData: Array.from({ length: 7 }, (_, i) => {
        const year = 2018 + i;
        const variation = Math.sin(i * 0.5) * 5 + (Math.random() * 3 - 1.5);
        return {
          year,
          fsci: Math.max(0, Math.min(100, metrics.averageFSCI + variation)),
          pci: Math.max(0, Math.min(100, metrics.averagePCI + variation * 1.2)),
          psi: Math.max(0, Math.min(100, metrics.averagePSI + variation * 0.8)),
          crs: Math.max(0, Math.min(100, metrics.averageCRS + variation * 1.1)),
        };
      }),

      // Correlation data (Climate vs Production)
      correlationData: activeData
        .slice(0, 50)
        .map((item) => {
          const name =
            activeLevel === "kabupaten"
              ? (item as KabupatenAnalysis).kabupaten_name
              : (item as KecamatanAnalysis).kecamatan_name;
          const pci =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_pci_score", "pci_score")
              : getValue(item, "pci_score");
          const production =
            activeLevel === "kabupaten"
              ? getValue(
                  item,
                  "latest_production_tons",
                  "average_production_tons"
                ) / 1000
              : Math.random() * 10 + 1;

          const fsci =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_fsci_score", "fsci_score")
              : getValue(item, "fsci_score");

          return {
            x: pci,
            y: production,
            name: name || `Region ${Math.floor(Math.random() * 1000)}`,
            category: fsci >= 70 ? "high" : fsci >= 50 ? "medium" : "low",
          };
        })
        .filter((item) => item.name && item.x > 0),

      // Efficiency data
      efficiencyData: activeData
        .slice(0, 20)
        .map((item) => {
          const name =
            activeLevel === "kabupaten"
              ? (item as KabupatenAnalysis).kabupaten_name
              : (item as KecamatanAnalysis).kecamatan_name;
          const fsci =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_fsci_score", "fsci_score")
              : getValue(item, "fsci_score");
          const pci =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_pci_score", "pci_score")
              : getValue(item, "pci_score");
          const psi =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_psi_score", "psi_score")
              : getValue(item, "psi_score");
          const crs =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_crs_score", "crs_score")
              : getValue(item, "crs_score");

          const potential = (pci + psi + crs) / 3;
          const efficiency = (fsci / potential) * 100;
          const gap = Math.max(0, potential - fsci);

          return {
            region: name || `Region ${Math.floor(Math.random() * 1000)}`,
            efficiency: Math.min(100, Math.max(0, efficiency)),
            potential,
            gap,
          };
        })
        .filter((item) => item.region),

      // FSCI breakdown data
      fsciBreakdownData: activeData
        .slice(0, 15)
        .map((item) => {
          const name =
            activeLevel === "kabupaten"
              ? (item as KabupatenAnalysis).kabupaten_name
              : (item as KecamatanAnalysis).kecamatan_name;
          const fsci =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_fsci_score", "fsci_score")
              : getValue(item, "fsci_score");
          const pci =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_pci_score", "pci_score")
              : getValue(item, "pci_score");
          const psi =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_psi_score", "psi_score")
              : getValue(item, "psi_score");
          const crs =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_crs_score", "crs_score")
              : getValue(item, "crs_score");

          return {
            region: name || `Region ${Math.floor(Math.random() * 1000)}`,
            pci,
            psi,
            crs,
            fsci,
          };
        })
        .filter((item) => item.region),
    };

    // Generate rankings
    const rankings = {
      topPerformers: activeData
        .map((item) => {
          const name =
            activeLevel === "kabupaten"
              ? (item as KabupatenAnalysis).kabupaten_name
              : (item as KecamatanAnalysis).kecamatan_name;
          const score =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_fsci_score", "fsci_score")
              : getValue(item, "fsci_score");
          return { name: name || "Unknown", score };
        })
        .filter((item) => item.name !== "Unknown" && item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 10)
        .map((item, index) => ({ ...item, rank: index + 1 })),

      criticalRegions: activeData
        .map((item) => {
          const name =
            activeLevel === "kabupaten"
              ? (item as KabupatenAnalysis).kabupaten_name
              : (item as KecamatanAnalysis).kecamatan_name;
          const score =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_fsci_score", "fsci_score")
              : getValue(item, "fsci_score");
          const priority = Math.max(
            1,
            Math.min(10, Math.round((100 - score) / 10))
          );
          return { name: name || "Unknown", score, priority };
        })
        .filter(
          (item) => item.name !== "Unknown" && item.score > 0 && item.score < 60
        )
        .sort((a, b) => a.score - b.score)
        .slice(0, 10),

      improvedRegions: activeData
        .map((item) => {
          const name =
            activeLevel === "kabupaten"
              ? (item as KabupatenAnalysis).kabupaten_name
              : (item as KecamatanAnalysis).kecamatan_name;
          const current =
            activeLevel === "kabupaten"
              ? getValue(item, "aggregated_fsci_score", "fsci_score")
              : getValue(item, "fsci_score");
          // Simulate improvement (would be calculated from historical data)
          const improvement = Math.random() * 10 - 2; // -2 to +8 range
          return { name: name || "Unknown", improvement, current };
        })
        .filter((item) => item.name !== "Unknown" && item.improvement > 0)
        .sort((a, b) => b.improvement - a.improvement)
        .slice(0, 10),
    };

    // Extract filter options
    const filters = {
      availableYears: Array.from({ length: 7 }, (_, i) => 2018 + i),
      availableSeasons: ["all", "wet", "dry", "transition"],
      availableDistricts: [
        "all",
        ...Array.from(
          new Set(
            activeData
              .map((item) =>
                activeLevel === "kabupaten"
                  ? (item as KabupatenAnalysis).kabupaten_name
                  : (item as KecamatanAnalysis).kecamatan_name
              )
              .filter((name) => name)
          )
        ),
      ],
    };

    return {
      rawData,
      kabupatenData,
      kecamatanData,
      metrics,
      chartData,
      rankings,
      filters,
    };
  }, [rawData, activeLevel]);

  // Additional hook functions
  const updateParams = (newParams: Partial<TwoLevelAnalysisParams>) => {
    // This would trigger a new query with updated parameters
    return refetch();
  };

  const switchLevel = (level: "kabupaten" | "kecamatan") => {
    setActiveLevel(level);
  };

  const exportData = (format: "json" | "csv" = "json") => {
    if (!processedData.rawData) return null;

    if (format === "json") {
      return JSON.stringify(processedData, null, 2);
    } else {
      // CSV export (simplified)
      const activeData =
        activeLevel === "kabupaten"
          ? processedData.kabupatenData
          : processedData.kecamatanData;

      const headers = ["Name", "FSCI", "PCI", "PSI", "CRS"];
      const rows = activeData.map((item) => [
        activeLevel === "kabupaten"
          ? (item as KabupatenAnalysis).kabupaten_name
          : (item as KecamatanAnalysis).kecamatan_name,
        activeLevel === "kabupaten"
          ? (item as KabupatenAnalysis).aggregated_fsci_score
          : (item as KecamatanAnalysis).fsci_score,
        activeLevel === "kabupaten"
          ? (item as KabupatenAnalysis).aggregated_pci_score
          : (item as KecamatanAnalysis).pci_score,
        activeLevel === "kabupaten"
          ? (item as KabupatenAnalysis).aggregated_psi_score
          : (item as KecamatanAnalysis).psi_score,
        activeLevel === "kabupaten"
          ? (item as KabupatenAnalysis).aggregated_crs_score
          : (item as KecamatanAnalysis).crs_score,
      ]);

      return [headers, ...rows].map((row) => row.join(",")).join("\n");
    }
  };

  return {
    // Data
    data: processedData,
    rawData: processedData.rawData,
    metrics: processedData.metrics,
    chartData: processedData.chartData,
    rankings: processedData.rankings,
    filters: processedData.filters,

    // State
    isLoading,
    error,
    isFetching,
    isRefetching,
    activeLevel,

    // Actions
    refetch,
    updateParams,
    switchLevel,
    exportData,

    // Utilities
    isEmpty: !processedData.rawData || processedData.metrics.totalRegions === 0,
    hasError: !!error,
    isReady: !isLoading && !error && !!processedData.rawData,
  };
}

// Additional utility hooks for specific use cases
export function useFoodSecurityMetrics(
  params?: Partial<TwoLevelAnalysisParams>
) {
  const { metrics, isLoading, error } = useDashboardData(params);
  return { metrics, isLoading, error };
}

export function useChartData(params?: Partial<TwoLevelAnalysisParams>) {
  const { chartData, isLoading, error } = useDashboardData(params);
  return { chartData, isLoading, error };
}

export function useRankingData(params?: Partial<TwoLevelAnalysisParams>) {
  const { rankings, isLoading, error } = useDashboardData(params);
  return { rankings, isLoading, error };
}
