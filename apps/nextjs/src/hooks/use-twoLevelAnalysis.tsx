"use client";

import { useState, useCallback, useEffect } from "react";
import {
  getTwoLevelAnalysis,
  exportTwoLevelAnalysisCsv,
  type TwoLevelAnalysisParams,
  type TwoLevelAnalysisResponse,
  type KabupatenAnalysis,
} from "@/lib/fetch/spatial.map.fetch";

interface UseTwoLevelAnalysisState {
  // Main Analysis Data
  analysisData: TwoLevelAnalysisResponse | null;

  // UI State
  loading: boolean;
  error: string | null;
  exporting: boolean;

  // Analysis Parameters
  analysisParams: TwoLevelAnalysisParams;

  // Selection State
  selectedKabupaten: string | null;
  selectedKecamatan: string | null;

  // Performance Metrics
  overperformingKabupaten: KabupatenAnalysis[];
  underperformingKabupaten: KabupatenAnalysis[];
  averageCorrelation: number | null;
  totalProductionTons: number;
}

interface UseTwoLevelAnalysisActions {
  // Data fetching
  fetchTwoLevelAnalysis: (params?: TwoLevelAnalysisParams) => Promise<void>;

  // Parameter management
  updateParams: (params: Partial<TwoLevelAnalysisParams>) => void;
  resetParams: () => void;

  // Selection management
  selectKabupaten: (kabupaten: string) => void;
  selectKecamatan: (kecamatan: string) => void;
  clearSelections: () => void;

  // Data utilities
  getKabupatenDetails: (kabupatenName: string) => KabupatenAnalysis | null;
  getCorrelationInsights: (kabupatenName: string) => any | null;
  getPerformanceCategory: (kabupatenName: string) => string | null;

  // Export functionality
  exportAnalysisToCsv: (filename?: string) => Promise<void>;

  // State management
  clearError: () => void;
  refresh: () => Promise<void>;
}

const defaultParams: TwoLevelAnalysisParams = {
  year_start: 2018,
  year_end: 2024,
  bps_start_year: 2018,
  bps_end_year: 2024,
  season: "all",
  aggregation: "mean",
  districts: "all",
};

export function useTwoLevelAnalysis(): UseTwoLevelAnalysisState &
  UseTwoLevelAnalysisActions {
  // State
  const [state, setState] = useState<UseTwoLevelAnalysisState>({
    analysisData: null,
    loading: false,
    error: null,
    exporting: false,
    analysisParams: defaultParams,
    selectedKabupaten: null,
    selectedKecamatan: null,
    overperformingKabupaten: [],
    underperformingKabupaten: [],
    averageCorrelation: null,
    totalProductionTons: 0,
  });

  // Calculate performance metrics when analysis data changes
  useEffect(() => {
    if (state.analysisData?.level_2_kabupaten_analysis?.data) {
      const kabupatenData = state.analysisData.level_2_kabupaten_analysis.data;

      // Filter performance categories
      const overperforming = kabupatenData.filter(
        (k) => k.performance_gap_category === "overperforming"
      );
      const underperforming = kabupatenData.filter(
        (k) => k.performance_gap_category === "underperforming"
      );

      // Calculate average correlation
      const correlations = kabupatenData
        .map((k) => k.climate_production_correlation)
        .filter((c) => c !== null && !isNaN(c));
      const averageCorrelation =
        correlations.length > 0
          ? correlations.reduce((sum, c) => sum + c, 0) / correlations.length
          : null;

      // Calculate total production
      const totalProduction = kabupatenData.reduce(
        (sum, k) => sum + (k.latest_production_tons || 0),
        0
      );

      setState((prev) => ({
        ...prev,
        overperformingKabupaten: overperforming,
        underperformingKabupaten: underperforming,
        averageCorrelation: averageCorrelation
          ? Math.round(averageCorrelation * 1000) / 1000
          : null,
        totalProductionTons: totalProduction,
      }));
    }
  }, [state.analysisData]);

  // Actions
  const fetchTwoLevelAnalysis = useCallback(
    async (params?: TwoLevelAnalysisParams) => {
      try {
        setState((prev) => ({ ...prev, loading: true, error: null }));

        const analysisParams = params || state.analysisParams;
        const data = await getTwoLevelAnalysis(analysisParams);

        setState((prev) => ({
          ...prev,
          analysisData: data,
          loading: false,
          analysisParams: params
            ? { ...prev.analysisParams, ...params }
            : prev.analysisParams,
        }));

        console.log("🏛️ Two-level analysis loaded:", {
          kabupaten_count: data.metadata.level_2_kabupaten_count,
          kecamatan_count: data.metadata.level_1_kecamatan_count,
          correlation:
            data.summary_statistics.average_climate_production_correlation,
        });
      } catch (error) {
        setState((prev) => ({
          ...prev,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : "Failed to fetch two-level analysis",
        }));
      }
    },
    [state.analysisParams]
  );

  const updateParams = useCallback(
    (params: Partial<TwoLevelAnalysisParams>) => {
      setState((prev) => ({
        ...prev,
        analysisParams: { ...prev.analysisParams, ...params },
      }));
    },
    []
  );

  const resetParams = useCallback(() => {
    setState((prev) => ({ ...prev, analysisParams: defaultParams }));
  }, []);

  const selectKabupaten = useCallback((kabupaten: string) => {
    setState((prev) => ({
      ...prev,
      selectedKabupaten: kabupaten,
      selectedKecamatan: null,
    }));
  }, []);

  const selectKecamatan = useCallback((kecamatan: string) => {
    setState((prev) => ({ ...prev, selectedKecamatan: kecamatan }));
  }, []);

  const clearSelections = useCallback(() => {
    setState((prev) => ({
      ...prev,
      selectedKabupaten: null,
      selectedKecamatan: null,
    }));
  }, []);

  const getKabupatenDetails = useCallback(
    (kabupatenName: string) => {
      if (!state.analysisData?.level_2_kabupaten_analysis?.data) return null;
      return (
        state.analysisData.level_2_kabupaten_analysis.data.find(
          (k) =>
            k.kabupaten_name === kabupatenName ||
            k.bps_compatible_name === kabupatenName
        ) || null
      );
    },
    [state.analysisData]
  );

  const getCorrelationInsights = useCallback(
    (kabupatenName: string) => {
      if (
        !state.analysisData?.cross_level_insights?.climate_production_alignment
      )
        return null;
      return (
        state.analysisData.cross_level_insights.climate_production_alignment[
          kabupatenName
        ] || null
      );
    },
    [state.analysisData]
  );

  const getPerformanceCategory = useCallback(
    (kabupatenName: string) => {
      const details = getKabupatenDetails(kabupatenName);
      return details?.performance_gap_category || null;
    },
    [getKabupatenDetails]
  );

  const exportAnalysisToCsv = useCallback(
    async (filename?: string) => {
      if (!state.analysisData) {
        throw new Error("No analysis data to export");
      }

      try {
        setState((prev) => ({ ...prev, exporting: true }));

        const result = await exportTwoLevelAnalysisCsv(
          state.analysisData,
          filename
        );

        if (result.success) {
          console.log("✅ Two-level analysis exported:", result.filename);
        } else {
          throw new Error(result.message);
        }
      } catch (error) {
        setState((prev) => ({
          ...prev,
          error: error instanceof Error ? error.message : "Export failed",
        }));
      } finally {
        setState((prev) => ({ ...prev, exporting: false }));
      }
    },
    [state.analysisData]
  );

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  const refresh = useCallback(async () => {
    await fetchTwoLevelAnalysis();
  }, [fetchTwoLevelAnalysis]);

  return {
    ...state,
    fetchTwoLevelAnalysis,
    updateParams,
    resetParams,
    selectKabupaten,
    selectKecamatan,
    clearSelections,
    getKabupatenDetails,
    getCorrelationInsights,
    getPerformanceCategory,
    exportAnalysisToCsv,
    clearError,
    refresh,
  };
}
