"use client";

import { useState, useCallback, useEffect } from "react";
import {
  getFoodSecurityAnalysis,
  getSpatialDistricts,
  getSpatialParameters,
  exportFoodSecurityAnalysisCsv,
  type FoodSecurityAnalysisParams,
  type FoodSecurityResponse,
  type SpatialFeature,
} from "@/lib/fetch/spatial.map.fetch";

interface UseFoodSecurityAnalysisState {
  // Data
  data: FoodSecurityResponse | null;
  districts: any[] | null;
  parameters: any | null;

  // UI State
  loading: boolean;
  error: string | null;
  exporting: boolean;

  // Analysis State
  selectedDistricts: string[];
  analysisParams: FoodSecurityAnalysisParams;

  // Statistics
  totalFeatures: number;
  averageFSCI: number | null;
  highPotentialCount: number;
  classificationDistribution: Record<string, number>;
}

interface UseFoodSecurityAnalysisActions {
  // Data fetching
  fetchAnalysis: (params?: FoodSecurityAnalysisParams) => Promise<void>;
  fetchDistricts: () => Promise<void>;
  fetchParameters: () => Promise<void>;

  // Parameter management
  updateParams: (params: Partial<FoodSecurityAnalysisParams>) => void;
  resetParams: () => void;

  // District management
  setSelectedDistricts: (districts: string[]) => void;
  toggleDistrict: (district: string) => void;

  // Export functionality
  exportToCsv: (filename?: string) => Promise<void>;

  // Data utilities
  getFeaturesByFSCIClass: (fsciClass: string) => SpatialFeature[];
  getDistrictSummary: (districtName: string) => SpatialFeature | null;

  // State management
  clearError: () => void;
  refresh: () => Promise<void>;
}

const defaultParams: FoodSecurityAnalysisParams = {
  districts: "all",
  year_start: 2018,
  year_end: 2024,
  season: "all",
  aggregation: "mean",
  include_recommendations: true,
  analysis_level: "both",
  include_bps_data: true,
  bps_start_year: 2018,
  bps_end_year: 2024,
};

export function useFoodSecurityAnalysis(): UseFoodSecurityAnalysisState &
  UseFoodSecurityAnalysisActions {
  // State
  const [state, setState] = useState<UseFoodSecurityAnalysisState>({
    data: null,
    districts: null,
    parameters: null,
    loading: false,
    error: null,
    exporting: false,
    selectedDistricts: [],
    analysisParams: defaultParams,
    totalFeatures: 0,
    averageFSCI: null,
    highPotentialCount: 0,
    classificationDistribution: {},
  });

  // Calculate statistics when data changes
  useEffect(() => {
    if (state.data?.features) {
      const features = state.data.features;
      const totalFeatures = features.length;

      // Calculate average FSCI
      const fsciScores = features
        .map((f) => f.properties.fsci_score)
        .filter((score) => score !== undefined && score !== null) as number[];
      const averageFSCI =
        fsciScores.length > 0
          ? fsciScores.reduce((sum, score) => sum + score, 0) /
            fsciScores.length
          : null;

      // Count high potential areas (FSCI >= 80)
      const highPotentialCount = fsciScores.filter(
        (score) => score >= 80
      ).length;

      // Calculate classification distribution
      const classificationDistribution = features.reduce((acc, feature) => {
        const fsciClass = feature.properties.fsci_class || "Unknown";
        acc[fsciClass] = (acc[fsciClass] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      setState((prev) => ({
        ...prev,
        totalFeatures,
        averageFSCI: averageFSCI ? Math.round(averageFSCI * 10) / 10 : null,
        highPotentialCount,
        classificationDistribution,
      }));
    }
  }, [state.data]);

  // Actions
  const fetchAnalysis = useCallback(
    async (params?: FoodSecurityAnalysisParams) => {
      try {
        setState((prev) => ({ ...prev, loading: true, error: null }));

        const analysisParams = params || state.analysisParams;
        const data = await getFoodSecurityAnalysis(analysisParams);

        setState((prev) => ({
          ...prev,
          data,
          loading: false,
          analysisParams: params
            ? { ...prev.analysisParams, ...params }
            : prev.analysisParams,
        }));

        console.log("🌾 Food security analysis loaded:", {
          features: data.features.length,
          avg_fsci:
            data.metadata?.food_security_analysis?.fsci_summary?.average_fsci,
        });
      } catch (error) {
        setState((prev) => ({
          ...prev,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : "Failed to fetch food security analysis",
        }));
      }
    },
    [state.analysisParams]
  );

  const fetchDistricts = useCallback(async () => {
    try {
      const districts = await getSpatialDistricts();
      setState((prev) => ({ ...prev, districts: districts.districts }));
    } catch (error) {
      console.error("Failed to fetch districts:", error);
    }
  }, []);

  const fetchParameters = useCallback(async () => {
    try {
      const parameters = await getSpatialParameters();
      setState((prev) => ({ ...prev, parameters }));
    } catch (error) {
      console.error("Failed to fetch parameters:", error);
    }
  }, []);

  const updateParams = useCallback(
    (params: Partial<FoodSecurityAnalysisParams>) => {
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

  const setSelectedDistricts = useCallback((districts: string[]) => {
    setState((prev) => ({ ...prev, selectedDistricts: districts }));
  }, []);

  const toggleDistrict = useCallback((district: string) => {
    setState((prev) => ({
      ...prev,
      selectedDistricts: prev.selectedDistricts.includes(district)
        ? prev.selectedDistricts.filter((d) => d !== district)
        : [...prev.selectedDistricts, district],
    }));
  }, []);

  const exportToCsv = useCallback(
    async (filename?: string) => {
      if (!state.data) {
        throw new Error("No data to export");
      }

      try {
        setState((prev) => ({ ...prev, exporting: true }));

        const result = await exportFoodSecurityAnalysisCsv(
          state.data,
          filename
        );

        if (result.success) {
          console.log("✅ Food security data exported:", result.filename);
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
    [state.data]
  );

  const getFeaturesByFSCIClass = useCallback(
    (fsciClass: string) => {
      if (!state.data?.features) return [];
      return state.data.features.filter(
        (f) => f.properties.fsci_class === fsciClass
      );
    },
    [state.data]
  );

  const getDistrictSummary = useCallback(
    (districtName: string) => {
      if (!state.data?.features) return null;
      return (
        state.data.features.find(
          (f) =>
            f.properties.NAME_3 === districtName ||
            f.properties.kecamatan_name === districtName ||
            f.properties.kabupaten_name === districtName
        ) || null
      );
    },
    [state.data]
  );

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  const refresh = useCallback(async () => {
    await fetchAnalysis();
  }, [fetchAnalysis]);

  // Initial data fetch
  useEffect(() => {
    fetchDistricts();
    fetchParameters();
  }, [fetchDistricts, fetchParameters]);

  return {
    ...state,
    fetchAnalysis,
    fetchDistricts,
    fetchParameters,
    updateParams,
    resetParams,
    setSelectedDistricts,
    toggleDistrict,
    exportToCsv,
    getFeaturesByFSCIClass,
    getDistrictSummary,
    clearError,
    refresh,
  };
}
