"use client";
import { useState, useEffect } from "react";
import {
  getSpatialAnalysis,
  getSpatialDistricts,
  getSpatialParameters,
  type SpatialAnalysisParams,
  type SpatialAnalysisResponse,
} from "@/lib/fetch/files.fetch";

interface UseSpatialAnalysisResult {
  // Data
  analysisData: SpatialAnalysisResponse | null;
  districts: any[] | null;
  parameters: any | null;

  // States
  isLoading: boolean;
  isAnalysisLoading: boolean;
  error: string | null;

  // Actions
  runAnalysis: (params: SpatialAnalysisParams) => Promise<void>;
  clearError: () => void;
  refetchDistricts: () => Promise<void>;
  refetchParameters: () => Promise<void>;
}

export function useSpatialAnalysis(
  initialParams: SpatialAnalysisParams = {
    districts: "all",
    parameters: "all",
    year_start: 2020,
    year_end: 2023,
    season: "all",
    aggregation: "mean",
  }
): UseSpatialAnalysisResult {
  // State management
  const [analysisData, setAnalysisData] =
    useState<SpatialAnalysisResponse | null>(null);
  const [districts, setDistricts] = useState<any[] | null>(null);
  const [parameters, setParameters] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAnalysisLoading, setIsAnalysisLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load initial data (districts + parameters)
  useEffect(() => {
    const loadInitialData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        console.log("🔄 Loading initial spatial data...");

        // Load districts and parameters in parallel
        const [districtsResult, parametersResult] = await Promise.all([
          getSpatialDistricts(),
          getSpatialParameters(),
        ]);

        setDistricts(districtsResult.data?.districts || []);
        setParameters(parametersResult);

        console.log("✅ Initial data loaded successfully");
        console.log(`📍 Found ${districtsResult.total_districts} districts`);

        // Run initial analysis
        await runAnalysisInternal(initialParams);
      } catch (err: any) {
        console.error("❌ Failed to load initial data:", err);
        setError(err.message || "Failed to load spatial data");
      } finally {
        setIsLoading(false);
      }
    };

    loadInitialData();
  }, []); // Run once on mount

  // Internal analysis function
  const runAnalysisInternal = async (params: SpatialAnalysisParams) => {
    setIsAnalysisLoading(true);
    setError(null);

    try {
      console.log("🗺️ Running spatial analysis with params:", params);

      const result = await getSpatialAnalysis(params);
      setAnalysisData(result);

      console.log("✅ Spatial analysis completed");
      console.log(`📊 Analyzed ${result.features?.length || 0} districts`);
    } catch (err: any) {
      console.error("❌ Spatial analysis failed:", err);
      setError(err.message || "Spatial analysis failed");
      setAnalysisData(null);
    } finally {
      setIsAnalysisLoading(false);
    }
  };

  // Public analysis function
  const runAnalysis = async (params: SpatialAnalysisParams) => {
    await runAnalysisInternal(params);
  };

  // Refetch functions
  const refetchDistricts = async () => {
    try {
      setError(null);
      console.log("🔄 Refetching districts...");

      const result = await getSpatialDistricts();
      setDistricts(result.data?.districts || []);

      console.log("✅ Districts refetched successfully");
    } catch (err: any) {
      console.error("❌ Failed to refetch districts:", err);
      setError(err.message || "Failed to refetch districts");
    }
  };

  const refetchParameters = async () => {
    try {
      setError(null);
      console.log("🔄 Refetching parameters...");

      const result = await getSpatialParameters();
      setParameters(result);

      console.log("✅ Parameters refetched successfully");
    } catch (err: any) {
      console.error("❌ Failed to refetch parameters:", err);
      setError(err.message || "Failed to refetch parameters");
    }
  };

  // Clear error function
  const clearError = () => {
    setError(null);
  };

  return {
    // Data
    analysisData,
    districts,
    parameters,

    // States
    isLoading,
    isAnalysisLoading,
    error,

    // Actions
    runAnalysis,
    clearError,
    refetchDistricts,
    refetchParameters,
  };
}
