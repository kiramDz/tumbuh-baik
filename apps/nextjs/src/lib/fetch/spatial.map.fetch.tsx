import axios from "axios";

export interface SpatialAnalysisResponse {
  type: "FeatureCollection";
  metadata: {
    analysis_date: string;
    parameters_used: {
      districts: string;
      fsci_parameters: string;
      analysis_period: {
        start_year: number;
        end_year: number;
      };
      season_filter: string;
      aggregation_method: string;
      include_geometry: boolean;
      output_format: string;
    };
    total_districts: number;
    analyzed_districts: number;
    data_source: string;
    processing_backend: string;
    fsci_methodology: string;
  };
  features: SpatialFeature[];
}

export interface SpatialAnalysisParams {
  districts?: string;
  year_start?: number;
  year_end?: number;
  season?: "wet" | "dry" | "all";
  aggregation?: "mean" | "median" | "percentile";
  analysis_type?: "two-level" | "food-security";
  include_bps_data?: boolean;
  bps_start_year?: number;
  bps_end_year?: number;
  fsci_components?: "fsci" | "pci" | "psi" | "crs" | "all";
}

export interface SpatialFeature {
  type: "Feature";
  id: string;
  geometry: {
    type: "MultiPolygon";
    coordinates: number[][][];
  };
  bbox: number[];
  properties: {
    // Administrative Data
    GID_3: string;
    NAME_1: string;
    NAME_2: string;
    NAME_3: string;
    TYPE_3: string;

    // Legacy Climate Data (for backward compatibility)
    avg_temperature: number;
    avg_precipitation: number;
    avg_humidity: number;

    //  ADD: Basic FSCI support in base SpatialFeature
    fsci_score?: number;
    fsci_class?:
      | "Lumbung Pangan Primer"
      | "Lumbung Pangan Sekunder"
      | "Lumbung Pangan Tersier";
    pci_score?: number;
    psi_score?: number;
    crs_score?: number;

    // Location Data
    nasa_lat: number;
    nasa_lng: number;
    centroid_lat: number;
    centroid_lng: number;
    nasa_match: string;
  };
}

export interface TwoLevelAnalysisParams {
  year_start?: number;
  year_end?: number;
  bps_start_year?: number;
  bps_end_year?: number;
  season?: "wet" | "dry" | "all";
  aggregation?: "mean" | "median" | "percentile";
  districts?: string;
}

export interface TwoLevelAnalysisResponse {
  type: "TwoLevelFoodSecurityAnalysis";
  metadata: {
    analysis_timestamp: string;
    analysis_period: string;
    bps_data_period: string;
    level_1_kecamatan_count: number;
    level_2_kabupaten_count: number;
    methodology_summary: string;
    data_sources: {
      climate_analysis: string;
      production_validation: string;
      spatial_boundaries: string;
      administrative_mapping: string;
    };
  };
  level_1_kecamatan_analysis: {
    analysis_count: number;
    description: string;
    data: KecamatanAnalysis[];
  };
  level_2_kabupaten_analysis: {
    analysis_count: number;
    description: string;
    data: KabupatenAnalysis[];
  };
  cross_level_insights: CrossLevelInsights;
  rankings: {
    climate_potential_ranking: ClimateRanking[];
    actual_production_ranking: ProductionRanking[];
  };
  recommendations: PolicyRecommendation[];
  summary_statistics: SummaryStatistics;
  temporal_alignment: TemporalAlignment;
}

export interface FoodSecurityAnalysisParams {
  districts?: string;
  year_start?: number;
  year_end?: number;
  season?: "wet" | "dry" | "all";
  aggregation?: "mean" | "median" | "percentile";
  include_recommendations?: boolean;
  analysis_level?: "kecamatan" | "kabupaten" | "both";
  include_bps_data?: boolean;
  bps_start_year?: number;
  bps_end_year?: number;
}

export interface FoodSecurityResponse {
  type: "FeatureCollection";
  metadata: SpatialAnalysisResponse["metadata"] & {
    food_security_analysis: {
      fsci_summary: {
        average_fsci: number;
        classification_distribution: Record<string, number>;
        high_potential_districts: number;
      };
      investment_priorities: {
        high_priority: string[];
        medium_priority: string[];
        low_priority: string[];
      };
    };
    administrative_mapping: {
      kabupaten_count: number;
      kecamatan_count: number;
      nasa_location_count: number;
      bps_compatible: boolean;
    };
  };
  features: FoodSecurityFeature[];
}

export interface FoodSecurityFeature extends SpatialFeature {
  properties: SpatialFeature["properties"] & {
    // FSCI Components
    fsci_score: number;
    fsci_class:
      | "Lumbung Pangan Primer"
      | "Lumbung Pangan Sekunder"
      | "Lumbung Pangan Tersier";
    pci_score: number;
    psi_score: number;
    crs_score: number;

    kabupaten_name: string;
    kecamatan_name: string;
    area_km2: number;
    area_weight: number;

    // Investment Recommendations
    investment_recommendation: string;
    priority_level: "high" | "medium" | "low";

    // BPS Integration (optional)
    bps_compatible_name?: string;
    latest_production_tons?: number;
    production_efficiency_score?: number;
    climate_production_correlation?: number;
  };
}

export interface ScatterPlotPoint {
  kabupaten_name: string;
  fsci_score: number;
  production_tons: number;
  area_km2: number;
  efficiency_score: number;
  performance_category: "overperforming" | "aligned" | "underperforming";
  climate_rank: number;
  production_rank: number;
  correlation: number;
}

export interface CorrelationMatrix {
  fsci_vs_production: number;
  pci_vs_production: number;
  psi_vs_production: number;
  crs_vs_production: number;
  climate_components_correlation: {
    pci_psi: number;
    pci_crs: number;
    psi_crs: number;
  };
  production_metrics_correlation: {
    latest_vs_average: number;
    production_vs_efficiency: number;
  };
}

export interface TrendAnalysis {
  overall_trend: "improving" | "declining" | "stable";
  trend_strength: "strong" | "moderate" | "weak";
  correlation_over_time: {
    year: number;
    correlation_coefficient: number;
  }[];
  seasonal_patterns: {
    season: "wet" | "dry" | "all";
    average_correlation: number;
  }[];
  regional_trends: {
    kabupaten_name: string;
    trend_direction: "improving" | "declining" | "stable";
    correlation_change: number;
  }[];
}

export interface StatisticalSummary {
  correlation_statistics: {
    mean_correlation: number;
    median_correlation: number;
    std_deviation: number;
    min_correlation: number;
    max_correlation: number;
    confidence_interval_95: {
      lower_bound: number;
      upper_bound: number;
    };
  };
  performance_distribution: {
    overperforming_count: number;
    aligned_count: number;
    underperforming_count: number;
    overperforming_percentage: number;
    aligned_percentage: number;
    underperforming_percentage: number;
  };
  climate_production_metrics: {
    total_climate_potential: number;
    total_actual_production: number;
    overall_efficiency_percentage: number;
    potential_production_gap: number;
  };
  outlier_analysis: {
    climate_outliers: string[];
    production_outliers: string[];
    correlation_outliers: string[];
  };
}

export interface KecamatanAnalysis {
  kecamatan_name: string;
  nasa_location_name: string;
  kabupaten_name: string;
  area_km2: number;
  area_weight: number;
  fsci_score: number;
  fsci_class: string;
  pci_score: number;
  psi_score: number;
  crs_score: number;
  investment_recommendation: string;
}

export interface KabupatenAnalysis {
  kabupaten_name: string;
  bps_compatible_name: string;
  total_area_km2: number;
  constituent_kecamatan: string[];
  constituent_nasa_locations: string[];

  // Aggregated Climate Metrics
  aggregated_fsci_score: number;
  aggregated_fsci_class: string;
  aggregated_pci_score: number;
  aggregated_psi_score: number;
  aggregated_crs_score: number;

  // BPS Production Data
  latest_production_tons: number;
  average_production_tons: number;
  production_trend: "increasing" | "decreasing" | "stable";
  data_coverage_years: number;

  // Validation Metrics
  climate_production_correlation: number;
  production_efficiency_score: number;
  climate_potential_rank: number;
  actual_production_rank: number;
  performance_gap_category: "overperforming" | "aligned" | "underperforming";
  validation_notes: string;
}

export interface KabupatenSummary {
  kabupaten_name: string;
  bps_compatible_name: string;
  climate_summary: {
    fsci_score: number;
    fsci_class: string;
    climate_rank: number;
  };
  production_summary: {
    latest_production_tons: number;
    production_rank: number;
    production_trend: string;
  };
  performance_summary: {
    correlation: number;
    efficiency_score: number;
    gap_category: string;
  };
}

export interface BPSHistoricalData {
  kabupaten_name: string;
  bps_compatible_name: string;
  data_period: string;
  data_coverage_years: number;
  yearly_production_data: {
    year: number;
    production_tons: number;
    luas_tanam_ha: number;
    luas_panen_ha: number;
    produktivitas_ton_ha: number;
    harvest_success_rate: number;
  }[];
  production_statistics: {
    total_production_tons: number;
    average_annual_production: number;
    max_production: number;
    min_production: number;
    production_volatility_percent: number;
  };
  trend_analysis: {
    overall_change_percent: number;
    linear_trend_slope: number;
    most_productive_year: number;
    least_productive_year: number;
  };
}

export interface MappingValidation {
  total_kabupaten: number;
  total_kecamatan: number;
  total_nasa_locations: number;
  bps_compatible_kabupaten: number;
  mapping_consistency: "excellent" | "good" | "fair" | "poor";
  validation_details: {
    kabupaten_name: string;
    kecamatan_count: number;
    nasa_location_count: number;
    total_area_km2: number;
    bps_compatible: boolean;
  }[];
}

export interface CrossLevelInsights {
  climate_production_alignment: Record<
    string,
    {
      climate_rank: number;
      production_rank: number;
      correlation: number;
      performance_category: string;
      rank_difference: number;
    }
  >;
  spatial_variability: Record<
    string,
    {
      fsci_variance: number;
      internal_variability: string;
      kecamatan_fsci_range: string;
      largest_contributor: string;
    }
  >;
  methodology_validation: {
    overall_climate_production_correlation: number;
    sample_size: string;
    validation_strength: "strong" | "moderate" | "weak";
  };
  policy_recommendations: Record<string, any>;
}

export interface PolicyRecommendation {
  category: string;
  target: string[];
  recommendation: string;
  rationale: string;
}

export interface ClimateRanking {
  rank: number;
  kabupaten_name: string;
  fsci_score: number;
  fsci_class: string;
  constituent_kecamatan: number;
}

export interface ProductionRanking {
  rank: number;
  kabupaten_name: string;
  latest_production_tons: number;
  production_trend: string;
  data_years: number;
}

export interface SummaryStatistics {
  total_production_tons: number;
  average_fsci_score: number;
  high_potential_kabupaten: number;
  underperforming_kabupaten: number;
  average_climate_production_correlation: number;
  data_coverage: {
    full_bps_coverage: number;
    limited_bps_coverage: number;
  };
}

export interface TemporalAlignment {
  nasa_period: string;
  bps_period: string;
  alignment_method: string;
  overlap_years: number;
  correlation_quality: "high" | "moderate" | "low";
}

// Spatial Analysis Response for flask
const getFlaskBaseUrl = () => {
  return process.env.NEXT_PUBLIC_FLASK_URL || "http://localhost:5001";
};

export const getSpatialDistricts = async () => {
  try {
    const response = await axios.get(
      `${getFlaskBaseUrl()}/api/v1/spatial-map/districts`,
      {
        timeout: 10000,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    // District data with food security context ONLY
    const enhancedDistricts = {
      districts: response.data.districts || response.data,
      metadata: {
        total_districts:
          response.data.total_districts || response.data.districts?.length || 0,
        analysis_capability: "FSCI Food Security Analysis",
        supported_analysis_types: ["food-security", "two-level"],
        bps_compatible_count: response.data.bps_compatible_count || "Unknown",
        last_updated: response.data.last_updated || new Date().toISOString(),
      },
    };

    return enhancedDistricts;
  } catch (error: any) {
    console.error("❌ Food security districts fetch error:", error);
    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    }
    throw new Error("Failed to fetch districts for food security analysis");
  }
};

export const getSpatialParameters = async () => {
  try {
    const response = await axios.get(
      `${getFlaskBaseUrl()}/api/v1/spatial-map/parameters`,
      {
        timeout: 10000,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    // Transform response to focus ONLY on FSCI parameters
    const fsciParameters = {
      analysis_types: [
        {
          value: "food-security",
          label: "Food Security Index (FSCI)",
          description: "Comprehensive FSCI with PCI/PSI/CRS components",
        },
        {
          value: "two-level",
          label: "Two-Level Analysis",
          description: "Climate potential vs BPS production validation",
        },
      ],
      fsci_components: [
        {
          value: "fsci",
          label: "Complete FSCI",
          description: "Production Capacity + Stability + Climate Risk",
        },
        {
          value: "pci",
          label: "Production Capacity Index",
          description: "Climate suitability for food crop production",
        },
        {
          value: "psi",
          label: "Production Stability Index",
          description: "Climate consistency and reliability",
        },
        {
          value: "crs",
          label: "Climate Risk Score",
          description: "Climate variability and extreme weather risk",
        },
      ],
      aggregation_methods: [
        {
          value: "mean",
          label: "Average",
          description: "Mean aggregation across time period",
        },
        {
          value: "median",
          label: "Median",
          description: "Median aggregation (robust to outliers)",
        },
        {
          value: "percentile",
          label: "Percentile",
          description: "75th percentile aggregation",
        },
      ],
      seasonal_options: [
        { value: "all", label: "Annual", description: "Full year analysis" },
        { value: "wet", label: "Wet Season", description: "November - April" },
        { value: "dry", label: "Dry Season", description: "May - October" },
      ],
    };

    return fsciParameters;
  } catch (error: any) {
    console.error("❌ FSCI parameters fetch error:", error);
    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    }
    throw new Error("Failed to fetch food security parameters");
  }
};

export const exportFoodSecurityAnalysisCsv = async (
  analysisData: FoodSecurityResponse,
  filename?: string
) => {
  try {
    const csvData = analysisData.features.map((feature) => ({
      // Administrative Information
      kecamatan_name:
        feature.properties.kecamatan_name || feature.properties.NAME_3,
      kabupaten_name:
        feature.properties.kabupaten_name || feature.properties.NAME_2,
      province: feature.properties.NAME_1,

      // Food Security Index Components
      fsci_score: feature.properties.fsci_score,
      fsci_classification: feature.properties.fsci_class,
      pci_score: feature.properties.pci_score,
      psi_score: feature.properties.psi_score,
      crs_score: feature.properties.crs_score,

      // Investment and Priority
      priority_level: feature.properties.priority_level,
      investment_recommendation: feature.properties.investment_recommendation,

      // BPS Integration (if available)
      latest_production_tons:
        feature.properties.latest_production_tons || "N/A",
      production_efficiency_score:
        feature.properties.production_efficiency_score || "N/A",
      climate_production_correlation:
        feature.properties.climate_production_correlation || "N/A",

      // Geographic Information
      area_km2: feature.properties.area_km2,
      area_weight: feature.properties.area_weight,

      // Legacy Climate Data (for reference)
      avg_temperature: feature.properties.avg_temperature,
      avg_precipitation: feature.properties.avg_precipitation,
      avg_humidity: feature.properties.avg_humidity,

      // Coordinates
      nasa_lat: feature.properties.nasa_lat,
      nasa_lng: feature.properties.nasa_lng,
    }));

    const headers = Object.keys(csvData[0]);
    let csvContent = headers.join(",") + "\n";

    csvData.forEach((row) => {
      const values = headers.map((header) => {
        let value = row[header as keyof typeof row];
        if (
          typeof value === "string" &&
          (value.includes(",") || value.includes('"'))
        ) {
          value = `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      });
      csvContent += values.join(",") + "\n";
    });

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");

    const defaultFilename = `food_security_analysis_${
      new Date().toISOString().split("T")[0]
    }.csv`;

    link.href = url;
    link.download = filename || defaultFilename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    return {
      success: true,
      message: `Successfully exported ${csvData.length} kecamatan with FSCI analysis`,
      filename: filename || defaultFilename,
    };
  } catch (error) {
    console.error("❌ Error exporting food security analysis CSV:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Export failed",
    };
  }
};

export const getTwoLevelAnalysis = async (
  params: TwoLevelAnalysisParams = {}
): Promise<TwoLevelAnalysisResponse> => {
  const maxRetries = 3;
  let currentRetry = 0;

  while (currentRetry < maxRetries) {
    try {
      const {
        year_start = 2018,
        year_end = 2024,
        bps_start_year = 2018,
        bps_end_year = 2024,
        season = "all",
        aggregation = "mean",
        districts = "all",
      } = params;

      const queryParams = new URLSearchParams({
        year_start: year_start.toString(),
        year_end: year_end.toString(),
        bps_start_year: bps_start_year.toString(),
        bps_end_year: bps_end_year.toString(),
        season,
        aggregation,
        districts,
      });

      console.log(
        `🔄 Attempt ${
          currentRetry + 1
        }/${maxRetries}: Fetching two-level analysis: ${queryParams.toString()}`
      );

      const response = await axios.get(
        `${getFlaskBaseUrl()}/api/v1/two-level/analysis?${queryParams}`,
        {
          timeout: 300000, // Increase to 5 minutes
          headers: {
            "Content-Type": "application/json",
          },
          // Add request interceptor for progress tracking
          onDownloadProgress: (progressEvent) => {
            // Fix: Add null check for progressEvent.total
            if (progressEvent.lengthComputable && progressEvent.total) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              console.log(`📥 Download progress: ${percentCompleted}%`);
            }
          },
        }
      );

      console.log("✅ Two-level analysis completed:", {
        kecamatan_count: response.data.metadata.level_1_kecamatan_count,
        kabupaten_count: response.data.metadata.level_2_kabupaten_count,
        correlation:
          response.data.summary_statistics
            .average_climate_production_correlation,
      });

      return response.data;
    } catch (error: any) {
      currentRetry++;

      console.error(`❌ Attempt ${currentRetry} failed:`, error.message);

      if (error.code === "ECONNABORTED" && currentRetry < maxRetries) {
        const delayMs = Math.pow(2, currentRetry) * 1000; // Exponential backoff
        console.log(`⏳ Retrying in ${delayMs / 1000}s...`);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        continue;
      }

      // Final attempt failed or non-timeout error
      if (error.code === "ECONNABORTED") {
        throw new Error(
          `Analysis timeout after ${maxRetries} attempts. The backend may be processing a large dataset. Please try with a smaller date range or fewer districts.`
        );
      }

      if (error.code === "ECONNREFUSED") {
        throw new Error(
          "Flask backend service unavailable - please check if the server is running"
        );
      }

      if (error.response?.status === 500 && error.response?.data?.error) {
        throw new Error(
          `Backend analysis failed: ${error.response.data.error}`
        );
      }

      if (error.response?.data?.message) {
        throw new Error(error.response.data.message);
      }

      throw new Error(`Failed to fetch two-level analysis: ${error.message}`);
    }
  }

  throw new Error("Max retries exceeded");
};

// export const getTwoLevelAnalysis = async (
//   params: TwoLevelAnalysisParams = {}
// ): Promise<TwoLevelAnalysisResponse> => {
//   try {
//     const {
//       year_start = 2018,
//       year_end = 2024,
//       bps_start_year = 2018,
//       bps_end_year = 2024,
//       season = "all",
//       aggregation = "mean",
//       districts = "all",
//     } = params;

//     const queryParams = new URLSearchParams({
//       year_start: year_start.toString(),
//       year_end: year_end.toString(),
//       bps_start_year: bps_start_year.toString(),
//       bps_end_year: bps_end_year.toString(),
//       season,
//       aggregation,
//       districts,
//     });
//     console.log(
//       `Fetching two-level food security analysis: ${queryParams.toString()}`
//     );
//     const response = await axios.get(
//       `${getFlaskBaseUrl()}/api/v1/two-level/analysis?${queryParams}`,
//       {
//         timeout: 120000, // Extended timeout for complex analysis
//         headers: {
//           "Content-Type": "application/json",
//         },
//       }
//     );
//     console.log("✅ Two-level analysis completed:", {
//       kecamatan_count: response.data.metadata.level_1_kecamatan_count,
//       kabupaten_count: response.data.metadata.level_2_kabupaten_count,
//       correlation:
//         response.data.summary_statistics.average_climate_production_correlation,
//     });

//     return response.data;
//   } catch (error: any) {
//     console.error("❌ Two-level analysis fetch error:", error);

//     if (error.code === "ECONNABORTED") {
//       throw new Error(
//         "Two-level analysis timeout - this is a complex operation, please try again"
//       );
//     }

//     if (error.code === "ECONNREFUSED") {
//       throw new Error("Flask backend service unavailable");
//     }

//     if (error.response?.status === 500 && error.response?.data?.error) {
//       throw new Error(`Analysis failed: ${error.response.data.error}`);
//     }

//     if (error.response?.data?.message) {
//       throw new Error(error.response.data.message);
//     }

//     throw new Error("Failed to fetch two-level food security analysis");
//   }
// };

export const getFoodSecurityAnalysis = async (
  params: FoodSecurityAnalysisParams = {}
): Promise<FoodSecurityResponse> => {
  try {
    const {
      districts = "all",
      year_start = 2018,
      year_end = 2024,
      season = "all",
      aggregation = "mean",
      include_recommendations = true,
      analysis_level = "both",
      include_bps_data = true,
      bps_start_year = 2018,
      bps_end_year = 2024,
    } = params;

    const queryParams = new URLSearchParams({
      districts,
      year_start: year_start.toString(),
      year_end: year_end.toString(),
      season,
      aggregation,
      analysis_type: "food-security", // Enhanced spatial analysis with FSCI
      include_recommendations: include_recommendations.toString(),
      analysis_level,
      include_bps_data: include_bps_data.toString(),
      bps_start_year: bps_start_year.toString(),
      bps_end_year: bps_end_year.toString(),
    });

    console.log("🌾📊 Fetching enhanced food security analysis...");

    const response = await axios.get(
      `${getFlaskBaseUrl()}/api/v1/spatial-map?${queryParams}`,
      {
        timeout: 90000, // Extended timeout for FSCI calculations
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    console.log("✅ Food security analysis completed:", {
      features_count: response.data.features.length,
      analysis_type: "Enhanced with FSCI components",
      bps_integration: include_bps_data,
    });

    return response.data;
  } catch (error: any) {
    console.error("❌ Food security analysis fetch error:", error);

    if (error.code === "ECONNABORTED") {
      throw new Error(
        "Food security analysis timeout - try with smaller date range or fewer districts"
      );
    }

    if (error.code === "ECONNREFUSED") {
      throw new Error("Flask backend service unavailable");
    }

    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    }

    throw new Error("Failed to fetch food security analysis");
  }
};

export const exportTwoLevelAnalysisCsv = async (
  analysisData: TwoLevelAnalysisResponse,
  filename?: string
) => {
  try {
    // Export kabupaten-level analysis with comprehensive metrics
    const csvData = analysisData.level_2_kabupaten_analysis.data.map(
      (kabupaten) => ({
        kabupaten_name: kabupaten.kabupaten_name,
        bps_compatible_name: kabupaten.bps_compatible_name,

        // Climate Metrics
        fsci_score: kabupaten.aggregated_fsci_score,
        fsci_class: kabupaten.aggregated_fsci_class,
        pci_score: kabupaten.aggregated_pci_score,
        psi_score: kabupaten.aggregated_psi_score,
        crs_score: kabupaten.aggregated_crs_score,

        // Production Metrics
        latest_production_tons: kabupaten.latest_production_tons,
        average_production_tons: kabupaten.average_production_tons,
        production_trend: kabupaten.production_trend,
        data_coverage_years: kabupaten.data_coverage_years,

        // Performance Metrics
        climate_production_correlation:
          kabupaten.climate_production_correlation,
        production_efficiency_score: kabupaten.production_efficiency_score,
        climate_potential_rank: kabupaten.climate_potential_rank,
        actual_production_rank: kabupaten.actual_production_rank,
        performance_gap_category: kabupaten.performance_gap_category,

        // Administrative
        total_area_km2: kabupaten.total_area_km2,
        constituent_kecamatan_count: kabupaten.constituent_kecamatan.length,
        validation_notes: kabupaten.validation_notes,
      })
    );

    const headers = Object.keys(csvData[0]);
    let csvContent = headers.join(",") + "\n";

    csvData.forEach((row) => {
      const values = headers.map((header) => {
        let value = row[header as keyof typeof row];
        if (
          typeof value === "string" &&
          (value.includes(",") || value.includes('"'))
        ) {
          value = `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      });
      csvContent += values.join(",") + "\n";
    });

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");

    const defaultFilename = `two_level_food_security_analysis_${
      new Date().toISOString().split("T")[0]
    }.csv`;

    link.href = url;
    link.download = filename || defaultFilename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    return {
      success: true,
      message: `Successfully exported ${csvData.length} kabupaten analysis`,
      filename: filename || defaultFilename,
    };
  } catch (error) {
    console.error("❌ Error exporting two-level analysis CSV:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Export failed",
    };
  }
};
