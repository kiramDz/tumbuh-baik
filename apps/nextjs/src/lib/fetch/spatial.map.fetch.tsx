import axios from "axios";

export interface SpatialAnalysisResponse {
  type: "FeatureCollection";
  metadata: {
    analysis_type: "food_security_index_spatial_analysis";
    analysis_date: string;
    parameters_used: {
      districts: string;
      analysis_period: {
        start_year: number;
        end_year: number;
      };
      aggregation_method: string;
      include_geometry: boolean;
      output_format: string;
    };
    total_districts: number;
    analyzed_districts: number;
    data_source: string;
    analysis_method: string;
    fsi_components: {
      natural_resources: string;
      availability: string;
    };
    summary_statistics: {
      average_scores: {
        fsi_score: number;
        natural_resources_score: number;
        availability_score: number;
      };
      fsi_distribution: Record<string, number>;
    };
    top_performing_districts: TopPerformingDistrict[];
    data_period: string;
    temporal_coverage: string;
  };
  features: SpatialFeature[];
}

export interface SpatialFeature {
  type: "Feature";
  id: string;
  geometry: {
    type: "MultiPolygon";
    coordinates: number[][][];
  };
  properties: {
    // Administrative Data
    GID_3: string;
    NAME_1: string;
    NAME_2: string;
    NAME_3: string;
    TYPE_3: string;

    // FSI Core Properties
    fsi_score: number;
    fsi_class: "Sangat Tinggi" | "Tinggi" | "Sedang" | "Rendah" | "No Data";
    natural_resources_score: number;
    availability_score: number;

    // Suitability (for backward compatibility)
    suitability_score: number;
    classification: string;
    confidence_level: string;

    // Component Scores
    temperature_score: number;
    precipitation_score: number;
    humidity_score: number;

    // Climate Averages
    avg_temperature: number;
    avg_precipitation: number;
    avg_humidity: number;

    // Risk Assessment
    overall_risk: string;

    // Location Data
    nasa_lat: number;
    nasa_lng: number;
    centroid_lat: number;
    centroid_lng: number;
    nasa_match: string;

    // Analysis Metadata
    analysis_timestamp: string;
  };
}

export interface TopPerformingDistrict {
  district: string;
  fsi_score: number;
  fsi_class: string;
  natural_resources_score: number;
  availability_score: number;
  ranking: number;
}

export interface FSIAnalysisParams {
  // Geographic scope
  districts?: string;

  // Time periods
  year_start?: number;
  year_end?: number;
  bps_start_year?: number;
  bps_end_year?: number;

  // Analysis options
  season?: "wet" | "dry" | "all";
  aggregation?: "mean" | "median" | "percentile";

  // Analysis levels
  analysis_level?: "kecamatan" | "kabupaten" | "both";
  include_bps_data?: boolean;
}

export interface SpatialAnalysisParams
  extends Omit<
    FSIAnalysisParams,
    "aggregation" | "bps_start_year" | "bps_end_year" | "season"
  > {
  aggregation?: "mean" | "median" | "max" | "min"; // Old enum
}

/**
 * @deprecated Use FSIAnalysisParams instead
 */
export interface TwoLevelAnalysisParams
  extends Omit<FSIAnalysisParams, "analysis_level" | "include_bps_data"> {
  // Same as FSIAnalysisParams but without analysis_level and include_bps_data
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
  summary_statistics: SummaryStatistics;
  temporal_alignment: TemporalAlignment;
}

export interface FoodSecurityAnalysisParams {
  districts?: string;
  year_start?: number;
  year_end?: number;
  season?: "wet" | "dry" | "all";
  aggregation?: "mean" | "median" | "percentile";
  analysis_level?: "kecamatan" | "kabupaten" | "both";
  include_bps_data?: boolean;
  bps_start_year?: number;
  bps_end_year?: number;
}

export interface FoodSecurityResponse {
  type: "FeatureCollection";
  metadata: SpatialAnalysisResponse["metadata"] & {
    food_security_analysis: {
      fsi_summary: {
        average_fsi: number;
        classification_distribution: Record<string, number>;
        high_potential_districts: number;
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
    fsi_score: number;
    fsi_class:
      | "Lumbung Pangan Primer"
      | "Lumbung Pangan Sekunder"
      | "Lumbung Pangan Tersier";
    natural_resources_score: number;
    availability_score: number;

    kabupaten_name: string;
    kecamatan_name: string;
    area_km2: number;
    area_weight: number;

    // BPS Integration (optional)
    bps_compatible_name?: string;
    latest_production_tons?: number;
    production_efficiency_score?: number;
    climate_production_correlation?: number;
  };
}

export interface ScatterPlotPoint {
  kabupaten_name: string;
  fsi_score: number;
  production_tons: number;
  area_km2: number;
  efficiency_score: number;
  performance_category: "overperforming" | "aligned" | "underperforming";
  climate_rank: number;
  production_rank: number;
  correlation: number;
}

export interface CorrelationMatrix {
  fsi_vs_production: number;
  natural_resources_vs_production: number;
  availability_vs_production: number;

  climate_components_correlation: {
    natural_resources_availability: number;
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

  // FSI Properties
  fsi_score: number;
  fsi_class: string;
  natural_resources_score: number;
  availability_score: number;

  // Location properties
  nasa_lat?: number;
  nasa_lng?: number;
  nasa_match?: string;
  centroid_lat?: number;
  centroid_lng?: number;
  GID_3?: string;
  NAME_1?: string;
  NAME_2?: string;
  NAME_3?: string;
  TYPE_3?: string;
}

export interface KabupatenAnalysis {
  kabupaten_name: string;
  bps_compatible_name: string;
  total_area_km2: number;
  constituent_kecamatan: string[];
  constituent_nasa_locations: string[];

  // Aggregated FSI Metrics
  aggregated_fsi_score: number;
  aggregated_fsi_class: string;
  aggregated_natural_resources_score: number;
  aggregated_availability_score: number;

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
    fsi_score: number;
    fsi_class: string;
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
      fsi_variance: number;
      internal_variability: string;
      kecamatan_fsi_range: string;
      largest_contributor: string;
    }
  >;
  methodology_validation: {
    overall_climate_production_correlation: number;
    sample_size: string;
    validation_strength: "strong" | "moderate" | "weak";
  };
}

export interface ClimateRanking {
  rank: number;
  kabupaten_name: string;
  fsi_score: number;
  fsi_class: string;
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
  average_fsi_score: number;
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

export const getSpatialFSIAnalysis = async (
  params: FSIAnalysisParams = {} //
): Promise<SpatialAnalysisResponse> => {
  try {
    const {
      districts = "all",
      year_start = 2018,
      year_end = 2024,
      aggregation = "mean",
    } = params;

    // Convert new enum values to backend-compatible values
    const backendAggregation =
      aggregation === "percentile" ? "mean" : aggregation;

    const queryParams = new URLSearchParams({
      districts,
      year_start: year_start.toString(),
      year_end: year_end.toString(),
      aggregation: backendAggregation,
    });

    console.log("🍚 Fetching FSI spatial analysis...");

    const response = await axios.get(
      `${getFlaskBaseUrl()}/api/v1/spatial-map?${queryParams}`,
      {
        timeout: 90000,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    console.log(" FSI spatial analysis completed:", {
      features_count: response.data.features.length,
      avg_fsi:
        response.data.metadata.summary_statistics.average_scores.fsi_score,
      analyzed_districts: response.data.metadata.analyzed_districts,
    });

    return response.data;
  } catch (error: any) {
    console.error("❌ FSI spatial analysis fetch error:", error);

    if (error.code === "ECONNABORTED") {
      throw new Error(
        "FSI analysis timeout - try with smaller date range or fewer districts"
      );
    }

    if (error.code === "ECONNREFUSED") {
      throw new Error("Flask backend service unavailable");
    }

    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    }

    throw new Error("Failed to fetch FSI spatial analysis");
  }
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

    const enhancedDistricts = {
      districts: response.data.data?.districts || response.data.districts || [],
      metadata: {
        total_districts: response.data.total_districts || 0,
        analysis_capability: "FSI Food Security Analysis",
        last_updated: new Date().toISOString(),
      },
    };

    return enhancedDistricts;
  } catch (error: any) {
    console.error("❌ FSI districts fetch error:", error);
    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    }
    throw new Error("Failed to fetch districts for FSI analysis");
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

    // Simplified FSI parameters
    const fsiParameters = {
      fsi_components: [
        {
          value: "natural_resources",
          label: "Natural Resources",
          description: "Climate sustainability and resilience (60%)",
          weight: 0.6,
        },
        {
          value: "availability",
          label: "Availability",
          description: "Food supply adequacy proxy (40%)",
          weight: 0.4,
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
          value: "max",
          label: "Maximum",
          description: "Maximum values",
        },
        {
          value: "min",
          label: "Minimum",
          description: "Minimum values",
        },
      ],
      fsi_classifications: [
        {
          value: "Sangat Tinggi",
          label: "Very High",
          range: "72.6+",
          color: "#22c55e",
          description: "Top producer regions (Aceh Utara - 330k tons)",
        },
        {
          value: "Tinggi",
          label: "High",
          range: "67.0 (Pidie)",
          color: "#84cc16",
          description: "High producer regions (Pidie - 220k tons)",
        },
        {
          value: "Sedang",
          label: "Medium",
          range: "67.6 (Aceh Besar)",
          color: "#f59e0b",
          description: "Medium producer regions (Aceh Besar - 182k tons)",
        },
        {
          value: "Rendah",
          label: "Low",
          range: "69.4 (Bireuen)",
          color: "#f97316",
          description: "Lower producer regions (Bireuen - 153k tons)",
        },
        {
          value: "Sangat Rendah",
          label: "Very Low",
          range: "<65.5",
          color: "#ef4444",
          description: "Lowest producer regions (Aceh Jaya - 50k tons)",
        },
      ],
    };

    return fsiParameters;
  } catch (error: any) {
    console.error("❌ FSI parameters fetch error:", error);
    if (error.response?.data?.message) {
      throw new Error(error.response.data.message);
    }
    throw new Error("Failed to fetch FSI parameters");
  }
};

export const exportFSIAnalysisCsv = async (
  analysisData: SpatialAnalysisResponse,
  filename?: string
) => {
  try {
    const csvData = analysisData.features.map((feature) => ({
      // Administrative Information
      kecamatan_name: feature.properties.NAME_3,
      kabupaten_name: feature.properties.NAME_2,
      province: feature.properties.NAME_1,

      // FSI Components
      fsi_score: feature.properties.fsi_score,
      fsi_classification: feature.properties.fsi_class,
      natural_resources_score: feature.properties.natural_resources_score,
      availability_score: feature.properties.availability_score,

      // Climate Suitability (legacy)
      suitability_score: feature.properties.suitability_score,
      classification: feature.properties.classification,
      confidence_level: feature.properties.confidence_level,

      // Component Scores
      temperature_score: feature.properties.temperature_score,
      precipitation_score: feature.properties.precipitation_score,
      humidity_score: feature.properties.humidity_score,

      // Climate Data
      avg_temperature: feature.properties.avg_temperature,
      avg_precipitation: feature.properties.avg_precipitation,
      avg_humidity: feature.properties.avg_humidity,

      // Risk & Location
      overall_risk: feature.properties.overall_risk,
      nasa_lat: feature.properties.nasa_lat,
      nasa_lng: feature.properties.nasa_lng,
      nasa_match: feature.properties.nasa_match,

      // Analysis Info
      analysis_timestamp: feature.properties.analysis_timestamp,
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

    const defaultFilename = `fsi_spatial_analysis_${
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
      message: `Successfully exported ${csvData.length} districts with FSI analysis`,
      filename: filename || defaultFilename,
    };
  } catch (error) {
    console.error("❌ Error exporting FSI analysis CSV:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Export failed",
    };
  }
};

export const getTwoLevelAnalysis = async (
  params: FSIAnalysisParams = {} //  Updated parameter type
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
        aggregation = "mean",
        districts = "all",
      } = params;

      const queryParams = new URLSearchParams({
        year_start: year_start.toString(),
        year_end: year_end.toString(),
        bps_start_year: bps_start_year.toString(),
        bps_end_year: bps_end_year.toString(),
        aggregation,
        districts,
      });

      console.log(
        `🔄 Attempt ${
          currentRetry + 1
        }/${maxRetries}: Fetching FSI two-level analysis: ${queryParams.toString()}`
      );

      const response = await axios.get(
        `${getFlaskBaseUrl()}/api/v1/two-level/analysis?${queryParams}`,
        {
          timeout: 300000,
          headers: {
            "Content-Type": "application/json",
          },
          onDownloadProgress: (progressEvent) => {
            if (progressEvent.lengthComputable && progressEvent.total) {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              console.log(`📥 Download progress: ${percentCompleted}%`);
            }
          },
        }
      );

      //  FIXED: Safe access to nested properties with fallback values
      const responseData = response.data;

      console.log(" FSI two-level analysis completed:", {
        kecamatan_count: responseData?.metadata?.level_1_kecamatan_count || 0,
        kabupaten_count: responseData?.metadata?.level_2_kabupaten_count || 0,
        avg_fsi: responseData?.summary_statistics?.average_fsi_score || 0,
        response_type: responseData?.type || "unknown",
      });

      return response.data;
    } catch (error: any) {
      currentRetry++;

      //  FIXED: Safe error message access
      const errorMessage =
        error?.message || error?.toString() || "Unknown error";
      console.error(`❌ Attempt ${currentRetry} failed:`, errorMessage);

      if (error.code === "ECONNABORTED" && currentRetry < maxRetries) {
        const delayMs = Math.pow(2, currentRetry) * 1000;
        console.log(`⏳ Retrying in ${delayMs / 1000}s...`);
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        continue;
      }

      if (error.code === "ECONNABORTED") {
        throw new Error(
          `FSI analysis timeout after ${maxRetries} attempts. The backend may be processing a large dataset.`
        );
      }

      if (error.code === "ECONNREFUSED") {
        throw new Error(
          "Flask backend service unavailable - please check if the server is running"
        );
      }

      if (error.response?.data?.message) {
        throw new Error(error.response.data.message);
      }

      throw new Error(
        `Failed to fetch FSI two-level analysis: ${errorMessage}`
      );
    }
  }

  throw new Error("Max retries exceeded");
};

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
      analysis_type: "food-security", // Enhanced spatial analysis with FSI
      analysis_level,
      include_bps_data: include_bps_data.toString(),
      bps_start_year: bps_start_year.toString(),
      bps_end_year: bps_end_year.toString(),
    });

    console.log("🌾📊 Fetching enhanced food security analysis...");

    const response = await axios.get(
      `${getFlaskBaseUrl()}/api/v1/spatial-map?${queryParams}`,
      {
        timeout: 90000, // Extended timeout for FSI calculations
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    console.log(" Food security analysis completed:", {
      features_count: response.data.features.length,
      analysis_type: "Enhanced with FSI components",
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
    const csvData = analysisData.level_2_kabupaten_analysis.data.map(
      (kabupaten) => ({
        kabupaten_name: kabupaten.kabupaten_name,
        bps_compatible_name: kabupaten.bps_compatible_name,

        // FSI Metrics
        fsi_score: kabupaten.aggregated_fsi_score,
        fsi_class: kabupaten.aggregated_fsi_class,
        natural_resources_score: kabupaten.aggregated_natural_resources_score,
        availability_score: kabupaten.aggregated_availability_score,

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

    const defaultFilename = `fsi_two_level_analysis_${
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
      message: `Successfully exported ${csvData.length} kabupaten FSI analysis`,
      filename: filename || defaultFilename,
    };
  } catch (error) {
    console.error("❌ Error exporting FSI two-level analysis CSV:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Export failed",
    };
  }
};

export const convertToSpatialParams = (
  params: FSIAnalysisParams
): SpatialAnalysisParams => {
  const { bps_start_year, bps_end_year, season, ...spatialParams } = params;

  // Convert aggregation enum
  let spatialAggregation: "mean" | "median" | "max" | "min" = "mean";
  if (params.aggregation === "median") spatialAggregation = "median";

  return {
    ...spatialParams,
    aggregation: spatialAggregation,
  };
};

export const convertToTwoLevelParams = (
  params: FSIAnalysisParams
): TwoLevelAnalysisParams => {
  const { analysis_level, include_bps_data, ...twoLevelParams } = params;
  return twoLevelParams;
};
