// FSI Spatial Analysis Components (Updated from FSCI)
export { FSIMap } from "./FSIMap";
export { FSIFilters } from "./FSIFilters";
export { FSILegend } from "./FSILegend";

// Re-export types for convenience
export type {
  TwoLevelAnalysisParams,
  TwoLevelAnalysisResponse,
} from "@/lib/fetch/spatial.map.fetch";

// Component-specific prop types
export type { FSIFiltersProps } from "./FSIFilters";
export type { FSILegendProps } from "./FSILegend";

// FSI-specific utility types (Updated from FSCI)
export type FSIPerformanceLevel =
  | "sangat_tinggi"
  | "tinggi"
  | "sedang"
  | "rendah";

export type FSIStats = {
  total_regions: number;
  avg_fsi: number;
  sangat_tinggi_count: number;
  tinggi_count: number;
  sedang_count: number;
  rendah_count: number;
};

// FSI Component Types
export type FSIComponent = {
  name: "natural_resources" | "availability";
  label: string;
  description: string;
  weight: number;
  score: number;
};
