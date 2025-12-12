// FSCI Spatial Analysis Components
export { FSCIMap } from "./FSCIMap";
export { FSCIFilters } from "./FSCIFilters";
export { FSCILegend } from "./FSCILegend";
export { FSCIMetadataPanel } from "./FSCIMetadataPanel";

// Re-export types for convenience
export type {
  TwoLevelAnalysisParams,
  TwoLevelAnalysisResponse,
} from "@/lib/fetch/spatial.map.fetch";

// Component-specific prop types
export type { FSCIFiltersProps } from "./FSCIFilters";
export type { FSCILegendProps } from "./FSCILegend";
export type { FSCIMetadataPanelProps } from "./FSCIMetadataPanel";

// FSCI-specific utility types
export type FSCIPerformanceLevel = "excellent" | "good" | "fair" | "poor";

export type FSCIStats = {
  total_regions: number;
  avg_fsci: number;
  excellent_count: number;
  good_count: number;
  fair_count: number;
  poor_count: number;
};
