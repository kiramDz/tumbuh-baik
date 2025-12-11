/**
 * Two-Level Food Security Analysis Components
 *
 * This module provides a comprehensive set of components for analyzing
 * and visualizing the relationship between climate potential (FSCI) and
 * actual production data across two administrative levels:
 * - Level 1: Kecamatan (District level)
 * - Level 2: Kabupaten (Regency level)
 */

// Main Components
export { TwoLevelMap } from "./TwoLevelMap";
export { TwoLevelFilters } from "./TwoLevelFilters";
export { KabupatenDetails } from "./KabupatenDetails";
export { KecamatanList } from "./KecamatanList";

export type { TwoLevelMapProps } from "./TwoLevelMap";
export type { TwoLevelFiltersProps } from "./TwoLevelFilters";
export type { KabupatenDetailsProps } from "./KabupatenDetails";
export type { KecamatanListProps } from "./KecamatanList";

// Re-export hook for convenience
export { useTwoLevelAnalysis } from "@/hooks/use-twoLevelAnalysis";

// Re-export key types from fetch utilities
export type {
  TwoLevelAnalysisParams,
  TwoLevelAnalysisResponse,
  KabupatenAnalysis,
  KecamatanAnalysis,
  CrossLevelInsights,
  PolicyRecommendation,
  SummaryStatistics,
} from "@/lib/fetch/spatial.map.fetch";

/**
 * Usage Example:
 *
 * ```tsx
 * import {
 *   TwoLevelMap,
 *   TwoLevelFilters,
 *   KabupatenDetails,
 *   KecamatanList,
 *   useTwoLevelAnalysis
 * } from "@/components/two-level";
 *
 * function TwoLevelDashboard() {
 *   const { selectedKabupaten, selectKabupaten } = useTwoLevelAnalysis();
 *
 *   return (
 *     <div className="grid grid-cols-12 gap-6">
 *       <TwoLevelFilters className="col-span-3" />
 *       <TwoLevelMap
 *         className="col-span-6"
 *         onKabupatenSelect={selectKabupaten}
 *       />
 *       <div className="col-span-3 space-y-6">
 *         <KabupatenDetails kabupatenName={selectedKabupaten} />
 *         <KecamatanList kabupatenFilter={selectedKabupaten} />
 *       </div>
 *     </div>
 *   );
 * }
 * ```
 */

/**
 * Component Architecture:
 *
 * ┌─────────────────────┐
 * │   TwoLevelFilters   │  ← Parameter management & analysis configuration
 * └─────────────────────┘
 *           │
 *           ▼
 * ┌─────────────────────┐
 * │    TwoLevelMap      │  ← Interactive geospatial visualization
 * └─────────────────────┘
 *           │
 *           ├─────────────────────┐
 *           ▼                     ▼
 * ┌─────────────────────┐  ┌─────────────────────┐
 * │  KabupatenDetails   │  │   KecamatanList     │
 * └─────────────────────┘  └─────────────────────┘
 *                                    │
 *                                    ▼
 *                          [Individual Kecamatan Details]
 *
 * Data Flow:
 * - useTwoLevelAnalysis hook provides centralized state management
 * - TwoLevelFilters controls analysis parameters
 * - TwoLevelMap visualizes spatial data + statistics
 * - KabupatenDetails shows aggregated insights
 * - KecamatanList provides detailed breakdowns
 */
