// Import chart components first
import { CorrelationScatter } from "./CorrelationScatter";
import { EfficiencyMatrix } from "./EfficiencyMatrix";
import { FSCIComponents } from "./FSCIComponents";
import { TimeSeriesChart } from "./TimeSeriesChart";

// Chart Components exports
export { CorrelationScatter };
export { EfficiencyMatrix };
export { FSCIComponents };
export { TimeSeriesChart };

// Chart Component Props Types
export type { CorrelationScatterProps } from "./CorrelationScatter";
export type { EfficiencyMatrixProps } from "./EfficiencyMatrix";
export type { FSCIComponentsProps } from "./FSCIComponents";
export type { TimeSeriesChartProps } from "./TimeSeriesChart";

// Re-export common types for convenience
export type {
  TwoLevelAnalysisParams,
  TwoLevelAnalysisResponse,
  KabupatenAnalysis,
  KecamatanAnalysis,
} from "@/lib/fetch/spatial.map.fetch";

// Chart utility types
export interface ChartDataPoint {
  x: number | string;
  y: number;
  label?: string;
  color?: string;
}

export interface ChartConfig {
  width?: number;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  responsive?: boolean;
}

// Chart collection for easy bulk import (now with proper imports)
export const Charts = {
  CorrelationScatter,
  EfficiencyMatrix,
  FSCIComponents,
  TimeSeriesChart,
} as const;

// Chart types for the collection
export type ChartComponent = (typeof Charts)[keyof typeof Charts];

// Utility to get all chart names
export const getChartNames = () =>
  Object.keys(Charts) as Array<keyof typeof Charts>;

// Chart metadata for dynamic usage
export const chartMetadata = {
  CorrelationScatter: {
    name: "Correlation Scatter",
    description: "Climate vs Production correlation analysis",
    category: "analysis",
  },
  EfficiencyMatrix: {
    name: "Efficiency Matrix",
    description: "Production efficiency performance analysis",
    category: "performance",
  },
  FSCIComponents: {
    name: "FSCI Components",
    description: "Food Security Climate Index breakdown",
    category: "breakdown",
  },
  TimeSeriesChart: {
    name: "Time Series",
    description: "Trends over time analysis",
    category: "trends",
  },
} as const;
