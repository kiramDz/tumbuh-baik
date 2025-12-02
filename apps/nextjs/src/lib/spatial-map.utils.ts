/**
 * Utility functions for spatial analysis map (Leaflet Choropleth)
 */

// ─────────────────────────────────────────────
// Color scheme for rice suitability choropleth
// ─────────────────────────────────────────────

export const getSuitabilityColor = (score: number): string => {
  if (score >= 85) return "#1a5f1a"; // Excellent
  if (score >= 70) return "#52b352"; // Good
  if (score >= 55) return "#ffeb3b"; // Fair
  if (score >= 40) return "#ff9800"; // Marginal
  return "#f44336"; // Poor
};

// ─────────────────────────────────────────────
// Classification label
// ─────────────────────────────────────────────

export const getSuitabilityClass = (score: number): string => {
  if (score >= 85) return "Excellent";
  if (score >= 70) return "Good";
  if (score >= 55) return "Fair";
  if (score >= 40) return "Marginal";
  return "Poor";
};

// ─────────────────────────────────────────────
// Legend static data (used in UI Legend component)
// ─────────────────────────────────────────────

export const LEGEND_DATA = [
  { min: 85, max: 100, color: "#1a5f1a", label: "Excellent (85–100)" },
  { min: 70, max: 84, color: "#52b352", label: "Good (70–84)" },
  { min: 55, max: 69, color: "#ffeb3b", label: "Fair (55–69)" },
  { min: 40, max: 54, color: "#ff9800", label: "Marginal (40–54)" },
  { min: 0, max: 39, color: "#f44336", label: "Poor (0–39)" },
];

// ─────────────────────────────────────────────
// Aceh bounding box (used for auto-fit map)
// ─────────────────────────────────────────────

export const ACEH_BOUNDS = {
  north: 6.5,
  south: 2.0,
  east: 98.5,
  west: 94.5,
};

// ─────────────────────────────────────────────
// Leaflet GeoJSON style function
// ─────────────────────────────────────────────

export const getFeatureStyle = (feature: any) => {
  const score = feature?.properties?.suitability_score ?? 0;

  return {
    fillColor: getSuitabilityColor(score),
    weight: 2,
    opacity: 1,
    color: "#ffffff",
    dashArray: "3",
    fillOpacity: 0.8,
  };
};

// ─────────────────────────────────────────────
// Highlight style on hover
// ─────────────────────────────────────────────

export const getHighlightStyle = {
  weight: 5,
  color: "#666",
  dashArray: "",
  fillOpacity: 0.9,
};
