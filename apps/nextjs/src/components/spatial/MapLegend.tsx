"use client";

import { LEGEND_DATA } from "@/lib/spatial-map.utils";

interface MapLegendProps {
  className?: string;
  title?: string;
  showTitle?: boolean;
}

export function MapLegend({
  className = "",
  title = "Rice Suitability Score",
  showTitle = true,
}: MapLegendProps) {
  return (
    <div className={`space-y-3 ${className}`}>
      {showTitle && (
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      )}

      <div className="space-y-2">
        {LEGEND_DATA.map((item, index) => (
          <div key={index} className="flex items-center space-x-3">
            <div
              className="w-4 h-4 rounded border border-gray-300"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-sm text-gray-700 font-medium">
              {item.label}
            </span>
          </div>
        ))}
      </div>

      {/* Additional info */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg border">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Legend Info</h4>
        <ul className="text-xs text-gray-600 space-y-1">
          <li>• Higher scores indicate better rice growing conditions</li>
          <li>• Scores based on temperature, precipitation & humidity</li>
          <li>• Click districts on map for detailed information</li>
        </ul>
      </div>
    </div>
  );
}
