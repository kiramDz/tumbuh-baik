"use client";

import { PreprocessingReport } from "@/lib/fetch/files.fetch";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Icons } from "@/app/dashboard/_components/icons";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

interface MetricsChartsProps {
  report: PreprocessingReport;
}

interface ParameterCoverageData {
  parameter: string;
  hw_coverage: number;
  lstm_coverage: number;
  recommended_model?: string;
  seasonality_strength?: number;
  is_stationary?: boolean;
  hw_uncovered?: Record<string, number>;
  lstm_uncovered?: Record<string, number>;
}

const COLORS = {
  hw: "#729192",
  lstm: "#e67054",
  uncovered: "#2b4955",
};

const getModelBadgeVariant = (model?: string) => {
  if (!model) return "outline";
  if (model === "both") return "default";
  if (model === "lstm_with_caution") return "secondary";
  return "outline";
};

// Custom Tooltip Component
const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0];
  const isHW = data.payload.layer === "hw";
  const model = isHW ? "Holt-Winters" : "LSTM";
  const isCovered = data.name.includes("Covered");

  return (
    <div className="bg-background border rounded-lg shadow-lg p-3 min-w-[200px]">
      <div className="font-semibold text-sm mb-2 flex items-center gap-2">
        <div
          className="h-3 w-3 rounded-sm"
          style={{ backgroundColor: data.fill }}
        />
        {model}
      </div>
      <div className="text-xs text-muted-foreground space-y-1">
        <div>
          <span className="font-medium">
            {isCovered ? "Coverage" : "Uncovered"}:
          </span>{" "}
          {data.value.toFixed(2)}%
        </div>
        {!isCovered && data.payload.uncovered_details && (
          <div className="mt-2 pt-2 border-t space-y-0.5">
            <div className="font-medium text-foreground mb-1">Issues:</div>
            {Object.entries(data.payload.uncovered_details).map(
              ([key, val]: [string, any]) => (
                <div key={key} className="flex justify-between gap-2">
                  <span className="capitalize">{key.replace(/_/g, " ")}:</span>
                  <span className="font-medium">{Number(val).toFixed(1)}%</span>
                </div>
              ),
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export function MetricsCharts({ report }: MetricsChartsProps) {
  const isNasa = report.dataset_type === "nasa";

  // Map Parameter-specific Coverage Data
  const rawParams = report.model_coverage?.per_parameter || {};
  const parameterData: ParameterCoverageData[] = Object.entries(rawParams)
    .map(([key, data]: [string, any]) => ({
      parameter: key,
      hw_coverage: data.holt_winters_coverage || 0,
      lstm_coverage: data.lstm_coverage || 0,
      recommended_model: data.recommended_model,
      seasonality_strength: data.seasonality_strength,
      is_stationary: data.is_stationary,
      hw_uncovered: data.holt_winters_uncovered || {},
      lstm_uncovered: data.lstm_uncovered || {},
    }))
    .filter((param) => param.hw_coverage > 0 || param.lstm_coverage > 0);

  return (
    <div className="space-y-6">
      {/* Signal & Trend Preservation Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Signal & Trend Preservation</CardTitle>
          <CardDescription>
            Trend retained post-cleansing (placeholder)
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[280px] flex items-center justify-center text-sm text-muted-foreground">
          Chart will be implemented here
        </CardContent>
      </Card>

      {/* PER-PARAMETER COVERAGE - Nested Donut Charts */}
      <div>
        <h3 className="text-xl font-semibold tracking-tight mb-4">
          Model Coverage by Parameter
        </h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {parameterData.map((param) => {
            // Outer ring: Holt-Winters
            const hwData = [
              {
                name: "HW Covered",
                value: param.hw_coverage,
                layer: "hw",
              },
              {
                name: "HW Uncovered",
                value: 100 - param.hw_coverage,
                layer: "hw",
                uncovered_details: param.hw_uncovered,
              },
            ];

            // Inner ring: LSTM
            const lstmData = [
              {
                name: "LSTM Covered",
                value: param.lstm_coverage,
                layer: "lstm",
              },
              {
                name: "LSTM Uncovered",
                value: 100 - param.lstm_coverage,
                layer: "lstm",
                uncovered_details: param.lstm_uncovered,
              },
            ];

            return (
              <Card key={param.parameter} className="flex flex-col">
                <CardHeader className="pb-3 flex-row items-start justify-between space-y-0">
                  <CardTitle className="text-lg font-semibold">
                    {param.parameter}
                  </CardTitle>
                  {param.recommended_model && (
                    <Badge
                      variant={getModelBadgeVariant(param.recommended_model)}
                      className="text-xs"
                    >
                      {param.recommended_model === "both"
                        ? "⭐ Both"
                        : param.recommended_model === "lstm_with_caution"
                          ? "⚠️ LSTM"
                          : param.recommended_model}
                    </Badge>
                  )}
                </CardHeader>

                <CardContent className="flex-1 space-y-4">
                  {/* Nested Donut Chart */}
                  <div className="h-[200px] relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        {/* Outer Ring - Holt-Winters */}
                        <Pie
                          data={hwData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={85}
                          cornerRadius={6}
                          paddingAngle={4}
                          dataKey="value"
                          stroke="none"
                        >
                          <Cell fill={COLORS.hw} />
                          <Cell fill={COLORS.uncovered} />
                        </Pie>

                        {/* Inner Ring - LSTM */}
                        <Pie
                          data={lstmData}
                          cx="50%"
                          cy="50%"
                          innerRadius={35}
                          outerRadius={55}
                          cornerRadius={6}
                          paddingAngle={4}
                          dataKey="value"
                          stroke="none"
                        >
                          <Cell fill={COLORS.lstm} />
                          <Cell fill={COLORS.uncovered} />
                        </Pie>

                        <Tooltip content={<CustomTooltip />} />
                      </PieChart>
                    </ResponsiveContainer>

                    {/* Center Label */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="text-center">
                        <div className="text-xs text-muted-foreground font-medium">
                          Coverage
                        </div>
                        <div className="text-lg font-bold">
                          {Math.max(
                            param.hw_coverage,
                            param.lstm_coverage,
                          ).toFixed(0)}
                          %
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Legend */}
                  <div className="flex justify-center gap-4 text-xs border-t pt-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-3 w-3 rounded-sm"
                        style={{ backgroundColor: COLORS.hw }}
                      />
                      <span className="font-medium">
                        HW {param.hw_coverage.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div
                        className="h-3 w-3 rounded-sm"
                        style={{ backgroundColor: COLORS.lstm }}
                      />
                      <span className="font-medium">
                        LSTM {param.lstm_coverage.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Characteristics */}
                  {(param.is_stationary !== undefined ||
                    param.seasonality_strength !== undefined) && (
                    <div className="flex justify-between text-xs text-muted-foreground border-t pt-3">
                      {param.seasonality_strength !== undefined && (
                        <span className="flex items-center gap-1.5">
                          <Icons.waves className="w-3.5 h-3.5" />
                          Seasonality: {param.seasonality_strength.toFixed(2)}
                        </span>
                      )}
                      {param.is_stationary !== undefined && (
                        <span className="flex items-center gap-1.5">
                          <Icons.activity className="w-3.5 h-3.5" />
                          {param.is_stationary
                            ? "✓ Stationary"
                            : "✗ Non-stationary"}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Uncovered Summary (Compact) */}
                  {(Object.keys(param.hw_uncovered || {}).length > 0 ||
                    Object.keys(param.lstm_uncovered || {}).length > 0) && (
                    <div className="space-y-2 text-xs border-t pt-3">
                      <div className="font-semibold text-muted-foreground mb-1">
                        Top Issues:
                      </div>
                      {Object.keys(param.hw_uncovered || {}).length > 0 && (
                        <div className="flex items-start gap-2">
                          <div
                            className="h-2 w-2 rounded-full mt-1 shrink-0"
                            style={{ backgroundColor: COLORS.hw }}
                          />
                          <div className="flex-1 leading-relaxed">
                            {Object.entries(param.hw_uncovered || {})
                              .slice(0, 2)
                              .map(([k, v]) => (
                                <span key={k} className="inline-block mr-2">
                                  {k.replace(/_/g, " ")} ({Number(v).toFixed(1)}
                                  %)
                                </span>
                              ))}
                          </div>
                        </div>
                      )}
                      {Object.keys(param.lstm_uncovered || {}).length > 0 && (
                        <div className="flex items-start gap-2">
                          <div
                            className="h-2 w-2 rounded-full mt-1 shrink-0"
                            style={{ backgroundColor: COLORS.lstm }}
                          />
                          <div className="flex-1 leading-relaxed">
                            {Object.entries(param.lstm_uncovered || {})
                              .slice(0, 2)
                              .map(([k, v]) => (
                                <span key={k} className="inline-block mr-2">
                                  {k.replace(/_/g, " ")} ({Number(v).toFixed(1)}
                                  %)
                                </span>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}
