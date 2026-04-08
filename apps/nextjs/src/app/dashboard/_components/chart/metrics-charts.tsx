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
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

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

  // Resolve validation metrics scatter plot
  const validationObj = isNasa
    ? report.smoothing_validation
    : report.imputation_validation;

  const validationData = validationObj
    ? Object.entries(validationObj).map(([key, val]) => ({
        parameter: key,
        trendPreservation: val.trend_preservation_pct || 0,
        gcvScore: val.gcv_score || 0,
        quality_status: val.quality_status || "unknown",
        size:
          val.quality_status === "excellent"
            ? 400
            : val.quality_status === "good"
              ? 250
              : 150,
      }))
    : [];

  const getQualityColor = (status: string) => {
    if (status === "excellent") return "hsl(var(--chart-2))";
    if (status === "good") return "hsl(var(--chart-1))";
    if (status === "fair") return "hsl(var(--chart-3))";
    return "hsl(var(--muted-foreground))";
  };

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
      {/* Signal & Trend Preservation */}
      <Card>
        <CardHeader>
          <CardTitle>Signal & Trend Preservation</CardTitle>
          <CardDescription>
            GCV vs Trend Retained (Lower GCV & Higher Trend is better)
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[320px] w-full">
          {validationData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="gcvScore"
                  name="GCV Score"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(val) => val.toFixed(2)}
                  domain={[0, "auto"]}
                />
                <YAxis
                  type="number"
                  dataKey="trendPreservation"
                  name="Trend Preservation"
                  domain={["auto", 100]}
                  tick={{ fontSize: 12 }}
                  tickFormatter={(val) => `${val}%`}
                />
                <ZAxis type="number" dataKey="size" range={[100, 400]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-background border rounded-lg shadow-lg p-3 min-w-[150px]">
                          <p className="font-semibold text-sm">
                            {data.parameter}
                          </p>
                          <p className="text-xs mt-1">
                            GCV:{" "}
                            <span className="font-medium">
                              {data.gcvScore.toFixed(4)}
                            </span>
                          </p>
                          <p className="text-xs">
                            Trend:{" "}
                            <span className="font-medium">
                              {data.trendPreservation.toFixed(2)}%
                            </span>
                          </p>
                          <p className="text-xs uppercase text-muted-foreground mt-1">
                            {data.quality_status}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter name="Parameters" data={validationData}>
                  {validationData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={getQualityColor(entry.quality_status)}
                    />
                  ))}
                </Scatter>
                <ReferenceLine
                  y={90}
                  stroke="hsl(var(--muted-foreground))"
                  strokeDasharray="3 3"
                  label={{
                    value: "Excellent",
                    position: "insideTopLeft",
                    fontSize: 11,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              No validation metrics available
            </div>
          )}
        </CardContent>
      </Card>

      {/* PER-PARAMETER COVERAGE - Nested Donut Charts */}
      <div>
        <h3 className="text-xl font-semibold tracking-tight mb-4">
          Model Coverage Tiap Parameter
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
