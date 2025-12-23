"use client";

import { useQuery } from "@tanstack/react-query";
import { getForecastConfigs } from "@/lib/fetch/files.fetch";
import { LabelList, Pie, PieChart } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";

export const description = "Pie charts with error metrics";

const COLORS = {
  mae: "#4ADE80",
  mape: "#F59E0B",
  rmse: "#EF4444",
};

const chartConfig = {
  mae: {
    label: "MAE",
    color: COLORS.mae,
  },
  mape: {
    label: "MAPE",
    color: COLORS.mape,
  },
  rmse: {
    label: "RMSE",
    color: COLORS.rmse,
  },
} satisfies ChartConfig;

export function RoundedPieChart() {
  const { data = [], isLoading } = useQuery({
    queryKey: ["forecast-config"],
    queryFn: getForecastConfigs,
  });

  if (isLoading) return <p>Loading pie charts...</p>;

  const completed = data.filter((item: any) => item.status === "done" && item.error_metrics?.length > 0);

  if (completed.length === 0) return <p>No completed forecasts with metrics available.</p>;

  const errorMetricsArray = completed[0]?.error_metrics ?? [];

  return (
    <div className="w-full grid md:grid-cols-2 gap-4">
      {errorMetricsArray.map((entry: any, index: number) => {
        const metrics = entry.metrics ?? {};
        const title = `${entry.collectionName} - ${entry.columnName}`;

        
        const chartData = [
          { key: "mae", value: metrics.mae || 0, fill: COLORS.mae },
          { key: "mape", value: metrics.mape || 0, fill: COLORS.mape },
          { key: "rmse", value: metrics.rmse || 0, fill: COLORS.rmse },
        ];

        return (
          <div key={index} className="flex flex-col rounded-2xl bg-background p-4 aspect-square max-h-[350px] relative">
            <div className="flex flex-col items-start pb-0">
              <h3 className="text-lg font-semibold flex items-center">{title}</h3>
            </div>

            <div className="flex gap-2">
              <div className="flex-1 pb-0">
                <ChartContainer config={chartConfig} className="[&_.recharts-text]:fill-background mx-auto aspect-square max-h-[250px]">
                  <PieChart>
                    <ChartTooltip content={<ChartTooltipContent nameKey="value" hideLabel />} />
                    <Pie data={chartData} dataKey="value" innerRadius={30} radius={12} cornerRadius={8} paddingAngle={4}>
                      <LabelList dataKey="value" stroke="none" fontSize={12} fontWeight={500} fill="currentColor" formatter={(value: number) => value.toFixed(1)} />
                    </Pie>
                  </PieChart>
                </ChartContainer>
              </div>

              {/* Custom Legend */}
              <div className="flex flex-col justify-center gap-2">
                {chartData.map((item) => (
                  <div key={item.key} className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded-sm" style={{ backgroundColor: item.fill }} />
                    <span className="text-sm text-muted-foreground uppercase">{item.key}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
