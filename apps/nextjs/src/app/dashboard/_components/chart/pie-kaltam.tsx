"use client";

import { useQuery } from "@tanstack/react-query";
import { getForecastConfigs } from "@/lib/fetch/files.fetch";
import { LabelList, Pie, PieChart } from "recharts";
import { ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";


export const description = "A pie chart with a label list";

const COLORS = {
  mae: "#4ADE80", // green
  rmse: "#60A5FA", // blue
  mape: "#F59E0B", // amber
  mse: "#EF4444", // red
};

const chartConfig = {
  mae: {
    label: "MAE",
    color: COLORS.mae,
  },
  rmse: {
    label: "RMSE",
    color: COLORS.rmse,
  },
  mape: {
    label: "MAPE",
    color: COLORS.mape,
  },
  mse: {
    label: "MSE",
    color: COLORS.mse,
  },
} satisfies ChartConfig;

export function RoundedPieChart() {
  const { data = [], isLoading } = useQuery({
    queryKey: ["forecast-config"],
    queryFn: getForecastConfigs,
  });

  if (isLoading) return <p>Loading pie chart...</p>;

  // Ambil hanya yang statusnya "done" dan punya error_metrics
  const completed = data.filter((item: any) => item.status === "done" && item.error_metrics?.length > 0);

  // Ambil metrik dari config pertama saja (bisa diubah sesuai kebutuhan)
  const metrics = completed[0]?.error_metrics[0]?.metrics ?? {};

  // Bentuk data untuk pie chart
  const chartData = [
    { key: "mae", value: metrics.mae, fill: COLORS.mae },
    { key: "rmse", value: metrics.rmse, fill: COLORS.rmse },
    { key: "mape", value: metrics.mape, fill: COLORS.mape },
    { key: "mse", value: metrics.mse, fill: COLORS.mse },
  ];

  return (
    <div className="flex flex-col rounded-2xl bg-background p-4 aspect-square max-h-[350px] relative w-1/3">
      <div className="flex flex-col items-start pb-0">
        <h3 className="text-lg font-semibold flex items-center">Error Metrics</h3>
      </div>

      <div className="flex gap-2">
        <div className="flex-1 pb-0">
          <ChartContainer config={chartConfig} className="[&_.recharts-text]:fill-background mx-auto aspect-square max-h-[350px]">
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
}
