"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { useQuery } from "@tanstack/react-query";
import { getForecastConfigs } from "@/lib/fetch/files.fetch";

export function ForecastConfigList() {
  const { data = [], isLoading } = useQuery({
    queryKey: ["forecast-config"],
    queryFn: getForecastConfigs,
  });

  if (isLoading) return <p>Loading...</p>;

  return (
    <ScrollArea className="w-full rounded-md border whitespace-nowrap">
      <div className="flex gap-4 p-4 w-max">
        {data.map((item: any) => (
          <Card key={item._id} className="w-[300px] shrink-0 overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-md">
                {getStatusIcon(item.status)} {item.name}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <p className="mb-2">Kolom: {item.columns?.map((col) => `${col.columnName}`).join(", ")}</p>
              <p className="text-xs text-destructive">{item.errorMessage && `Error: ${item.errorMessage}`}</p>
            </CardContent>
          </Card>
        ))}
      </div>
      <ScrollBar orientation="horizontal" />
    </ScrollArea>
  );
}

// STATUS ICON HANDLER (gunakan kode svg milikmu sendiri)
const getStatusIcon = (status: any) => {
  switch (status) {
    case "done":
      return (
        <svg width="16" height="16" viewBox="0 0 16 16" className="drop-shadow-sm">
          <circle cx="8" cy="8" r="8" fill="#22c55e" />
          <path d="M5 8l2.5 2.5 3.5-4" stroke="white" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "failed":
      return (
        <svg width="16" height="16" viewBox="0 0 16 16">
          <path d="M8 1.5L14.5 13H1.5L8 1.5Z" fill="#eab308" stroke="#eab308" strokeWidth="1" strokeLinejoin="round" />
          <path d="M8 6v3M8 11h0" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      );
    case "pending":
      return <span className="text-yellow-500">…</span>;
    case "running":
      return (
        <svg width="16" height="16" viewBox="0 0 16 16">
          {/* Create 8 dashes around the circle */}
          {Array.from({ length: 8 }).map((_, index) => {
            const angle = index * 45 - 90; // Start from top, -90 to offset
            const radian = (angle * Math.PI) / 180;
            const radius = 6;
            const dashLength = 1.8;

            // Calculate start and end points for each dash
            const startX = 8 + (radius - dashLength / 2) * Math.cos(radian);
            const startY = 8 + (radius - dashLength / 2) * Math.sin(radian);
            const endX = 8 + (radius + dashLength / 2) * Math.cos(radian);
            const endY = 8 + (radius + dashLength / 2) * Math.sin(radian);

            return <line key={index} x1={startX} y1={startY} x2={endX} y2={endY} stroke="#6b7280" strokeWidth="2" strokeLinecap="round" />;
          })}
        </svg>
      );
    default:
      return <span>?</span>;
  }
};
