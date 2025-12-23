"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { WeatherLineChart } from "./weather-line-chart";

interface WeatherChartTabsProps {
  hourlyForecast: {
    time: string;
    temperature: number;
    cloud_cover?: number;
    wind_speed?: number;
  }[];
}

export function WeatherChartTabs({ hourlyForecast }: WeatherChartTabsProps) {
  // Bentuk dataset terpisah untuk setiap chart
  const tempData = hourlyForecast.map((d) => ({
    time: d.time,
    value: d.temperature,
  }));

  const rainData = hourlyForecast.map((d) => ({
    time: d.time,
    value: d.cloud_cover ?? 0,
  }));

  const windData = hourlyForecast.map((d) => ({
    time: d.time,
    value: d.wind_speed ?? 0,
  }));

  return (
    <Tabs defaultValue="temperature" className="w-full mt-4 p-0">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="temperature">Temperature</TabsTrigger>
        <TabsTrigger value="cloud_cover">Tutupan Awan</TabsTrigger>
        <TabsTrigger value="wind">Wind Speed</TabsTrigger>
      </TabsList>

      <TabsContent className="p-0" value="temperature">
        <WeatherLineChart data={tempData} type="temperature" />
      </TabsContent>

      <TabsContent value="cloud_cover">
        <WeatherLineChart data={rainData} type="cloud_cover" />
      </TabsContent>

      <TabsContent value="wind">
        <WeatherLineChart data={windData} type="wind" />
      </TabsContent>
    </Tabs>
  );
}
