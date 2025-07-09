"use client";

import React, { useEffect, useState } from "react";
import { WeatherData } from "@/types/weather";
import DayDuration from "./day-duration";
import WeatherConclusion from "./weather-conclusion";
import TemperatureHumidityChart from "./temp-humidity";
import CurrentWeatherCard from "./current-weather";
import WindPressureCard from "./wind-pressure";
import HourlyForecast from "./hourly-forecast";
import { Banner } from "./banner";
import { WeatherHeader } from "./weather-header";
import { WeatherTabs } from "./weather-tabs";
import { getBmkgApi, getBmkgSummary } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { getTodayWeather, getChartData, getHourlyForecastData, getWeatherConclusionFromDailyData, getTodayWeatherConlusion, getFinalConclusion } from "@/lib/bmkg-utils";
import type { BMKGApiData, PlantSummaryData } from "@/types/table-schema";

interface ChartDataPoint {
  time: string;
  temperature: number;
  humidity: number;
}

interface WeatherDashboardProps {
  weatherData: WeatherData;
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ weatherData, unit }) => {
  const { currentWeather } = weatherData;
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [selectedGampong, setSelectedGampong] = useState<string | null>(null);
  const now = new Date();
  const end = new Date();

  const { data: bmkgApiResponse } = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgApi,
  });

  const { data: bmkgSummary } = useQuery({
    queryKey: ["bmkg-summary"],
    queryFn: getBmkgSummary,
  });

  const today = new Date();
  const currentMonth = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}`;

  const currentSummary = bmkgSummary?.find((item: PlantSummaryData) => item.month === currentMonth);

  const bmkgData = bmkgApiResponse?.data;
  const selected = bmkgData?.find((item: BMKGApiData) => item.kode_gampong === selectedGampong);
  const todayOnlyData = getTodayWeatherConlusion(selected?.data || []);
  const conclusion = getWeatherConclusionFromDailyData(todayOnlyData);
  const finalConclusion = getFinalConclusion(conclusion, currentSummary?.status ?? "tidak cocok tanam");

  const latestData = selected?.data ? getTodayWeather(selected.data) : null;

  useEffect(() => {
    if (bmkgData && !selectedGampong) {
      setSelectedGampong(bmkgData[0]?.kode_gampong);
    }
  }, [bmkgData, selectedGampong]);

  useEffect(() => {
    if (!selectedGampong || !bmkgData) return;
    const selected = bmkgData.find((item: any) => item.kode_gampong === selectedGampong);
    if (!selected?.data) return;
    const mapped = getChartData(selected.data);
    setChartData(mapped);
  }, [bmkgData, selectedGampong]);

  if (!bmkgApiResponse || !bmkgSummary) return <div>Loading...</div>;

  end.setDate(now.getDate() + 1);
  end.setHours(23, 59, 59);

  const forecastData = selected?.data ? getHourlyForecastData(selected.data) : [];

  return (
    <div className="bg-inherit min-h-screen flex flex-col">
      <Banner />
      <WeatherTabs defaultTab="weather">
        {selectedGampong && <WeatherHeader bmkgData={bmkgData} selectedCode={selectedGampong} onGampongChange={setSelectedGampong} />}
        <div className=" grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 py-4">
          {latestData && selected && <CurrentWeatherCard bmkgCurrent={{ ...latestData, nama_gampong: selected.nama_gampong }} unit={unit} />}

          <div className="grid grid-rows-2 gap-4">
            {latestData && <WindPressureCard bmkgCurrent={latestData} unit={unit} />}

            <HourlyForecast forecast={forecastData} unit={unit} />
          </div>
          <WeatherConclusion conclusion={finalConclusion} tcc={latestData?.t ?? 0} />

          {chartData.length === 0 ? (
            <p className="text-sm text-muted-foreground">Tidak ada data suhu/kelembapan untuk gampong ini dalam rentang waktu tersebut.</p>
          ) : (
            <TemperatureHumidityChart
              data={{
                list: chartData.map((item) => ({
                  dt: parseInt(item.time) / 1000,
                  main: {
                    temp: item.temperature,
                    humidity: item.humidity,
                  },
                })),
              }}
              unit="metric"
            />
          )}

          <DayDuration data={currentWeather} />
        </div>
      </WeatherTabs>
    </div>
  );
};

export default WeatherDashboard;
