"use client";

import React, { useEffect, useState } from "react";
import { WeatherData } from "@/types/weather";
import DayDuration from "./day-duration";
import AirPollutionChart from "./air-pollution";
import TemperatureHumidityChart from "./temp-humidity";
// import ClientMap from "@/components/views/client-map";
import CurrentWeatherCard from "./current-weather";
import WindPressureCard from "./wind-pressure";
import HourlyForecast from "./hourly-forecast";
import { Banner } from "./banner";
import { WeatherTabs } from "./weather-tabs";
import { getBmkgApi } from "@/lib/fetch/files.fetch";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

interface WeatherDashboardProps {
  weatherData: WeatherData;
  unit: "metric" | "imperial";
}

const WeatherDashboard: React.FC<WeatherDashboardProps> = ({ weatherData, unit }) => {
  const { currentWeather, forecast, airPollution } = weatherData;
  const query = useQuery({
    queryKey: ["bmkg-api"],
    queryFn: getBmkgApi,
  });

  const bmkgData = query.data?.data; // akses isi dari response API
  console.log("BMKG raw response:", query.data);

  const [chartData, setChartData] = useState<{ time: string; temperature: number; humidity: number }[]>([]);

  useEffect(() => {
    if (!bmkgData) return;

    // Ambil bmkgData terbaru (asumsi per gampong, kita ambil 1 saja untuk chart)
    const today = new Date("2025-05-06T14:00:00");
    const end = new Date("2025-05-08T23:00:00");

    const selected = bmkgData.find((item) => item.kode_gampong === "11.06.02.2002");

    if (!selected || !selected.bmkgData) {
      console.error("Data tidak ditemukan atau bmkgData tidak tersedia.");
      return;
    }

    const filtered = selected.bmkgData.filter((item: any) => {
      const dt = new Date(item.local_datetime.replace(" ", "T"));
      return dt >= today && dt <= end;
    });

    const mapped = filtered.map((item: any) => ({
      time: new Date(item.local_datetime).getTime().toString(),
      temperature: item.t,
      humidity: item.hu,
    }));

    setChartData(mapped);
  }, [bmkgData]);

  console.log("bmkgData setelah filter", bmkgData);

  if (query.isLoading) return <div>Loading...</div>;
  if (query.error || !bmkgData) return <div>Error loading BMKG data.</div>;

  // Extract hourly forecast data for the first 5 items
  const hourlyForecastData = forecast.list.slice(0, 5).map((item) => ({
    time: new Date(item.dt * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
    temperature: Math.round(item.main.temp),
    weather: item.weather[0].main,
  }));

  return (
    <div className="bg-inherit min-h-screen flex flex-col">
      <Banner />
      <WeatherTabs defaultTab="weather">
        <div className="container grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
          <CurrentWeatherCard currentWeather={currentWeather} forecast={forecast} unit={unit} />
          <div className="grid grid-rows-2 gap-4">
            <WindPressureCard currentWeather={currentWeather} unit={unit} />
            <HourlyForecast forecast={hourlyForecastData} unit={unit} />
          </div>
          <AirPollutionChart data={airPollution} />
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
