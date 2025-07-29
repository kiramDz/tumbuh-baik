"use client";

import { useState, useCallback, Suspense, lazy } from "react";
import Navbar from "@/components/navbar";
import WeatherDashboardSkeleton from "@/components/dashboard-skeleton";

import { useWeatherData } from "@/hooks/use-weatherData";

// Lazy loaded component
const WeatherDashboard = lazy(() => import("@/components/view/weather-dashboard-new"));

export default function PublicPage() {
  const [coordinates] = useState({ lat: 0, lon: 0 });
  const [unit] = useState<"metric" | "imperial">("metric");

  const { weatherData, error, isLoading } = useWeatherData(coordinates);

  // Content renderer
  const renderContent = useCallback(() => {
    if (error) {
      return (
        <div className="flex items-center justify-center flex-1" role="alert" aria-live="assertive">
          <p className="text-red-500">{error}</p>
        </div>
      );
    }

    if (!weatherData || isLoading) {
      return <WeatherDashboardSkeleton />;
    }

    return (
      <Suspense fallback={<WeatherDashboardSkeleton />}>
        <WeatherDashboard unit={unit} />
      </Suspense>
    );
  }, [error, weatherData, isLoading, unit]);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      {renderContent()}
    </div>
  );
}
