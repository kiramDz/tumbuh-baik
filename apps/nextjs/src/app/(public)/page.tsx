"use client";

import { useState, useCallback, Suspense, lazy } from "react";
import Navbar from "@/components/navbar";
import WeatherDashboardSkeleton from "@/components/dashboard-skeleton";


// Lazy loaded component
const WeatherDashboard = lazy(() => import("@/components/view/weather-dashboard-new"));

export default function PublicPage() {
  const [unit] = useState<"metric" | "imperial">("metric");

  // Content renderer
  const renderContent = useCallback(() => {
    return (
      <Suspense fallback={<WeatherDashboardSkeleton />}>
        <WeatherDashboard unit={unit} />
      </Suspense>
    );
  }, [unit]);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      {renderContent()}
    </div>
  );
}
