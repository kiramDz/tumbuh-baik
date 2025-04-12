"use client";

//cpde : https://github.com/manju1807/Weatherly-nextjs-weather-app/blob/main/src/app/page.tsx

import { useGeolocation } from "@/hooks/use-geolocation";
import { useEffect, useState, useCallback, Suspense, lazy } from "react";
import { DEFAULT_COORDINATES } from "@/constant/index";
import Navbar from "@/components/navbar";
import WeatherDashboardSkeleton from "@/components/dashboard-skeleton";
import { ErrorBoundary } from "react-error-boundary";
import LocationPermissionDialog from "@/components/location-dialog";
import { useWeatherData } from "@/hooks/use-weatherData";
import { useDebounceWithComparison } from "@/hooks/use-DebounceWithComparison";
import { useCoordinatesComparison } from "@/hooks/use-coordinatesComparison";
import { ErrorFallback } from "@/components/error-fallback";

// Lazy loaded component
const WeatherDashboard = lazy(() => import("@/components/view/weather-dashboard"));

export default function PublicPage() {
  const [coordinates, setCoordinates] = useState(DEFAULT_COORDINATES);
  const [isManualSelection, setIsManualSelection] = useState(false);
  const [unit] = useState<"metric" | "imperial">("metric");
  const [showLocationDialog, setShowLocationDialog] = useState(true);
  const [geolocationEnabled, setGeolocationEnabled] = useState(false);
  const [hasInitialLoad, setHasInitialLoad] = useState(false);

  const compareCoordinates = useCoordinatesComparison();
  const debouncedCoordinates = useDebounceWithComparison(coordinates, 500, compareCoordinates);

  const {
    latitude,
    longitude,
    loading: locationLoading,
    hasPermission,
  } = useGeolocation({
    timeout: 1000,
    enabled: geolocationEnabled,
  });

  const { weatherData, error, isLoading, setError } = useWeatherData(debouncedCoordinates, locationLoading, hasInitialLoad);

  // Location handlers
  const handleLocationChange = useCallback(
    (lat: number, lon: number) => {
      const newCoords = { lat, lon };
      if (!compareCoordinates(coordinates, newCoords)) {
        setIsManualSelection(true);
        setGeolocationEnabled(false);
        setCoordinates(newCoords);
        setHasInitialLoad(true);
      }
    },
    [coordinates, compareCoordinates]
  );

  const handleUseCurrentLocation = useCallback(() => {
    setIsManualSelection(false);
    setGeolocationEnabled(true);
  }, []);

  const handleLocationPermission = useCallback((allow: boolean) => {
    setShowLocationDialog(false);
    setGeolocationEnabled(allow);
    setIsManualSelection(!allow);
    if (!allow) {
      setCoordinates(DEFAULT_COORDINATES);
    }
    setHasInitialLoad(true);
  }, []);

  // Update coordinates based on geolocation
  useEffect(() => {
    if (geolocationEnabled && hasPermission && !isManualSelection && !compareCoordinates(coordinates, { lat: latitude, lon: longitude })) {
      setCoordinates({ lat: latitude, lon: longitude });
      setHasInitialLoad(true);
    }
  }, [geolocationEnabled, hasPermission, isManualSelection, latitude, longitude, coordinates, compareCoordinates]);

  // Content renderer
  const renderContent = useCallback(() => {
    if (showLocationDialog || (!isManualSelection && locationLoading && !hasInitialLoad)) {
      return <WeatherDashboardSkeleton />;
    }

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
        <WeatherDashboard weatherData={weatherData} unit={unit} />
      </Suspense>
    );
  }, [showLocationDialog, isManualSelection, locationLoading, hasInitialLoad, error, weatherData, isLoading, unit]);

  return (
    <>
      <div className="min-h-screen flex flex-col">
        <LocationPermissionDialog open={showLocationDialog} onOpenChange={setShowLocationDialog} onAllowLocation={() => handleLocationPermission(true)} onDenyLocation={() => handleLocationPermission(false)} />
        <ErrorBoundary
          FallbackComponent={ErrorFallback}
          onReset={() => {
            setError(null);
            handleUseCurrentLocation();
          }}
        >
          <Navbar onLocationChange={handleLocationChange} onUseCurrentLocation={handleUseCurrentLocation} isUsingCurrentLocation={!isManualSelection} />
          {renderContent()}
        </ErrorBoundary>
      </div>
    </>
  );
}
