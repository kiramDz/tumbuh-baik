"use client";

import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export const WeatherLoading = React.memo(() => {
  return (
    <div className="space-y-6">
      {/* Header Skeleton - Fixed height to prevent CLS */}
      <Card className="bg-white/90 dark:bg-gray-800/90 border border-gray-200/50 min-h-[120px]">
        <CardHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gray-300 dark:bg-gray-600 rounded-full animate-pulse" />
            <div className="space-y-2">
              <div className="h-6 w-40 bg-gray-300 dark:bg-gray-600 rounded animate-pulse" />
              <div className="h-4 w-32 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-12 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
        </CardContent>
      </Card>

      {/* Main Content Skeleton - Fixed height */}
      <Card className="bg-white/90 dark:bg-gray-800/90 min-h-[200px]">
        <CardContent className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
            <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse" />
            <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
          </div>
        </CardContent>
      </Card>

      {/* Stats Grid Skeleton - Fixed height */}
      <Card className="bg-white/90 dark:bg-gray-800/90 min-h-[140px]">
        <CardContent className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-20 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-white/90 dark:bg-gray-800/90">
        <CardContent className="p-6">
          <div className="h-80 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
        </CardContent>
      </Card>
    </div>
  );
});

WeatherLoading.displayName = 'WeatherLoading';