"use client";

import React from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface WeatherErrorProps {
  error: any;
  onRetry: () => void;
}

export const WeatherError = React.memo(({ error, onRetry }: WeatherErrorProps) => {
  return (
    <Card className="bg-white/90 dark:bg-gray-800/90 border border-red-200/50">
      <CardContent className="p-6">
        <Alert className="border-red-200 bg-red-50 dark:bg-red-900/20">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-700 dark:text-red-300">
            Gagal memuat data cuaca. Silakan coba lagi.
            <Button 
              onClick={onRetry} 
              variant="outline" 
              size="sm" 
              className="mt-4 w-full border-red-200 text-red-700 hover:bg-red-100"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Coba Lagi
            </Button>
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
});

WeatherError.displayName = 'WeatherError';