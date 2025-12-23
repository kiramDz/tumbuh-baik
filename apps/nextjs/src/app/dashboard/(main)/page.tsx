"use client";

import PageContainer from "@/components/ui/page-container";
import { FileText, Sparkles } from "lucide-react";
import { DashboardCard, DashboardCardSkeleton } from "../_components/dashboard-card";
import { GetAllDatasetMeta } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";

export default function Page() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["datasets"],
    queryFn: GetAllDatasetMeta,
  });

  return (
    <PageContainer>
      <div className="flex flex-1 flex-col gap-6">
        {/* Header Section */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-bold tracking-tight">
              Hi, Welcome back
            </h1>
            <span className="text-3xl">👋</span>
          </div>
          <p className="text-muted-foreground">
            Manage your datasets and explore agricultural data insights
          </p>
        </div>

        <Separator />

        {/* Content Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Your Datasets</h2>
            {data && data.length > 0 && (
              <span className="text-sm text-muted-foreground">
                {data.length} {data.length === 1 ? "dataset" : "datasets"} available
              </span>
            )}
          </div>

          {/* Error State */}
          {error && (
            <Alert variant="destructive">
              <AlertDescription>
                Failed to load datasets. Please try refreshing the page.
              </AlertDescription>
            </Alert>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {[...Array(4)].map((_, i) => (
                <DashboardCardSkeleton key={i} />
              ))}
            </div>
          )}

          {/* Data Grid */}
          {data && data.length > 0 && (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {data.map((dataset: any) => (
                <DashboardCard
                  key={dataset.collectionName}
                  href={`/dashboard/data/${dataset.collectionName}`}
                  title={dataset.name}
                  description={
                    dataset.description || 
                    `Explore data from ${dataset.collectionName} collection`
                  }
                  icon={FileText}
                />
              ))}
            </div>
          )}

          {/* Empty State */}
          {data && data.length === 0 && !isLoading && (
            <div className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-muted">
                <Sparkles className="h-10 w-10 text-muted-foreground" />
              </div>
              <h3 className="mt-4 text-lg font-semibold">No datasets found</h3>
              <p className="mt-2 text-sm text-muted-foreground max-w-sm">
                Get started by adding your first dataset to begin analyzing agricultural data.
              </p>
            </div>
          )}
        </div>
      </div>
    </PageContainer>
  );
}