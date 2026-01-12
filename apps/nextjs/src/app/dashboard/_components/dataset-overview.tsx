"use client";

import { useQuery } from "@tanstack/react-query";
import { Icons } from "./icons";
import { GetAllDatasetMeta } from "@/lib/fetch/files.fetch";
import { DashboardCard } from "./dashboard-card";
import { Skeleton } from "@/components/ui/skeleton";

export default function DatasetOverview() {
  const {
    data: datasets,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["dataset-meta"],
    queryFn: GetAllDatasetMeta,
  });

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-40 w-full rounded-2xl" />
        ))}
      </div>
    );
  }

  if (error) {
    console.error("🚨 Dataset fetch error:", error);
    return (
      <p className="text-red-500">Error loading datasets: {error.message}</p>
    );
  }

  if (!datasets || datasets.length === 0) {
    return <p className="text-muted-foreground">No datasets available.</p>;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
      {datasets.map((dataset, index) => {
        // Prepare a fixed-length description to prevent wrapping issues
        const formattedDescription = dataset.description
          ? dataset.description.length > 120
            ? dataset.description.substring(0, 120) + "..."
            : dataset.description
          : "No description available";

        return (
          <DashboardCard
            key={dataset._id}
            href={`/dashboard/data/${encodeURIComponent(
              dataset.collectionName
            )}`}
            icon={Icons.database}
            title={dataset.name}
            description={formattedDescription}
            dataset={dataset}
          />
        );
      })}
    </div>
  );
}
