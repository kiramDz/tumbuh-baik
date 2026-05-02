"use client";

import PageContainer from "@/components/ui/page-container";
import { FileText, Sparkles } from "lucide-react";
import {
  DashboardCard,
  DashboardCardSkeleton,
} from "../_components/dashboard-card";
import { GetAllDatasetMeta } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import DatasetTabs from "../_components/dashboard-tabs";

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
          </div>
          <p className="text-muted-foreground">
            Manage your datasets and explore agricultural data insights
          </p>
        </div>

        <Separator />

        {/* Error State */}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>
              Failed to load datasets. Please try refreshing the page.
            </AlertDescription>
          </Alert>
        )}

        {/* Loading State */}
        {isLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <DashboardCardSkeleton key={i} />
            ))}
          </div>
        ) : (
          data && <DatasetTabs datasets={data} />
        )}
      </div>
    </PageContainer>
  );
}
