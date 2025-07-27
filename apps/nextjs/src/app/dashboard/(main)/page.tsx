"use client";

import PageContainer from "@/components/ui/page-container";

import { FileText } from "lucide-react";
import { DashboardCard, DashboardCardSkeleton } from "../_components/dashboard-card";
import { GetAllDatasetMeta } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
export default function Page() {
  const { data, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: GetAllDatasetMeta,
  });
  return (
    <>
      <PageContainer>
        <div className="flex flex-1 flex-col gap-4 space-y-2">
          <div className="flex items-center  justify-between space-y-2 ">
            <h2 className="text-2xl font-bold tracking-tight">Hi, Welcome back 👋</h2>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {isLoading && [...Array(4)].map((_, i) => <DashboardCardSkeleton key={i} />)}

            {data &&
              data.map((dataset: any) => (
                <DashboardCard key={dataset.collectionName} href={`/dashboard/data/${dataset.collectionName}`} title={dataset.name} description={dataset.description || `Dataset dari collcetion ${dataset.collectionName}`} icon={FileText} />
              ))}
          </div>

        </div>
      </PageContainer>
    </>
  );
}
