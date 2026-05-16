"use client";

import { useQuery } from "@tanstack/react-query";
import { GetAllDatasetMeta } from "@/lib/fetch/files.fetch";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Icons } from "@/app/dashboard/_components/icons";
import Link from "next/link";
import { format } from "date-fns";
import { id } from "date-fns/locale";

// Expand type locally to handle preprocessing fields
interface CleanedDataset {
  _id: string;
  name: string;
  collectionName: string;
  originalCollectionName?: string;
  status: string;
  source: string;
  totalRecords: number;
  preprocessingReportId?: string | { $oid: string };
  lastUpdated?: string | { $date: string };
}

export default function ResultsPage() {
  const { data: datasets, isLoading } = useQuery({
    queryKey: ["dataset-meta"],
    queryFn: GetAllDatasetMeta,
  });

  // Filter only datasets with status 'preprocessed'
  const cleanedDatasets = (datasets || []).filter(
    (dataset) => dataset.status === "preprocessed",
  ) as unknown as CleanedDataset[];

  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <Icons.spinner className="text-muted-foreground h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          Preprocessing Reports
        </h1>
        <p className="text-muted-foreground">
          View analysis, imputation, and smoothing results for your cleaned
          datasets.
        </p>
      </div>

      {cleanedDatasets.length === 0 ? (
        <Card className="flex flex-col items-center justify-center py-12">
          <Icons.fileCheck className="text-muted-foreground mb-4 h-12 w-12 opacity-20" />
          <CardTitle>No Cleaned Datasets Found</CardTitle>
          <CardDescription className="mt-2 text-center">
            Anda belum memiliki dataset yang sudah dibersihkan. Lakukan
            preprocessing dan lihat hasil reportnya
          </CardDescription>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {cleanedDatasets.map((dataset) => {
            // Extract ID flexibly based on how Mongoose serialized it
            const reportId =
              typeof dataset.preprocessingReportId === "object"
                ? dataset.preprocessingReportId.$oid
                : dataset.preprocessingReportId;

            // Fallback in case ID is missing for some reason
            const targetUrl = reportId ? `/dashboard/results/${reportId}` : "#";

            return (
              <Card
                key={dataset._id}
                className="flex flex-col hover:shadow-md transition-shadow"
              >
                <CardHeader>
                  <CardTitle className="line-clamp-1">{dataset.name}</CardTitle>
                  <CardDescription className="line-clamp-1">
                    Original Series:{" "}
                    {dataset.originalCollectionName || "Unknown"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex-1 space-y-4">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Icons.database className="h-4 w-4" />
                    <span>{dataset.totalRecords.toLocaleString()} Records</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Icons.calendar className="h-4 w-4" />
                    <span>
                      Cleaned on{" "}
                      {dataset.lastUpdated
                        ? format(
                            new Date(
                              String(
                                typeof dataset.lastUpdated === "object"
                                  ? dataset.lastUpdated.$date
                                  : dataset.lastUpdated,
                              ).replace(/Z$/, ""),
                            ),
                            "dd MMM yyyy",
                            { locale: id },
                          )
                        : "N/A"}
                    </span>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button asChild className="w-full" disabled={!reportId}>
                    <Link href={targetUrl}>
                      <Icons.view className="mr-2 h-4 w-4" />
                      View Results
                    </Link>
                  </Button>
                </CardFooter>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
