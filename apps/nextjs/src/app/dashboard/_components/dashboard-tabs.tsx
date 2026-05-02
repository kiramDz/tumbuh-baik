import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DashboardCard } from "./dashboard-card";
import { FileText, Sparkles } from "lucide-react";

interface DatasetTabsProps {
  datasets: any[];
}

export default function DatasetTabs({ datasets }: DatasetTabsProps) {
  // Phase 1: Filtering Logic
  const originalDatasets = datasets.filter(
    (dataset) => !dataset.collectionName.endsWith("_cleaned"),
  );

  const cleanedDatasets = datasets.filter((dataset) =>
    dataset.collectionName.endsWith("_cleaned"),
  );

  return (
    <Tabs defaultValue="original" className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Your Datasets</h2>
        <div className="flex items-center gap-4">
          <TabsList>
            <TabsTrigger value="original">
              Original ({originalDatasets.length})
            </TabsTrigger>
            <TabsTrigger value="cleaned">
              Cleaned ({cleanedDatasets.length})
            </TabsTrigger>
          </TabsList>
        </div>
      </div>

      <TabsContent value="original" className="m-0">
        {originalDatasets.length > 0 ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {originalDatasets.map((dataset) => (
              <DashboardCard
                key={dataset.collectionName}
                collectionName={dataset.collectionName}
                dataset={dataset}
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
        ) : (
          <div className="mt-4 flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-muted">
              <Sparkles className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="mt-4 text-lg font-semibold">
              No original datasets found
            </h3>
            <p className="mt-2 max-w-sm text-sm text-muted-foreground">
              Get started by adding your first dataset to begin analyzing data.
            </p>
          </div>
        )}
      </TabsContent>

      <TabsContent value="cleaned" className="m-0">
        {cleanedDatasets.length > 0 ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {cleanedDatasets.map((dataset) => (
              <DashboardCard
                key={dataset.collectionName}
                collectionName={dataset.collectionName}
                dataset={dataset}
                href={`/dashboard/data/${dataset.collectionName}`}
                title={dataset.name}
                description={
                  dataset.description ||
                  `Explore cleaned data from ${dataset.collectionName}`
                }
                icon={FileText}
              />
            ))}
          </div>
        ) : (
          <div className="mt-4 flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
            <div className="flex h-20 w-20 items-center justify-center rounded-full bg-muted">
              <Sparkles className="h-10 w-10 text-muted-foreground" />
            </div>
            <h3 className="mt-4 text-lg font-semibold">
              No cleaned datasets found
            </h3>
            <p className="mt-2 max-w-sm text-sm text-muted-foreground">
              Clean or preprocess your datasets to view them here.
            </p>
          </div>
        )}
      </TabsContent>
    </Tabs>
  );
}
