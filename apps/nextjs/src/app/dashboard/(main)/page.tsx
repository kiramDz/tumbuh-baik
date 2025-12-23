"use client";

import PageContainer from "@/components/ui/page-container";
import DatasetOverview from "../_components/dataset-overview";

export default function Page() {
  return (
    <>
      <PageContainer>
        <div className="flex flex-1 flex-col gap-4 space-y-2">
          <div className="flex items-center  justify-between space-y-2 ">
            <h2 className="text-2xl font-bold tracking-tight">Hi, Welcome back 👋</h2>
          </div>

          {/* Using DatasetOverview component for dataset display */}
          <DatasetOverview />
        </div>
      </PageContainer>
    </>
  );
}
