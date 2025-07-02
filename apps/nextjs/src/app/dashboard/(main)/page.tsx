import PageContainer from "@/components/ui/page-container";

import { LifeBuoy,Earth } from "lucide-react";
import DashboardCard from "../_components/dashboard-card";

export const metadata = {
  title: "Dashboard",
};

export default function Page() {
  return (
    <>
      <PageContainer>
        <div className="flex flex-1 flex-col gap-4 space-y-2">
          <div className="flex items-center  justify-between space-y-2 ">
            <h2 className="text-2xl font-bold tracking-tight">Hi, Welcome back 👋</h2>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <DashboardCard
              href="/dashboard/bmkg"
              icon={Earth} // Pass the icon component itself
              title="BMKG"
              description="Data cuaca dari stasiun BMKG Aceh Besar"
            />
            <DashboardCard
              href="/dashboard/buoys" 
              icon={LifeBuoy}
              title="BUOYS"
              description="Informasi suhu permukaan dari website buoys"
            />
          </div>

          <div className="w-full mx-auto py-4"></div>
        </div>
      </PageContainer>
    </>
  );
}
