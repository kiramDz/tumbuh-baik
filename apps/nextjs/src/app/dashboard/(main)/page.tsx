import PageContainer from "@/components/ui/page-container";

import { LifeBuoy, Satellite, CloudSunRain, Earth } from "lucide-react";
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
            {/* BMKG Card */}
            <DashboardCard
              href="/dashboard/bmkg-station"
              icon={Earth} // Pass the icon component itself
              title="BMKG"
              description="Data cuaca dari stasiun BMKG Aceh Besar"
            />

            {/* Daily Weather Card */}
            <DashboardCard
              href="/dashboard/daily-weather" // Adjusted href for clarity, assuming a different path
              icon={CloudSunRain}
              title="DAILY WEATHER"
              description="Perkiraan cuaca harian dari OpenWeather"
            />

            {/* Satellite Card */}
            <DashboardCard
              href="/dashboard/satellite" // Adjusted href for clarity
              icon={Satellite}
              title="SATELITE"
              description="Gambaran kondisi lahan dan vegetasi berbasis citra satelit"
            />

            {/* Buoys Card */}
            <DashboardCard
              href="/dashboard/buoys" // Adjusted href for clarity
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
