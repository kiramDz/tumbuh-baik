import type { Metadata } from "next";
import PageContainer from "@/components/page-container";
import { MapPin } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Chart1 } from "@/components/chart/chart1";
import { Chart2 } from "@/components/chart/chart2";
import { Chart3 } from "@/components/chart/chart3";
import { Chart4 } from "@/components/chart/chart4";
import WeatherChart from "@/components/chart/weather-chart";
export const metadata: Metadata = {
  title: "Next Shadcn Dashboard Starter",
  description: "Basic dashboard with Next.js and Shadcn",
};
export default async function PublicPage() {
  return (
    <>
      <PageContainer>
        <div className="flex flex-1 flex-col space-y-2">
          <div className="flex items-center justify-between space-y-2">
            <h2 className="text-2xl font-bold tracking-tight">Hi, Welcome back 👋</h2>
          </div>

          <div className="w-full p-4">
            {/* Main grid with 2 columns on md screens */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Left section */}
              <div className="space-y-4">
                {/* Top 3 cards */}
                <div className="grid grid-cols-3 gap-4">
                  {/* Total Shipments Card */}
                  <WeatherChart />

                  {/* Active Tracking Card */}
                  <Chart2 />

                  {/* Delivered Shipment Card */}
                  <Chart3 />
                </div>

                {/* Chart Card - Full width below the 3 cards */}
                <Chart4 />
              </div>

              {/* Right section - Map Card */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Delivery In Progress</CardTitle>
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="h-[400px] bg-muted rounded-md flex items-center justify-center">
                    <span className="text-sm text-muted-foreground">Map View</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </PageContainer>
    </>
  );
}
