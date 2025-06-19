"use client";

import { type ReactNode, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { YearlyCalender } from "./calender/planting-calender-2";
import { CloudSun, CalendarDays, Pi, ChartNoAxesGantt } from "lucide-react";
import Formula from "@/content/formula/aceh-besar.mdx";
interface ProjectTabsProps {
  defaultTab?: string;
  children: ReactNode;
}

export function WeatherTabs({ defaultTab = "activity", children }: ProjectTabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <div className="px-6 ">
      <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full justify-start border-b bg-transparent p-0">
          <TabsTrigger value="weather" className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-green-600 data-[state=active]:bg-transparent">
            <CloudSun />
            Weather
          </TabsTrigger>
          <TabsTrigger value="calender" className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-green-600 data-[state=active]:bg-transparent">
            <CalendarDays />
            Calender
          </TabsTrigger>
          <TabsTrigger value="calculation" className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-green-600 data-[state=active]:bg-transparent">
            <Pi />
            Calculation
          </TabsTrigger>
          <TabsTrigger value="overview" className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-green-600 data-[state=active]:bg-transparent">
            <ChartNoAxesGantt />
            Overview
          </TabsTrigger>
        </TabsList>

        <TabsContent value="weather" className="mt-6">
          {children}
        </TabsContent>

        <TabsContent value="calender" className="mt-6">
          <YearlyCalender />
        </TabsContent>

        <TabsContent value="calculation" className="mt-6">
          <div className="prose max-w-none dark:prose-invert">
            <Formula />
          </div>
        </TabsContent>

        <TabsContent value="overview" className="mt-6">
          <div className="rounded-lg border p-6">
            <h3 className="text-lg font-medium">Documents Content</h3>
            <p className="text-gray-500">Documents information will be displayed here.</p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
