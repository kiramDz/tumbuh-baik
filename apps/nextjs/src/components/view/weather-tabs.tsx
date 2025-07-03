"use client";

import { type ReactNode, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { YearlyCalender } from "./calender/planting-calender-2";

import { CloudSun, CalendarDays, Pi, ChartNoAxesGantt } from "lucide-react";
import Formula from "@/content/formula/aceh-besar.mdx";
import Overview from "@/content/overview/aceh-besar.mdx";
interface ProjectTabsProps {
  defaultTab?: string;
  children: ReactNode;
}

export function WeatherTabs({ defaultTab = "activity", children }: ProjectTabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <div>
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
          {/* <TabsTrigger value="calculation" className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-green-600 data-[state=active]:bg-transparent">
            <Pi />
            Calculation
          </TabsTrigger> */}
          <TabsTrigger value="overview" className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-green-600 data-[state=active]:bg-transparent">
            <ChartNoAxesGantt />
            Overview
          </TabsTrigger>
        </TabsList>

        <TabsContent value="weather" className="mt-6 mx-6">
          {children}
        </TabsContent>

        <TabsContent value="calender" className="mt-6 mx-6">
          <YearlyCalender />
        </TabsContent>

        {/* <TabsContent value="calculation" className="mt-6">
          <div className="prose max-w-none dark:prose-invert">
            <Formula />
          </div>
        </TabsContent> */}

        <TabsContent value="overview" className="mt-6 mx-24">
          <Overview />
        </TabsContent>
      </Tabs>
    </div>
  );
}
