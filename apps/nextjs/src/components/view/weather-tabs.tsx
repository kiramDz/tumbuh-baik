"use client";

import React, { type ReactNode, useState } from "react";
import dynamic from "next/dynamic";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { CloudSun, CalendarDays, ChartNoAxesGantt } from "lucide-react";

// Lazy load heavy tab components
const YearlyCalender = dynamic(() => import("./calender/planting-calender-2").then(mod => ({ default: mod.YearlyCalender })), {
  loading: () => <div className="h-64 bg-gray-200 dark:bg-gray-800 rounded-lg" />,
  ssr: false
});

const ChartPetani = dynamic(() => import("../kuesioner-chart/chartpetani").then(mod => ({ default: mod.ChartPetani })), {
  loading: () => <div className="h-64 bg-gray-200 dark:bg-gray-800 rounded-lg" />,
  ssr: false
});

const AboutContent = dynamic(() => import("./about/about-main").then(mod => ({ default: mod.AboutContent })), {
  loading: () => <div className="h-64 bg-gray-200 dark:bg-gray-800 rounded-lg" />,
  ssr: false
});

interface ProjectTabsProps {
  defaultTab?: string;
  children: ReactNode;
}

export const WeatherTabs = React.memo<ProjectTabsProps>(({ defaultTab = "activity", children }) => {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <div>
      <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="w-full px-0">
        <TabsList className="w-full justify-start border-b bg-transparent p-0">
          <TabsTrigger 
            value="weather" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 text-base data-[state=active]:border-teal-600 data-[state=active]:bg-transparent data-[state=active]:text-teal-600"
          >
            <CloudSun className="h-4 w-4" />
            Weather
          </TabsTrigger>
          <TabsTrigger 
            value="calender" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 text-base data-[state=active]:border-teal-600 data-[state=active]:bg-transparent data-[state=active]:text-teal-600"
          >
            <CalendarDays className="h-4 w-4" />
            Calendar
          </TabsTrigger>
          <TabsTrigger 
            value="kuesioner" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 text-base data-[state=active]:border-teal-600 data-[state=active]:bg-transparent data-[state=active]:text-teal-600"
          >
            <ChartNoAxesGantt className="h-4 w-4" />
            Kuesioner
          </TabsTrigger>
          {/* <TabsTrigger 
            value="about" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary"
          >
            <ChartNoAxesGantt className="h-4 w-4" />
            About
          </TabsTrigger> */}
        </TabsList>

        <TabsContent value="weather" className="mt-6 mx-6">
          {children}
        </TabsContent>

        <TabsContent value="calender" className="mt-6 mx-6">
          <YearlyCalender />
        </TabsContent>

        <TabsContent value="kuesioner" className="mt-6 mx-6 md:mx-24">
          <ChartPetani />
        </TabsContent>

        <TabsContent value="about" className="mt-6 mx-6 md:mx-24">
          <AboutContent />
        </TabsContent>
      </Tabs>
    </div>
  );
});

WeatherTabs.displayName = 'WeatherTabs';
