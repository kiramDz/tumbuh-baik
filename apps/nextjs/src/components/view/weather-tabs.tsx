"use client";

import { type ReactNode, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { YearlyCalender } from "./calender/planting-calender-2";
import { ChartPetani } from "../kuesioner-chart/chartpetani";
import { CloudSun, CalendarDays, ChartNoAxesGantt } from "lucide-react";
import { AboutContent } from "./about/about-main";

interface ProjectTabsProps {
  defaultTab?: string;
  children: ReactNode;
}

export function WeatherTabs({ defaultTab = "activity", children }: ProjectTabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab);

  return (
    <div>
      <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="w-full px-0">
        <TabsList className="w-full justify-start border-b bg-transparent p-0">
          <TabsTrigger 
            value="weather" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary"
          >
            <CloudSun className="h-4 w-4" />
            Weather
          </TabsTrigger>
          <TabsTrigger 
            value="calender" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary"
          >
            <CalendarDays className="h-4 w-4" />
            Calendar
          </TabsTrigger>
          <TabsTrigger 
            value="kuesioner" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary"
          >
            <ChartNoAxesGantt className="h-4 w-4" />
            Kuesioner
          </TabsTrigger>
          <TabsTrigger 
            value="about" 
            className="rounded-none flex items-center gap-2 border-b-2 border-transparent px-4 py-2 data-[state=active]:border-primary data-[state=active]:bg-transparent data-[state=active]:text-primary"
          >
            <ChartNoAxesGantt className="h-4 w-4" />
            About
          </TabsTrigger>
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
}
