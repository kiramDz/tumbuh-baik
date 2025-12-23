"use client";

import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import HeaderProfile from "@/app/dashboard/_components/header-profile";
import DashboardBreadcrumb from "@/app/dashboard/_components/dashboard-breadcrumb";

interface PublicHeaderProps {
  farmId: string;
}

export default function PublicHeader({ farmId }: PublicHeaderProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center gap-4 px-4 sm:px-6">
        {/* Sidebar Trigger */}
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />

        {/* Breadcrumb */}
        <div className="flex-1">
          <DashboardBreadcrumb />
        </div>

        {/* Profile */}
        <div className="ml-auto">
          <HeaderProfile />
        </div>
      </div>
    </header>
  );
}