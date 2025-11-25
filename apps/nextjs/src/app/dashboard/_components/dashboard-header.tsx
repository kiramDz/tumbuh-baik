"use client";

// tambar theme dari : https://shadcn-dashboard.kiranism.dev/dashboard/overview

import { SidebarTrigger } from "@/components/ui/sidebar";
import DashboardBreadcrumb from "./dashboard-breadcrumb";
import AddDatasetDialog from "./dialog/add-dataset-unify-dialog";
import HeaderProfile from "./header-profile";
import { Separator } from "@/components/ui/separator";
import RefreshAllDatasetsDialog from "./dialog/refresh-all-datasets-dialog";

const DashboardHeader = () => {
  return (
    <header className="group-has-data-[collapsible=icon]/sidebar-wrapper:h-12 flex h-12 shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear">
      <div className="flex items-center gap-4 justify-start flex-1">
        <SidebarTrigger />
        <DashboardBreadcrumb />
      </div>
      <Separator
        orientation="vertical"
        className="mx-2 data-[orientation=vertical]:h-4"
      />
      <div className="w-full h-fit flex itemsq-center gap-4 justify-end flex-1">
        <RefreshAllDatasetsDialog />
        <AddDatasetDialog />
        <HeaderProfile />
      </div>
    </header>
  );
};

export default DashboardHeader;
