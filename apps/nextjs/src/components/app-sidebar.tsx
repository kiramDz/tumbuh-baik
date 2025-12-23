"use client";

import * as React from "react";
import { ArrowUpCircleIcon, BarChartIcon, ClipboardListIcon, DatabaseIcon, FileIcon, LayoutDashboardIcon, Trash, Users, Bean } from "lucide-react";

import { NavMain } from "@/components/nav-main";
import { Sidebar, SidebarContent, SidebarHeader, SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";

const data = {
  user: {
    name: "shadcn",
    email: "m@example.com",
    avatar: "/avatars/shadcn.jpg",
  },
  navMain: [
    {
      title: "Dashboard",
      url: "/dashboard",
      icon: LayoutDashboardIcon,
    },
    {
      title: "Kalender Tanam",
      url: "/dashboard/kaltam",
      icon: BarChartIcon,
      items: [
        {
          title: "Holt Winter",
          url: "/dashboard/kaltam",
        },
        {
          title: "LSTM",
          url: "/dashboard/kaltam-lstm",
        },
      ],
    },
    {
      title: "Data Bibit",
      url: "/dashboard/bibit",
      icon: Bean,
    },
    {
      title: "User Management",
      url: "/dashboard/users",
      url: "/dashboard/users",
      icon: Users,
    },
    {
      title: "Recycle Bin",
      url: "/dashboard/recycle-bin",
      title: "Recycle Bin",
      url: "/dashboard/recycle-bin",
      icon: Trash,
    },
  ],

  documents: [
    {
      name: "Data Library",
      url: "#",
      icon: DatabaseIcon,
    },
    {
      name: "Reports",
      url: "#",
      icon: ClipboardListIcon,
    },
    {
      name: "Word Assistant",
      url: "#",
      icon: FileIcon,
    },
  ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" className="pl-4" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild className="data-[slot=sidebar-menu-button]:!p-1.5">
              <a href="#">
                <ArrowUpCircleIcon className="h-5 w-5" />
                <span className="text-base font-semibold">ZonaPetik.</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
      </SidebarContent>
      {/* <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter> */}
    </Sidebar>
  );
}
