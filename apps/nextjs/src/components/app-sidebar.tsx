"use client";

import * as React from "react";
import { Icons } from "@/app/dashboard/_components/icons";

import { NavMain } from "@/components/nav-main";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

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
      icon: Icons.layoutDashboardIcon,
    },
    {
      title: "Reports",
      url: "/dashboard/results",
      icon: Icons.clipboardListIcon,
    },
    {
      title: "Kalender Tanam",
      url: "/dashboard/kaltam",
      icon: Icons.barChartIcon,
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
      icon: Icons.bean,
    },
    {
      title: "User Management",
      url: "/dashboard/users",
      icon: Icons.users,
    },
    {
      title: "Recycle Bin",
      url: "/dashboard/recycle-bin",

      icon: Icons.trash,
    },
  ],

  documents: [
    {
      name: "Data Library",
      url: "#",
      icon: Icons.databaseIcon,
    },
    {
      name: "Reports",
      url: "#",
      icon: Icons.clipboardListIcon,
    },
    {
      name: "Word Assistant",
      url: "#",
      icon: Icons.fileIcon,
    },
  ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" className="pl-4" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <a href="#">
                <Icons.arrowUpCircleIcon className="h-6 w-6" />
                <span className="text-lg font-semibold">ZonaPetik.</span>
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
