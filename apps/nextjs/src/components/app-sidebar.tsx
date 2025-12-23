"use client";

import * as React from "react";
import {
  ArrowUpCircleIcon,
  BarChartIcon,
  ClipboardListIcon,
  DatabaseIcon,
  FileIcon,
  LayoutDashboardIcon,
  Trash,
  Users,
  Bean,
} from "lucide-react";

import { NavMain } from "@/components/nav-main";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useSession } from "@/lib/better-auth/auth-client";

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { isPending, data } = useSession();

  // Ambil data user dari session
  const userData = {
    name: data?.user?.name || "",
    email: data?.user?.email || "",
    avatar: data?.user?.image || "", // pastikan backend mengirim field ini
  };

  const navMain = [
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
          title: "lstm",
          url: "/dashboard/kaltam",
        },
      ],
    },
    {
      title: "Data bibit",
      url: "/dashboard/bibit",
      icon: Bean,
    },
    {
      title: "User Management",
      url: "dashboard/users",
      icon: Users,
    },
    {
      title: "Recycle Bin",
      url: "dashboard/recycle-bin",
      icon: Trash,
    },
  ];
  const documents = [
    { name: "Data Library", url: "#", icon: DatabaseIcon },
    { name: "Reports", url: "#", icon: ClipboardListIcon },
    { name: "Word Assistant", url: "#", icon: FileIcon },
  ];

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
                <ArrowUpCircleIcon className="h-5 w-5" />
                <span className="text-base font-semibold">Zona Petik</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={navMain} />
      </SidebarContent>
      <SidebarFooter>
        {/* ⬇ Kirim data user yang sudah di-fetch */}
        <NavUser user={userData} />
      </SidebarFooter>
            {/* <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter> */}
    </Sidebar>
  );
}
