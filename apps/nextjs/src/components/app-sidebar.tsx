"use client";

import * as React from "react";
import { Icons } from "@/app/dashboard/_components/icons";
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
      icon: Icons.dashboard,
    },
    {
      title: "Kalender Tanam",
      url: "/dashboard/kaltam",
      icon: Icons.calendar,
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
      title: "Food Security",
      url: "/dashboard/food-security",
      icon: Icons.map,
    },
    {
      title: "Data bibit",
      url: "/dashboard/bibit",
      icon: Icons.seed,
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
  ];
  const documents = [
    { name: "Data Library", url: "#", icon: Icons.database },
    { name: "Reports", url: "#", icon: Icons.fileCheck },
    { name: "Word Assistant", url: "#", icon: Icons.fileUpload },
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
                <Icons.logo className="h-5 w-5" />
                <span className="text-base font-semibold">Tumbuh Baik.</span>
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
    </Sidebar>
  );
}
