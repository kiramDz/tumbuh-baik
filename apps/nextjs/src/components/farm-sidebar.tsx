"use client";

import * as React from "react";
import { 
  Home,
  TrendingUp,
  Settings,
  MapPin,
  Calendar,
  Sprout,
  BarChart3,
  DollarSign,
  ChevronRight,
  ChevronsUpDown,
  User,
  CreditCard,
  Bell,
  LogOut
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { 
  Sidebar, 
  SidebarContent, 
  SidebarFooter, 
  SidebarHeader, 
  SidebarMenu, 
  SidebarMenuButton, 
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail,
  SidebarGroup,
  SidebarGroupLabel
} from "@/components/ui/sidebar";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import Avatar from "boring-avatars";

interface FarmSidebarProps extends React.ComponentProps<typeof Sidebar> {
  farmId: string;
}

export function FarmSidebar({ farmId, ...props }: FarmSidebarProps) {
  const pathname = usePathname();
  const router = useRouter();

  const navItems = [
    {
      title: "Overview",
      url: `/${farmId}`,
      icon: Home,
    },
    {
      title: "Statistics",
      url: `/${farmId}/statistics`,
      icon: BarChart3,
      items: [
        {
          title: "Productivity",
          url: `/${farmId}/statistics/productivity`,
        },
        {
          title: "Seasonal",
          url: `/${farmId}/statistics/seasonal`,
        },
        {
          title: "Comparison",
          url: `/${farmId}/statistics/comparison`,
        },
      ],
    },
    {
      title: "Farm Data",
      url: `/${farmId}/data`,
      icon: Sprout,
      items: [
        {
          title: "Crops",
          url: `/${farmId}/data/crops`,
        },
        {
          title: "Harvest",
          url: `/${farmId}/data/harvest`,
        },
        {
          title: "Planting",
          url: `/${farmId}/data/planting`,
        },
      ],
    },
    {
      title: "Financial",
      url: `/${farmId}/financial`,
      icon: DollarSign,
      items: [
        {
          title: "Revenue",
          url: `/${farmId}/financial/revenue`,
        },
        {
          title: "Expenses",
          url: `/${farmId}/financial/expenses`,
        },
        {
          title: "Profit",
          url: `/${farmId}/financial/profit`,
        },
      ],
    },
    {
      title: "Calendar",
      url: `/${farmId}/calendar`,
      icon: Calendar,
    },
    {
      title: "Location",
      url: `/${farmId}/location`,
      icon: MapPin,
    },
    {
      title: "Settings",
      url: `/${farmId}/settings`,
      icon: Settings,
    },
  ];

  const user = {
    name: "Farm Owner",
    email: "owner@farm.com",
  };

  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild size="lg">
              <Link href={`/${farmId}`}>
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <Sprout className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">Farm {farmId}</span>
                  <span className="truncate text-xs text-muted-foreground">
                    Agricultural Dashboard
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent className="overflow-x-hidden">
        <SidebarGroup>
          <SidebarGroupLabel>Farm Management</SidebarGroupLabel>
          <SidebarMenu>
            {navItems.map((item) => {
              const Icon = item.icon;
              return item?.items && item?.items?.length > 0 ? (
                <Collapsible
                  key={item.title}
                  asChild
                  defaultOpen={pathname.startsWith(item.url)}
                  className="group/collapsible"
                >
                  <SidebarMenuItem>
                    <CollapsibleTrigger asChild>
                      <SidebarMenuButton
                        tooltip={item.title}
                        isActive={pathname === item.url}
                      >
                        {Icon && <Icon className="size-4" />}
                        <span>{item.title}</span>
                        <ChevronRight className="ml-auto size-4 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                      </SidebarMenuButton>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <SidebarMenuSub>
                        {item.items?.map((subItem) => (
                          <SidebarMenuSubItem key={subItem.title}>
                            <SidebarMenuSubButton
                              asChild
                              isActive={pathname === subItem.url}
                            >
                              <Link href={subItem.url}>
                                <span>{subItem.title}</span>
                              </Link>
                            </SidebarMenuSubButton>
                          </SidebarMenuSubItem>
                        ))}
                      </SidebarMenuSub>
                    </CollapsibleContent>
                  </SidebarMenuItem>
                </Collapsible>
              ) : (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    tooltip={item.title}
                    isActive={pathname === item.url}
                  >
                    <Link href={item.url}>
                      {Icon && <Icon className="size-4" />}
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              );
            })}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                >
                  <div className="flex items-center gap-2">
                    <Avatar
                      name={user.name}
                      variant="beam"
                      colors={["#0a0310", "#49007e", "#ff005b", "#ff7d10", "#ffb238"]}
                      size={32}
                    />
                    <div className="grid flex-1 text-left text-sm leading-tight">
                      <span className="truncate font-semibold">{user.name}</span>
                      <span className="truncate text-xs text-muted-foreground">
                        {user.email}
                      </span>
                    </div>
                  </div>
                  <ChevronsUpDown className="ml-auto size-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
                side="bottom"
                align="end"
                sideOffset={4}
              >
                <DropdownMenuLabel className="p-0 font-normal">
                  <div className="flex items-center gap-2 px-1 py-1.5 text-left text-sm">
                    <Avatar
                      name={user.name}
                      variant="beam"
                      colors={["#0a0310", "#49007e", "#ff005b", "#ff7d10", "#ffb238"]}
                      size={32}
                    />
                    <div className="grid flex-1 text-left text-sm leading-tight">
                      <span className="truncate font-semibold">{user.name}</span>
                      <span className="truncate text-xs text-muted-foreground">
                        {user.email}
                      </span>
                    </div>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem
                    onClick={() => router.push(`/${farmId}/profile`)}
                  >
                    <User className="mr-2 h-4 w-4" />
                    Profile
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <CreditCard className="mr-2 h-4 w-4" />
                    Billing
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Bell className="mr-2 h-4 w-4" />
                    Notifications
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                  <LogOut className="mr-2 h-4 w-4" />
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}