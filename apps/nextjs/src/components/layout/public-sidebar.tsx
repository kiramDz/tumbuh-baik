"use client";

import { cn } from "@/lib/utils";
import { BarChart3, Home, TrendingUp, Calendar, MapPin } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface PublicSidebarProps {
  farmId: string;
}

export default function PublicSidebar({ farmId }: PublicSidebarProps) {
  const pathname = usePathname();

  const sidebarItems = [
    {
      title: "Dashboard Overview",
      href: `/${farmId}`,
      icon: Home,
      isMain: true
    },
    {
      title: "Analytics",
      href: `/${farmId}/analytics`, 
      icon: BarChart3
    },
    {
      title: "Seasonal Reports",
      href: `/${farmId}/seasonal`,
      icon: Calendar
    },
    {
      title: "Farm Management",
      href: `/${farmId}/management`,
      icon: MapPin
    }
  ];

  return (
    <div className="fixed left-0 top-16 h-full w-64 bg-white border-r border-gray-200 z-40">
      <div className="p-6">
        {/* Farm Profile Section */}
        <div className="mb-8 p-4 bg-green-50 rounded-lg border border-green-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
              <span className="text-white font-semibold">F</span>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900">Farm {farmId}</p>
              <p className="text-xs text-gray-500">Agriculture Dashboard</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="space-y-2">
          {sidebarItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-green-100 text-green-700 border border-green-200"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900",
                  item.isMain && "ring-1 ring-green-300"
                )}
              >
                <item.icon className="h-5 w-5" />
                <span>{item.title}</span>
                {item.isMain && (
                  <span className="ml-auto text-xs bg-green-600 text-white px-2 py-1 rounded-full">
                    Main
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Farm Info Card */}
        <div className="absolute bottom-6 left-6 right-6">
          <div className="p-4 bg-gray-50 rounded-lg border">
            <div className="text-center">
              <div className="text-sm font-medium text-gray-900">Farm ID</div>
              <div className="text-lg font-bold text-green-600">{farmId}</div>
              <div className="text-xs text-gray-500 mt-1">Active Dashboard</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}