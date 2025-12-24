"use client";

import { type LucideIcon, ChevronRight } from "lucide-react";
import { SidebarGroup, SidebarGroupLabel, SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarMenuSub, SidebarMenuSubButton, SidebarMenuSubItem } from "@/components/ui/sidebar";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useState } from "react";

export function NavMain({
  items,
}: {
  items: {
    title: string;
    url: string;
    icon?: LucideIcon;
    isActive?: boolean;
    items?: {
      title: string;
      url: string;
    }[];
  }[];
}) {
  // Kontrol open state secara eksplisit per item
  const [openItems, setOpenItems] = useState<string[]>(() => {
    return items.filter((item) => item.isActive).map((item) => item.title);
  });

  const toggleOpen = (title: string) => {
    setOpenItems((prev) => (prev.includes(title) ? prev.filter((t) => t !== title) : [...prev, title]));
  };

  return (
    <SidebarGroup>
      <SidebarGroupLabel className="mt-10">Platform</SidebarGroupLabel>
      <SidebarMenu>
        {items.map((item) => {
          const isOpen = openItems.includes(item.title);
          const hasSubItems = !!item.items?.length;

          return (
            <SidebarMenuItem key={item.title}>
              <Collapsible open={isOpen} onOpenChange={() => toggleOpen(item.title)}>
                <CollapsibleTrigger asChild>
                  <SidebarMenuButton
                    tooltip={item.title}
                    className="w-full"
                    asChild={!hasSubItems} // hanya jadi link jika tidak ada subitem
                  >
                    {hasSubItems ? (
                      <div className="w-full flex gap-1 items-center">
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                        <ChevronRight className={`ml-auto transition-transform duration-200 ${isOpen ? "rotate-90" : ""}`} />
                      </div>
                    ) : (
                      <a href={item.url}>
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                      </a>
                    )}
                  </SidebarMenuButton>
                </CollapsibleTrigger>

                <CollapsibleContent>
                  <SidebarMenuSub>
                    {item.items?.map((subItem) => (
                      <SidebarMenuSubItem key={subItem.title}>
                        <SidebarMenuSubButton asChild>
                          <a href={subItem.url}>
                            <span>{subItem.title}</span>
                          </a>
                        </SidebarMenuSubButton>
                      </SidebarMenuSubItem>
                    ))}
                  </SidebarMenuSub>
                </CollapsibleContent>
              </Collapsible>
            </SidebarMenuItem>
          );
        })}
      </SidebarMenu>
    </SidebarGroup>
  );
}
