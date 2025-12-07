import type { Metadata } from "next";
import { SidebarProvider } from "@/components/ui/sidebar";
import { FarmSidebar } from "@/components/farm-sidebar";
import PublicHeader from "./_components/public-header";

interface LayoutProps {
  children: React.ReactNode;
  params: Promise<{ id: string }>;
}

export default async function DynamicPublicLayout({ children, params }: LayoutProps) {
  const { id } = await params;
  
  return (
    <main className="bg-[#f9f9f9]">
      <SidebarProvider>
        <FarmSidebar farmId={id} />
        <div className="w-full bg-white rounded-lg shadow-sm">
          <PublicHeader farmId={id} />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="py-4 md:py-6">{children}</div>
            </div>
          </div>
        </div>
      </SidebarProvider>
    </main>
  );
}