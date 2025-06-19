import { Children } from "@/props/types";
import { SidebarProvider } from "@/components/ui/sidebar";
import DashboardHeader from "./_components/dashboard-header";
import { AppSidebar } from "@/components/app-sidebar";

const Layout = ({ children }: Children) => {
  return (
    <main className="bg-[#f9f9f9]">
      <SidebarProvider>
        <AppSidebar />
        <div className="w-full  bg-white  rounded-lg shadow-sm">
          <DashboardHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="flex flex-col py-4 md:py-6 ">{children}</div>
            </div>
          </div>
        </div>
      </SidebarProvider>
    </main>
  );
};

export default Layout;
