import { Children } from "@/props/types";
import { SidebarProvider } from "@/components/ui/sidebar";
// import { AppSidebar } from "./_components/app-sidebar";
import DashboardHeader from "./_components/header";
import { AppSidebar } from "@/components/app-sidebar";

const Layout = ({ children }: Children) => {
  return (
    <main className="bg-[#f9f9f9]">
      <SidebarProvider>
        <AppSidebar />
        <div className="w-full px-5 bg-white m-1 rounded-lg shadow-sm">
          <DashboardHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">{children}</div>
            </div>
          </div>
        </div>
      </SidebarProvider>
    </main>
  );
};

export default Layout;
