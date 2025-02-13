import { Children } from "@/props/types";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "./_components/app-sidebar";
import DashboardHeader from "./_components/header";


const Layout = ({ children }: Children) => {
  return (
    <main className="bg-[#f9f9f9]">
      <SidebarProvider>
        <AppSidebar />
        <div className="w-full px-5 bg-white m-1 rounded-lg shadow-sm">
          <DashboardHeader />
          <div className=" w-full min-h-[calc(100vh-80px)] rounded-lg p-5">{children}</div>
        </div>
      </SidebarProvider>
    </main>
  );
};

export default Layout;
