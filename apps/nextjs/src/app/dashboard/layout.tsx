import { Children } from "@/props/types";
import { SidebarProvider } from "@/components/ui/sidebar";
import DashboardHeader from "./_components/dashboard-header";
import { AppSidebar } from "@/components/app-sidebar";
import { Toaster } from "react-hot-toast";

const Layout = ({ children }: Children) => {
  return (
    <main className="bg-[#f9f9f9]">
      <SidebarProvider>
        <AppSidebar />
        <div className="w-full bg-white rounded-lg shadow-sm">
          <DashboardHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="py-4 md:py-6">{children}</div>
            </div>
          </div>
        </div>
      </SidebarProvider>

      {/* Tambah Toaster di sini */}
      <Toaster
        position="bottom-right"
        containerStyle={{
          bottom: "1rem",
          right: "1rem",
        }}
        toastOptions={{
          duration: 4000,
          className: "font-sans",
          style: {
            background: "hsl(var(--background))",
            color: "hsl(var(--foreground))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "0.5rem", // rounded-lg equivalent
            fontSize: "0.875rem", // text-sm equivalent
            fontWeight: "500", // font-medium equivalent
            lineHeight: "1.25rem",
            padding: "0.75rem 1rem", // px-4 py-3 equivalent
            boxShadow:
              "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)", // shadow-md equivalent
            fontFamily: "inherit",
            minWidth: "288px", // min-w-72 equivalent
            maxWidth: "400px",
          },
          success: {
            iconTheme: {
              primary: "hsl(var(--primary))",
              secondary: "hsl(var(--primary-foreground))",
            },
            style: {
              background: "hsl(var(--background))",
              color: "hsl(var(--foreground))",
              border: "1px solid hsl(var(--primary))",
            },
          },
          error: {
            iconTheme: {
              primary: "hsl(var(--destructive))",
              secondary: "hsl(var(--destructive-foreground))",
            },
            style: {
              background: "hsl(var(--background))",
              color: "hsl(var(--foreground))",
              border: "1px solid hsl(var(--destructive))",
            },
          },
          loading: {
            iconTheme: {
              primary: "hsl(var(--muted-foreground))",
              secondary: "hsl(var(--background))",
            },
            style: {
              background: "hsl(var(--background))",
              color: "hsl(var(--foreground))",
              border: "1px solid hsl(var(--muted))",
            },
          },
        }}
      />
    </main>
  );
};

export default Layout;
