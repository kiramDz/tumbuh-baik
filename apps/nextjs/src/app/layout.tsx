import type { Metadata } from "next";
import "./globals.css";
import { fontMono, fontSans } from "@/lib/font";
import { cn } from "@/lib/utils";
import { Toaster as Sonner } from "@/components/ui/sonner";
import QueryProvider from "@/context/query-provider";
export const metadata: Metadata = {
  title: "Lotus",
  description: "The only solution you ever need for secure files storage.",
};

// TODO: Implement session and account data storage in MongoDB container
// Ensure login functionality works properly

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={cn(`min-h-svh bg-background font-mono antialiased`, fontSans.variable, fontMono.variable)}>
        <QueryProvider>
          {children}
          <Sonner />
        </QueryProvider>
      </body>
    </html>
  );
}
