import type { Metadata } from "next";
import "./globals.css";
import { fontMono, fontMonserrat } from "@/lib/font";
import { cn } from "@/lib/utils";
import { Toaster as Sonner } from "@/components/ui/sonner";
import QueryProvider from "@/context/query-provider";
import { siteConfig } from "@/config/site";

export const metadata: Metadata = {
  title: siteConfig.name,
  description: siteConfig.description,
  metadataBase: new URL("https://zonapetik.tech/"),
  authors: [
    {
      name: "Zona Petik",
      url: "https://zonapetik.tech/",
    },
  ],
  creator: "brokariim",
  openGraph: {
    type: "website",
    locale: "en_US",
    url: siteConfig.url,
    title: siteConfig.name,
    description: siteConfig.description,
    siteName: siteConfig.name,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={cn(`min-h-svh bg-background font-mono antialiased`, fontMonserrat.variable, fontMono.variable)}>
        <QueryProvider>
          {children}
          <Sonner />
        </QueryProvider>
      </body>
    </html>
  );
}
