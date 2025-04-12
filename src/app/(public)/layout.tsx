import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Next Shadcn Dashboard Starter",
  description: "Basic dashboard with Next.js and Shadcn",
};

export default async function PublicLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <main className="flex flex-1 w-full flex-col min-h-screen text-nowrap">{children}</main>
    </>
  );
}
