import type { Metadata } from "next";
import MdxWrapper from "@/components/mdx-wrapper";
export const metadata: Metadata = {
  title: "Zona Petik",
  description: "Forecast and plan your planting activities with Zona Petik's weather dashboard and planting calendar.",
};

//TODO : Mash lambat kli login diawal, lalu kok ada tulisaj "loading padhl udh ada skeleton"
export default async function PublicLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <main className=" w-full overflow-hidden min-h-screen ">
        <MdxWrapper>{children}</MdxWrapper>
      </main>
    </>
  );
}
