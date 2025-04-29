import PageContainer from "@/components/page-container";
import { Card } from "@/components/ui/card";

import Link from "next/link";
import MainTable from "./_components/main-table";
import { LifeBuoy, Satellite, CloudSunRain, Earth } from "lucide-react";
export default function StorageLayout() {
  return (
    <>
      <PageContainer>
        <div className="flex flex-1 flex-col space-y-2">
          <div className="flex items-center justify-between space-y-2">
            <h2 className="text-2xl font-bold tracking-tight">Hi, Welcome back 👋</h2>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Link href="/dashboard/storage/bmkg-station">
              <Card className="w-full relative overflow-hidden  flex flex-col h-48 rounded-xl ">
                <div className="group relative cursor-pointer overflow-hidden bg-white px-6 pt-6 pb-8  ring-gray-900/5 transition-all duration-300   sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
                  <span className="absolute top-6 z-0 h-10 w-10 rounded-full bg-sky-500 transition-all duration-300 group-hover:scale-[100]"></span>
                  <div className="relative z-10 mx-auto max-w-md">
                    <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-500 transition-all duration-300 group-hover:bg-sky-400">
                      <Earth className="text-white w-5 h-5" />
                    </span>

                    <div className="line-clamp-1 md:mt-6 flex gap-2 font-medium">BMKG</div>
                    <div className="text-muted-foreground text-base">Visitors for the last 6 months</div>
                  </div>
                </div>
              </Card>
            </Link>

            <Card className="w-full relative overflow-hidden  flex flex-col h-48 rounded-xl ">
              <div className="group relative cursor-pointer overflow-hidden bg-white px-6 pt-6 pb-8  ring-gray-900/5 transition-all duration-300   sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
                <span className="absolute top-6 z-0 h-10 w-10 rounded-full bg-sky-500 transition-all duration-300 group-hover:scale-[100]"></span>
                <div className="relative z-10 mx-auto max-w-md">
                  <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-500 transition-all duration-300 group-hover:bg-sky-400">
                    <CloudSunRain className="text-white w-5 h-5" />
                  </span>

                  <div className="line-clamp-1 md:mt-6 flex gap-2 font-medium">OPENWEATHER</div>
                  <div className="text-muted-foreground text-base">Visitors for the last 6 months</div>
                </div>
              </div>
            </Card>
            <Card className="w-full relative overflow-hidden  flex flex-col h-48 rounded-xl ">
              <div className="group relative cursor-pointer overflow-hidden bg-white px-6 pt-6 pb-8  ring-gray-900/5 transition-all duration-300   sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
                <span className="absolute top-6 z-0 h-10 w-10 rounded-full bg-sky-500 transition-all duration-300 group-hover:scale-[100]"></span>
                <div className="relative z-10 mx-auto max-w-md">
                  <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-500 transition-all duration-300 group-hover:bg-sky-400">
                    <Satellite className="text-white w-5 h-5" />
                  </span>

                  <div className="line-clamp-1 md:mt-6 flex gap-2 font-medium">SATELITE</div>
                  <div className="text-muted-foreground text-base">Visitors for the last 6 months</div>
                </div>
              </div>
            </Card>
            <Card className="w-full relative overflow-hidden  flex flex-col h-48 rounded-xl ">
              <div className="group relative cursor-pointer overflow-hidden bg-white px-6 pt-6 pb-8  ring-gray-900/5 transition-all duration-300   sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
                <span className="absolute top-6 z-0 h-10 w-10 rounded-full bg-sky-500 transition-all duration-300 group-hover:scale-[100]"></span>
                <div className="relative z-10 mx-auto max-w-md">
                  <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-500 transition-all duration-300 group-hover:bg-sky-400">
                    <LifeBuoy className="text-white w-5 h-5" />
                  </span>

                  <div className="line-clamp-1 md:mt-6 flex gap-2 font-medium">BUOYS</div>
                  <div className="text-muted-foreground text-base">Visitors for the last 6 months</div>
                </div>
              </div>
            </Card>
          </div>

          <div className="w-full mx-auto py-4">
            <MainTable />
          </div>
        </div>
      </PageContainer>
    </>
  );
}
