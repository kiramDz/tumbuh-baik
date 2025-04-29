import PageContainer from "@/components/page-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/icons";
import RecentTable from "@/components/recent-table";
import Link from "next/link";
import MainTable from "./_components/main-table";
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
              <Card className="w-full h-48 flex-col items-center justify-center ">
                <CardHeader className="flex flex-row items-center justify-center s">
                  <CardTitle className="text-sm font-medium">Buoys Data</CardTitle>
                </CardHeader>
                <CardContent className="flex items-center w-full justify-center p-0">
                  <Icons.msfile />
                </CardContent>
              </Card>
            </Link>
            <Card className="w-full h-48 flex-col items-center justify-center ">
              <CardHeader className="flex flex-row items-center justify-center s">
                <CardTitle className="text-sm font-medium">BMKG Data</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center w-full justify-center p-0">
                <Icons.msfile />
              </CardContent>
            </Card>
            <Card className="w-full h-48 flex-col items-center justify-center ">
              <CardHeader className="flex flex-row items-center justify-center s">
                <CardTitle className="text-sm font-medium">Citra Satelite</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center w-full justify-center p-0">
                <Icons.msfile />
              </CardContent>
            </Card>
            <Card className="w-full h-48 flex-col items-center justify-center ">
              <CardHeader className="flex flex-row items-center justify-center s">
                <CardTitle className="text-sm font-medium">Daily weather</CardTitle>
              </CardHeader>
              <CardContent className="flex items-center w-full justify-center p-0">
                <Icons.msfile />
              </CardContent>
            </Card>
          </div>
          <RecentTable />
          <div className="container mx-auto py-4">
            <MainTable />
          </div>
        </div>
      </PageContainer>
    </>
  );
}
