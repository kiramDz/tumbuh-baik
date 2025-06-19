import PageContainer from "@/components/ui/page-container";
import { Heading } from "@/components/ui/heading";
import { Separator } from "@/components/ui/separator";
import { Suspense } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { DataTableSkeleton } from "../_components/data-table-skeleton";
import KaltamTable from "../_components/kaltam/kaltam-table";
export const metadata = {
  title: "Dashboard: Data Table",
};

export default async function Page() {
  return (
    <>
      <PageContainer scrollable={false}>
        <div className="flex flex-1 flex-col space-y-4">
          <div className="flex items-start justify-start">
            <Heading title="Kalender Tanam" description="Manage products (Server side table functionalities.)" />
          </div>
          <Separator />
          <Dialog>
            <form>
              <DialogTrigger asChild>
                <Button variant="outline">Tambah Data</Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>Edit profile</DialogTitle>
                  <DialogDescription>Make changes to your profile here. Click save when you&apos;re done.</DialogDescription>
                </DialogHeader>
                <div className="grid gap-4">
                  <div className="grid gap-3">
                    <Label htmlFor="name-1">Name</Label>
                    <Input id="name-1" name="name" defaultValue="Pedro Duarte" />
                  </div>
                  <div className="grid gap-3">
                    <Label htmlFor="username-1">Username</Label>
                    <Input id="username-1" name="username" defaultValue="@peduarte" />
                  </div>
                </div>
                <DialogFooter>
                  <DialogClose asChild>
                    <Button variant="outline">Cancel</Button>
                  </DialogClose>
                  <Button type="submit">Save changes</Button>
                </DialogFooter>
              </DialogContent>
            </form>
          </Dialog>
          <Suspense fallback={<DataTableSkeleton columnCount={5} rowCount={8} filterCount={2} />}>
            <KaltamTable />
          </Suspense>
        </div>
      </PageContainer>
    </>
  );
}
