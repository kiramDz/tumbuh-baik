import Link from "next/link";
import { Card } from "@/components/ui/card"; // Assuming you have a Card component from your UI library
import { ElementType } from "react";
import { EllipsisVertical } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { SoftDeleteDataset } from "@/lib/fetch/files.fetch";
import { useQueryClient } from "@tanstack/react-query";

interface DashboardCardProps {
  href: string;
  icon: ElementType;
  title: string;
  description: string;
  collectionName: string;
}

const DashboardCard: React.FC<DashboardCardProps> = ({ href, icon: Icon, title, description, collectionName }) => {
  const queryClient = useQueryClient();
  return (
    <Card className="relative flex h-48 w-full flex-col overflow-hidden rounded-xl">
      <div className="absolute right-2 top-2 z-20">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="rounded-full p-1 hover:bg-gray-100">
              <EllipsisVertical className="h-5 w-5 text-gray-600" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={async () => {
                try {
                  await SoftDeleteDataset(collectionName);
                  console.log("Soft deleted:", collectionName);
                  queryClient.invalidateQueries({ queryKey: ["datasets"] }); // ✅ refresh data
                } catch (err) {
                  console.error("Failed to soft delete:", err);
                }
              }}
            >
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <Link href={href} passHref className="flex-1">
        <div className="group relative cursor-pointer overflow-hidden bg-white px-6 pb-8 pt-6 ring-gray-900/5 transition-all duration-300 sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
          <span className="absolute top-6 z-0 h-10 w-10 rounded-full bg-sky-500 transition-all duration-300 group-hover:scale-[100]"></span>

          <div className="relative z-10 mx-auto max-w-md">
            <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-500 transition-all duration-300 group-hover:bg-sky-400">
              <Icon className="h-5 w-5 text-white" />
            </span>
            <div className="line-clamp-1 md:mt-6 flex gap-2 font-medium">{title}</div>
            <div className="text-base text-muted-foreground">{description}</div>
          </div>
        </div>
      </Link>
    </Card>
  );
};

const DashboardCardSkeleton: React.FC = () => {
  return (
    <Card className="relative flex h-48 w-full flex-col overflow-hidden rounded-xl">
      <div className="relative overflow-hidden bg-white px-6 pb-8 pt-6 ring-gray-900/5 sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
        <Skeleton className="absolute top-6 z-0 h-10 w-10 rounded-full" />

        <div className="relative z-10 mx-auto max-w-md">
          <Skeleton className="h-10 w-10 rounded-full" />
          <Skeleton className="mt-6 h-6 w-32" />
          <Skeleton className="mt-2 h-4 w-full" />
        </div>
      </div>
    </Card>
  );
};

export { DashboardCard, DashboardCardSkeleton };
