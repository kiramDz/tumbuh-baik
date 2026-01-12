import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { ElementType } from "react";
import { EllipsisVertical, Trash } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { DatasetMetaType } from "@/lib/fetch/files.fetch";
import EditDatasetDialog from "./edit-dataset-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import RefreshSingleDatasetDialog from "../(main)/data/_components/refresh-single-dataset-dialog";

interface DashboardCardProps {
  href: string;
  icon: ElementType;
  title: string;
  description?: string;
  collectionName?: string;
  dataset?: Partial<DatasetMetaType>;
  showMenu?: boolean;
  onSelect?: () => void; // <-- Add this
}

const DashboardCard: React.FC<DashboardCardProps> = ({
  href,
  icon: Icon,
  title,
  description,
  collectionName,
  dataset,
  showMenu = true, // Default to true if not provided
  onSelect,
}) => {
  const router = useRouter();

  // Use collectionName from props or from dataset object
  const effectiveCollectionName = collectionName || dataset?.collectionName;

  const encodedHref = effectiveCollectionName
    ? `/dashboard/data/${encodeURIComponent(effectiveCollectionName)}`
    : href;

  const handleCardClick = () => {
    if (onSelect) onSelect();
    router.push(encodedHref);
  };

  return (
    <Card
      className="relative flex h-48 w-full flex-col overflow-hidden rounded-xl cursor-pointer"
      onClick={handleCardClick}
    >
      <div className="group relative h-full w-full cursor-pointer overflow-hidden bg-white px-6 pb-8 pt-6 transition-all duration-300">
        {/* Blue hover effect background - positioned to fill entire card */}
        <span className="absolute inset-0 z-0 bg-white transition-all duration-300 group-hover:bg-sky-500/10"></span>

        {/* Blue accent circle that expands on hover */}
        <span className="absolute top-6 left-6 z-0 h-10 w-10 rounded-full bg-sky-500 transition-all duration-500 group-hover:scale-[6] group-hover:opacity-70"></span>

        {/* Menu in top right corner */}
        <div
          className="absolute top-6 right-6 z-20"
          onClick={(e) => {
            e.stopPropagation(); // Prevent card click
            e.preventDefault(); // Prevent default behavior
          }}
        >
          {dataset && showMenu ? (
            <EditDatasetDialog
              dataset={{
                _id: dataset._id || "",
                name: dataset.name || title || "",
                source: dataset.source || "",
                collectionName: dataset.collectionName || collectionName || "",
                description: dataset.description || description || "",
                status: dataset.status || "active",
                isAPI: dataset.isAPI,
                apiConfig: dataset.apiConfig
                  ? {
                      type: dataset.apiConfig.type || "default",
                      params: dataset.apiConfig.params,
                    }
                  : undefined,
                lastUpdated: dataset.lastUpdated,
              }}
            />
          ) : showMenu ? (
            <div className="text-xs text-gray-400">No menu</div>
          ) : null}
        </div>

        {/* Content */}
        <div className="relative z-10 h-full flex flex-col">
          {/* Icon */}
          <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-500 transition-all duration-300 group-hover:bg-sky-600">
            <Icon className="h-5 w-5 text-white" />
          </span>

          {/* Title with line clamp */}
          <div className="mt-4 line-clamp-1 font-medium text-gray-900 group-hover:text-sky-700 transition-colors duration-300">
            {title}
          </div>

          {/* Description with fixed height and line clamp */}
          <div className="mt-2 flex-grow">
            <p className="text-sm text-gray-600 line-clamp-3 group-hover:text-gray-800 transition-colors duration-300">
              {description || "No description available"}
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
};

const DashboardCardSkeleton: React.FC = () => {
  return (
    <Card className="relative flex h-48 w-full flex-col overflow-hidden rounded-xl">
      <div className="relative overflow-hidden bg-white px-6 pb-8 pt-6 ring-gray-900/5 sm:mx-auto sm:max-w-sm sm:rounded-lg sm:px-10">
        <Skeleton className="absolute top-6 z-0 h-10 w-10 rounded-full" />

        {/* Menu skeleton */}
        <Skeleton className="absolute top-6 right-6 h-8 w-8 rounded-full" />

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
