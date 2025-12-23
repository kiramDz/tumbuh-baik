import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { ElementType } from "react";
import { EllipsisVertical } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowRight } from "lucide-react";
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
    <Link href={href} className="group block">
      <Card className="h-full transition-all duration-300 hover:shadow-lg hover:border-primary/50">
        <CardContent className="p-6">
          <div className="flex flex-col gap-4">
            {/* Icon Container */}
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary transition-all duration-300 group-hover:bg-primary group-hover:text-primary-foreground">
                <Icon className="h-6 w-6" />
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground transition-all duration-300 group-hover:translate-x-1 group-hover:text-primary" />
            </div>

            {/* Content */}
            <div className="space-y-2">
              <h3 className="font-semibold text-lg leading-tight line-clamp-1 transition-colors group-hover:text-primary">
                {title}
              </h3>
              <p className="text-sm text-muted-foreground line-clamp-2">
                {description}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
};

const DashboardCardSkeleton: React.FC = () => {
  return (
    <Card className="h-full">
      <CardContent className="p-6">
        <div className="flex flex-col gap-4">
          {/* Icon Container Skeleton */}
          <div className="flex items-center justify-between">
            <Skeleton className="h-12 w-12 rounded-lg" />
            <Skeleton className="h-5 w-5 rounded" />
          </div>

          {/* Content Skeleton */}
          <div className="space-y-2">
            <Skeleton className="h-6 w-3/4" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export { DashboardCard, DashboardCardSkeleton };
