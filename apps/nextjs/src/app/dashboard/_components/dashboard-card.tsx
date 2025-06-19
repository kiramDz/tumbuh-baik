import Link from "next/link";
import { Card } from "@/components/ui/card"; // Assuming you have a Card component from your UI library
import { ElementType } from "react";

interface DashboardCardProps {
  href: string;
  icon: ElementType; 
  title: string; // The main title of the card
  description: string; // The descriptive text below the title
}

const DashboardCard: React.FC<DashboardCardProps> = ({ href, icon: Icon, title, description }) => {
  return (
    <Link href={href} passHref>
      <Card className="relative flex h-48 w-full flex-col overflow-hidden rounded-xl">
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
      </Card>
    </Link>
  );
};

export default DashboardCard;
