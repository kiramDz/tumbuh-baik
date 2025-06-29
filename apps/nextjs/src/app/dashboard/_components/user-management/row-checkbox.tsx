import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

export const RowCheckbox = ({ className, ...props }: ComponentProps<"input">) => {
  return <input type="checkbox" className={cn("block relative z-10", className)} {...props} />;
};
