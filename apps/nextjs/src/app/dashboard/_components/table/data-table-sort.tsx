import { ArrowDownAZ, ArrowUpZA } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface DataTableSortProps {
  currentSortOrder: "asc" | "desc";
  onSortChange: (sortOrder: "asc" | "desc") => void;
}

export function DataTableSort({ currentSortOrder, onSortChange }: DataTableSortProps) {
  return (
    <div className="flex items-center gap-2">
      <Select value={currentSortOrder} onValueChange={onSortChange}>
        <SelectTrigger className="w-[120px] h-8">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="desc">
            <div className="flex items-center gap-1">
              <ArrowDownAZ size={16} />
              <span>Terbaru</span>
            </div>
          </SelectItem>
          <SelectItem value="asc">
            <div className="flex items-center gap-1">
              <ArrowUpZA size={16} />
              <span>Terlama</span>
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
