// components/table/sort.tsx
"use client";

import { useEffect, useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ArrowDownAZ, ArrowUpZA } from "lucide-react";

interface SortProps {
  onChange: (value: "asc" | "desc") => void;
}

export const Sort = ({ onChange }: SortProps) => {
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    onChange(sortOrder);
  }, [sortOrder]);

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground">Urutkan Tanggal:</span>
      <Select value={sortOrder} onValueChange={(val) => setSortOrder(val as "asc" | "desc")}>
        <SelectTrigger className="w-[120px]">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="desc">
            <div className="flex items-center gap-1">
              <ArrowDownAZ size={16} /> Terbaru
            </div>
          </SelectItem>
          <SelectItem value="asc">
            <div className="flex items-center gap-1">
              <ArrowUpZA size={16} /> Terlama
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};
