import * as React from "react";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function YearlyOption() {
  return (
    <>
      <Select>
        <SelectTrigger className="w-[280px]">
          <SelectValue placeholder="2025" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="est">2024</SelectItem>
          <SelectItem value="cst">2023</SelectItem>
          <SelectItem value="mst">2022</SelectItem>
          <SelectItem value="mst">2021</SelectItem>
          <SelectItem value="mst">2020</SelectItem>
        </SelectContent>
      </Select>
    </>
  );
}
