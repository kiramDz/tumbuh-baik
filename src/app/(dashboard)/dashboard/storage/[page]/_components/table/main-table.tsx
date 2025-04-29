

import { useState } from "react";
import { bmkgColumns } from "./columns/bmkg-columns";
import { MainTableUI } from "./main-table-ui";
import { getBmkgData } from "@/lib/fetch/files.fetch";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";


export default function MainTable() {
  const [page, setPage] = useState(1);

  const { data, isLoading, error } = useQuery({
    queryKey: ["bmkgData", page],
    queryFn: () => getBmkgData(page),
    refetchOnWindowFocus: false,
  });

  console.log("Query data received:", data);
  console.log("Items to display:", data?.items);

  if (isLoading) return <div>Loading...</div>;

  if (error) {
    toast("Failed to load BMKG data");
    return <div>Error loading data.</div>;
  }
  console.log("Rendering table with data:", data?.items);
  return (
    <MainTableUI
      data={data?.items || []}
      columns={bmkgColumns}
      pagination={{
        currentPage: data?.currentPage || 1,
        totalPages: data?.totalPages || 1,
        total: data?.total || 0,
        onPageChange: (newPage) => setPage(newPage),
      }}
    />
  );
}
