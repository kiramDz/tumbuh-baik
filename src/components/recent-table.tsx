

"use client";

import { useEffect, useState } from "react";
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { getRecentFiles } from "@/lib/fetch/files.fetch";
import { IFile } from "@/lib/database/schema/file.model";

interface RecentFile {
  _id: string;
  name: string;
  category: string;
  size: number;
  userInfo: {
    name: string;
  };
}


export default function RecentTable() {
  const queryClient = useQueryClient();
  const [recentFiles, setRecentFiles] = useState<IFile[]>([]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["recentFiles"],
    queryFn: getRecentFiles,
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });
  // Gunakan useMutation seperti di category untuk konsistensi
  // untuk update data otomatis
  const mutation = useMutation({
    mutationFn: getRecentFiles,
    onSuccess: (newData) => {
      // Perbarui cache dan state lokal
      queryClient.setQueryData(["recentFiles"], newData);
      setRecentFiles(newData.files);
    },
    onError: (e) => {
      toast("Error loading recent files", { description: e.message });
    },
  });
//trigger mutation, jika ada perubahan data
  useEffect(() => {
    mutation.mutate();
  }, [data]);

  if (isLoading) return <div>Loading...</div>;
  if (error) {
    toast("Error loading recent files");
    return <div>Error loading data.</div>;
  }

  const filesss = data?.files as IFile[];
  console.log("recente file",recentFiles);
  console.log("filess",filesss)

  return (
    <div className="flex flex-1 flex-col space-y-4">
      <div className="relative flex flex-1">
        <div className="flex overflow-scroll rounded-md border md:overflow-auto  w-full">
          <ScrollArea className="flex-1">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  {/* <TableHead>Category</TableHead> */}
                  <TableHead>Size</TableHead>
                  <TableHead>Owner</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filesss.map((file) => (
                  <TableRow key={file._id}>
                    <TableCell>{file.name}</TableCell>
                    <TableCell>{file.category}</TableCell>
                    <TableCell>{file.size}</TableCell>
                    <TableCell>{file.userInfo.name}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
