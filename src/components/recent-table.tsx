"use client";

import { useEffect, useState } from "react";
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { MoreVertical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getRecentFiles } from "@/lib/fetch/files.fetch";
import { IFile } from "@/lib/database/schema/file.model";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
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

  const { isLoading, error } = useQuery<RecentFile[]>({
    queryKey: ["recentFiles"],
    queryFn: getRecentFiles,
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });

  // Gunakan useMutation untuk update data secara otomatis
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
  }, []);

  if (isLoading) return <div>Loading...</div>;
  if (error) {
    toast("Error loading recent files");
    return <div>Error loading data.</div>;
  }

  return (
    <div className="flex flex-1 flex-col space-y-4">
      <div className="relative flex flex-1">
        <div className="flex overflow-scroll rounded-md border md:overflow-auto  w-full">
          <ScrollArea className="flex-1">
            <Table className="relative w-full">
              <TableHeader>
                <TableRow>
                  <TableHead className="min-w-[300px]">Name</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Owner</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentFiles.map((file) => (
                  <TableRow key={file._id}>
                    <TableCell>{file.name}</TableCell>
                    <TableCell>{String(file.category)}</TableCell>
                    <TableCell>{file.size < 1024 ? `${file.size} B` : file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(2)} KB` : `${(file.size / (1024 * 1024)).toFixed(2)} MB`}</TableCell>
                    <TableCell>{file.userInfo.name}</TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" className="h-8 w-8 p-0">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>Download</DropdownMenuItem>
                          <DropdownMenuItem>Rename</DropdownMenuItem>
                          <DropdownMenuItem>Delete</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
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
