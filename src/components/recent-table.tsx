"use client";

// Tbale sudah berhasil, hanya saja belum bisa menmapilan daya terbaru
/*
TES PROMPT : 
Coba abis ni perbaiki promt

Coba pakai pendekatan kita yg hadnya ingin menampilkan 10 data terbaru yg diupload melalui endpoint upload, data terabit akan ditampilkan di /recent-table.tsx 
Bantu saya memagut endpoint untuk get 10 data terbaru dan memasukkan ke table
Share : upload-button (file), dan Schema (table), upload endpoint (code)
*/
import { MoreVertical, ChevronDown, Folder } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
// import axios from "axios";
import { useQuery } from "@tanstack/react-query";
import { getRecentFiles } from "@/lib/fetch/files.fetch";

// interface FileItem {
//   name: string;
//   type: "Folder" | "Document";
//   size: string;
//   modified: string;
// }

interface FileItem {
  id: string;
  name: string;
  type: string;
  size: number;
  modified: string;
}

// const files: FileItem[] = [
//   {
//     name: "Dribbble Shots",
//     type: "Folder",
//     size: "48 MB",
//     modified: "09/04/2023 20:29",
//   },
//   {
//     name: "Invoice for Victor.pdf",
//     type: "Document",
//     size: "19 MB",
//     modified: "08/04/2023 20:29",
//   },
// ];

export default function RecentTable() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["recentFiles"],
    queryFn: async () => await getRecentFiles(),
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });

  const files = data?.files ?? [];
  console.log("API Response:", data);
  console.log("isi:", files);

  if (isLoading) return <p>Loading...</p>;
  if (error) return <p>Error loading files</p>;

  return (
    <div className="flex flex-1 flex-col space-y-4">
      <div className="relative flex flex-1">
        <div className="flex overflow-scroll rounded-md border md:overflow-auto  w-full">
          <ScrollArea className="flex-1">
            <Table className="relative w-full">
              <TableHeader>
                <TableRow>
                  <TableHead className="min-w-[300px]">
                    Name <ChevronDown className="ml-1 h-4 w-4 inline-block" />
                  </TableHead>
                  <TableHead>
                    Type <ChevronDown className="ml-1 h-4 w-4 inline-block" />
                  </TableHead>
                  <TableHead>File size</TableHead>
                  <TableHead>Last modified</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {files.map((file: FileItem) => (
                  <TableRow key={file.id}>
                    <TableCell className="font-medium text-black">
                      <div className="flex items-center gap-2">
                        {/* {file.type === "Folder" ? <Folder className="h-5 w-5 text-yellow-400" /> : <FileText className="h-5 w-5 text-red-400" />} */}
                        <Folder className="h-5 w-5 text-yellow-400" />
                        {file.name}tes
                      </div>
                    </TableCell>
                    <TableCell className="text-black">{file.type}</TableCell>
                    <TableCell>{(file.size / 1024 / 1024).toFixed(2)} MB</TableCell>
                    <TableCell>{new Date(file.modified).toLocaleString()}</TableCell>
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
