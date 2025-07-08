"use client";

import { Table, TableHeader, TableBody, TableHead, TableRow, TableCell } from "@/components/ui/table";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { MoreVertical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

export default function RecentTable() {
  // Data dummy untuk tampilan UI
  const recentFiles = [
    {
      _id: "1",
      name: "example-file-1.jpg",
      category: "citra-satelit",
      size: 1024 * 512, // 512KB
      userInfo: {
        name: "John Doe",
      },
    },
    {
      _id: "2",
      name: "example-file-2.pdf",
      category: "daily-weather",
      size: 1024 * 1024 * 2.5, // 2.5MB
      userInfo: {
        name: "Jane Smith",
      },
    },
    {
      _id: "3",
      name: "example-file-3.csv",
      category: "bmkg-station",
      size: 1024 * 250, // 250KB
      userInfo: {
        name: "Robert Johnson",
      },
    },
  ];

  // Fungsi helper untuk format ukuran file
  const formatFileSize = (size: number) => {
    return size < 1024 ? `${size} B` : size < 1024 * 1024 ? `${(size / 1024).toFixed(2)} KB` : `${(size / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="flex flex-1 flex-col space-y-4">
      <div className="relative flex flex-1">
        <div className="flex overflow-scroll rounded-md border md:overflow-auto w-full">
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
                    <TableCell>{file.category}</TableCell>
                    <TableCell>{formatFileSize(file.size)}</TableCell>
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
