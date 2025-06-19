"use client";

import { P } from "@/components/custom/p";
import { IFile } from "@/lib/database/schema/file.model";
import { getFiles } from "@/lib/fetch/files.fetch";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Image from "next/image";
import { useEffect, useState } from "react";
import { useInView } from "react-intersection-observer";
import { toast } from "sonner";
import MainTable from "../../../_components/table/main-table";

import { DataTableSkeleton } from "@/app/dashboard/_components/data-table-skeleton";

interface PageFilesProps {
  category: string;
}

const PageCategory = ({ category }: PageFilesProps) => {
  const { inView } = useInView();
  const [currentPage, setCurrentPage] = useState(1);
  const [isPageFull, setIsPageFull] = useState(false);
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["files", category, currentPage],
    queryFn: async () => await getFiles({ category, currentPage }),
    refetchOnMount: false,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
  });

  const mutation = useMutation({
    mutationFn: ({ category, page }: { category: string; page: number }) => getFiles({ category, currentPage: page }),
    onSuccess: (newData) => {
      if (currentPage === newData.totalPages) {
        setIsPageFull(true);
      }

      queryClient.setQueryData(["files", category], (oldData: unknown) => {
        const oldFiles = (oldData as { files: IFile[] })?.files || [];
        const newFiles = (newData.files as IFile[]) || [];

        // Ensure no duplicates using a Set or filtering by `_id`
        const mergedFiles = [...oldFiles, ...newFiles.filter((newFile) => !oldFiles.some((oldFile) => oldFile._id === newFile._id))];

        return {
          files: mergedFiles,
          total: newData.totalFiles,
          currentPage: newData.currentPage,
          totalPages: newData.totalPages,
        };
      });
    },
    onError(e) {
      toast(e.name, {
        description: e.message,
      });
    },
  });

  useEffect(() => {
    if (currentPage === data?.totalPages) {
      setIsPageFull(true);

      return;
    }
    if (inView && !isPageFull) {
      setCurrentPage((prev) => {
        const nextPage = prev + 1;

        mutation.mutateAsync({ category, page: nextPage });

        return nextPage;
      });
    }
  }, [inView, data]);

  if (isLoading) return <DataTableSkeleton columnCount={7} filterCount={2} cellWidths={["10rem", "30rem", "10rem", "10rem", "6rem", "6rem", "6rem"]} shrinkZero />;

  if (error)
    return (
      <P size="large" weight="bold">
        Error: {error.message}
      </P>
    );

  const files = data.files as IFile[];

  if (files?.length === 0)
    return (
      <div className="w-full h-[500px] flex items-center justify-center flex-col">
        <h2 className="text-xl font-semibold">File not found</h2>
        <Image src="/file-not-found.png" width={400} height={400} alt="not-found" />
      </div>
    );

  return (
    <>
      <div className="container mx-auto p-0">
        <MainTable />
      </div>
    </>
  );
};

export default PageCategory;
