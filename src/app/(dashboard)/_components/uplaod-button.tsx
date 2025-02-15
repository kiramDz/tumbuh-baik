"use client";

import { Button } from "@/components/ui/button";
import { RiFileAddFill } from "@remixicon/react";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import axios from "axios";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { P } from "@/components/custom/p";
import { IFile } from "@/lib/database/schema/file.model";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useId } from "react";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const formSchema = z.object({
  category: z.enum(["citra-satelit", "daily-weather", "bmkg-station", "sea-​​temperature", ""]),
  file: z.custom<FileList>((val) => val instanceof FileList, "Required").refine((files) => files.length > 0, `Required`),
});

const UploadButton = () => {
  const queryClient = useQueryClient();
  const id = useId();
  const [fileProgress, setFileProgress] = useState<Record<string, number>>({});
  const [isUploading, setIsUploading] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      file: undefined,
    },
  });
  //Fungsi yang dioper ke useMutation
  // saat diapnggil oleh mutation, dia kaan mengembalikan data respon
  async function uploadFile(formData: FormData) {
    const file = formData.get("file") as File; // Pastikan ini adalah File
    const fileName = file ? file.name : "unknown";
    const res = await axios.post("/api/v1/files/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (progressEvent) => {
        const total = progressEvent.total || 1;
        const loaded = progressEvent.loaded;
        const percent = Math.round((loaded / total) * 100);

        setFileProgress((prev) => ({
          ...prev,
          [fileName]: percent,
        }));
      },
    });

    return res.data;
  }
  //Mengelola pemanggilan uploadFile dan menangani hasilnya.
  const mutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: (newData) => {
      queryClient.setQueryData(["files", newData.category], (oldData: { files: IFile[] }) => {
        // Pastikan oldData tidak undefined
        if (!oldData) {
          oldData = { files: [] };
        }
        const uploadedFile = newData.file;
        const oldFile = oldData.files || [];

        const newMergeFiles = [uploadedFile, ...oldFile];

        const updatedData = { ...oldData, files: newMergeFiles };

        return updatedData;
      });

      toast(newData?.message, {
        description: newData?.description,
      });
    },
    onError: (c) => {
      toast(c.name, {
        description: c.message,
      });
    },
    onSettled: () => {
      setIsUploading(false);
    },
  });

  // async function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
  //   const files = Array.from(e.target.files || []);

  //   if (!files.length) {
  //     toast("No file selected", {
  //       description: "Please select a file to upload",
  //     });

  //     return;
  //   }

  //   const progressMap: Record<string, number> = {};

  //   files.forEach((file) => {
  //     progressMap[file.name] = 0;
  //   });

  //   setFileProgress(progressMap);

  //   setIsUploading(true);

  //   await Promise.all(files.map((file) => mutation.mutateAsync(file)));

  //   e.target.value = "";
  // }

  return (
    <>
      {isUploading &&
        Object.entries(fileProgress).map(([fileName, progress], i) => (
          <TooltipProvider key={i}>
            <Tooltip>
              <TooltipTrigger>
                <div className="relative size-9 rounded-full flex items-center justify-center drop-shadow-md cursor-default animate-pulse">
                  <svg className="absolute w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                    <circle className="text-gray-300" stroke="currentColor" strokeWidth="4" fill="transparent" r="16" cx="18" cy="18" />
                    <circle className="text-primary" stroke="currentColor" strokeWidth="4" strokeLinecap="round" fill="transparent" r="16" cx="18" cy="18" strokeDasharray="100" strokeDashoffset={100 - progress} />
                  </svg>
                  <P className="text-xs text-primary font-bold">{progress}%</P>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <P className="text-xs font-bold">{fileName}</P>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      <Dialog>
        <DialogTrigger asChild>
          <Button className="bg-black shadow-none rounded-sm">
            <RiFileAddFill /> Upload
          </Button>
        </DialogTrigger>
        <DialogContent className="px-8 sm:max-w-[425px]">
          <DialogHeader>
            {" "}
            <DialogTitle>Upload files</DialogTitle>
            <DialogDescription>Drag and drop your files here or click to browse.</DialogDescription>
          </DialogHeader>
          <Form {...form}>
            <form
              className="space-y-3"
              onSubmit={form.handleSubmit((data) => {
                console.log("Submitting data:", data);
                if (!data.file || !data.category) {
                  toast("Please select a file and category", { description: "Both fields are required" });
                  return;
                }

                const formData = new FormData();
                formData.append("category", data.category);
                Array.from(data.file).forEach((file) => formData.append("file", file));
                console.log("FormData before sending:", formData.get("category"), formData.get("file"));
                mutation.mutate(formData);
              })}
            >
              <FormField
                control={form.control}
                name="file"
                render={({ field }) => (
                  <div className="space-y-2 w-full">
                    <Label htmlFor={id}>File input</Label>
                    <Input id={id} className="p-0 pe-3 file:me-3 file:border-0 file:border-e" type="file" multiple onChange={(e) => field.onChange(e.target.files)} />
                  </div>
                )}
              />
              <FormField
                control={form.control}
                name="category"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Category</FormLabel>
                    <Select {...field} onValueChange={field.onChange}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select category" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="citra-satelit">Citra Satelit</SelectItem>
                        <SelectItem value="bmkg-station">BMKG Station</SelectItem>
                        <SelectItem value="daily-weather">Daily Weather</SelectItem>
                        <SelectItem value="temperatur-laut">Temperatur Laut</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button type="submit">Save changes</Button>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* <input type="file" className="hidden" id="file-upload" multiple onChange={handleFileChange} /> */}
    </>
  );
};

export default UploadButton;
