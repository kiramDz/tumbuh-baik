"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, useCallback } from "react";
import { AddDatasetMeta } from "@/lib/fetch/files.fetch";
import { parseFile } from "@/lib/parse-upload";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { Plus, Upload, FileSpreadsheet, Loader2, X } from "lucide-react";

export default function AddDatasetDialog() {
  const queryClient = useQueryClient();

  const [form, setForm] = useState({
    name: "",
    source: "",
    collectionName: "",
    description: "",
    status: "raw",
  });

  const [file, setFile] = useState<File | null>(null);
  const [open, setOpen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const { mutate, isPending } = useMutation({
    mutationKey: ["add-dataset"],
    mutationFn: AddDatasetMeta,
    onSuccess: () => {
      toast.success("Dataset berhasil ditambahkan");
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      setOpen(false);
      resetForm();
    },
    onError: () => {
      toast.error("Gagal menambahkan dataset");
    },
  });

  const resetForm = () => {
    setForm({ name: "", source: "", collectionName: "", description: "", status: "raw" });
    setFile(null);
  };

  const handleFileChange = useCallback((selectedFile: File | null) => {
    if (!selectedFile) return;

    const validTypes = [".csv", ".json"];
    const isValid = validTypes.some((ext) => selectedFile.name.toLowerCase().endsWith(ext));

    if (!isValid) {
      toast.error("Format tidak didukung. Gunakan file CSV atau JSON.");
      return;
    }

    setFile(selectedFile);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const droppedFile = e.dataTransfer.files[0];
      handleFileChange(droppedFile);
    },
    [handleFileChange]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!form.name || !form.source || !file) {
      return toast.error("Mohon lengkapi semua data wajib");
    }

    const fileType = file.name.endsWith(".json") ? "json" : "csv";
    const buffer = await file.arrayBuffer();
    const parsed = await parseFile({ fileBuffer: Buffer.from(buffer), fileType });

    mutate({
      name: form.name,
      source: form.source,
      fileType,
      collectionName: form.collectionName,
      description: form.description,
      status: form.status,
      records: parsed,
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="bg-gradient-to-r from-teal-600 to-emerald-600 hover:from-teal-700 hover:to-emerald-700 text-white">
          <Plus className="w-4 h-4 mr-2" />
          Tambah Dataset
        </Button>
      </DialogTrigger>

      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Tambah Dataset Baru</DialogTitle>
          <DialogDescription>
            Unggah file CSV atau JSON untuk menambahkan dataset baru.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* File Upload Area */}
          <div className="space-y-2">
            <Label>File Dataset</Label>
            {!file ? (
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className={`
                  relative border-2 border-dashed rounded-lg p-8
                  transition-colors cursor-pointer
                  ${isDragging
                    ? "border-primary bg-primary/5"
                    : "border-muted-foreground/25 hover:border-muted-foreground/50"
                  }
                `}
              >
                <input
                  type="file"
                  accept=".csv,.json"
                  onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className="flex flex-col items-center gap-2 text-center">
                  <div className="p-3 rounded-full bg-muted">
                    <Upload className="w-5 h-5 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">
                      Seret file ke sini atau klik untuk memilih
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      CSV atau JSON (maks. 10MB)
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 p-3 rounded-lg border bg-muted/30">
                <div className="p-2 rounded-lg bg-primary/10">
                  <FileSpreadsheet className="w-5 h-5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0"
                  onClick={() => setFile(null)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>

          {/* Form Fields */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="name">
                Nama Dataset <span className="text-destructive">*</span>
              </Label>
              <Input
                id="name"
                placeholder="Contoh: Data Cuaca 2024"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="source">
                Sumber Data <span className="text-destructive">*</span>
              </Label>
              <Input
                id="source"
                placeholder="Contoh: BMKG, NASA"
                value={form.source}
                onChange={(e) => setForm({ ...form, source: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="collectionName">Nama Koleksi</Label>
              <Input
                id="collectionName"
                placeholder="Opsional"
                value={form.collectionName}
                onChange={(e) => setForm({ ...form, collectionName: e.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="status">Status</Label>
              <Select
                value={form.status}
                onValueChange={(value) => setForm({ ...form, status: value })}
              >
                <SelectTrigger id="status">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="raw">Raw</SelectItem>
                  <SelectItem value="cleaned">Cleaned</SelectItem>
                  <SelectItem value="validated">Validated</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Deskripsi</Label>
            <Textarea
              id="description"
              placeholder="Deskripsi singkat tentang dataset ini..."
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
              rows={3}
            />
          </div>

          <DialogFooter className="gap-2 sm:gap-0">
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Batal
              </Button>
            </DialogClose>
            <Button type="submit" disabled={isPending || !file || !form.name || !form.source}>
              {isPending ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Menyimpan...
                </>
              ) : (
                "Simpan Dataset"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
