"use client";

import * as React from "react";
import Image from "next/image";
import { Icons } from "@/app/dashboard/_components/icons";
import { cn } from "@/lib/utils";
import { Button } from "./button";

interface FileUploaderProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  accept?: string;
  maxSize?: number; // in MB per file
  maxTotalSize?: number; // in MB total for multi-file
  onFileSelect?: (file: File | null) => void;
  onFilesSelect?: (files: File[]) => void;
  selectedFile?: File | null;
  selectedFiles?: File[];
  className?: string;
  loading?: boolean;
  multiple?: boolean;
  maxFiles?: number;
}

export function FileUploader({
  accept = ".csv,.json,.xlsx",
  maxSize = 50,
  maxTotalSize = 100,
  onFileSelect,
  onFilesSelect,
  selectedFile,
  selectedFiles = [],
  className,
  loading = false,
  multiple = false,
  maxFiles = 12,
  ...props
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (multiple) {
      validateAndSetFiles(files);
    } else {
      const file = files[0] || null;
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file: File | null) => {
    setError(null);

    if (!file) {
      if (onFileSelect) onFileSelect(null);
      return;
    }

    // Validate file type
    if (accept) {
      const acceptedTypes = accept.split(",").map((type) => type.trim());
      const fileType = `.${file.name.split(".").pop()?.toLowerCase()}`;
      if (!acceptedTypes.includes(fileType)) {
        setError(`File harus bertipe ${accept}`);
        if (onFileSelect) onFileSelect(null);
        return;
      }
    }

    // Validate file size
    if (maxSize && file.size > maxSize * 1024 * 1024) {
      setError(`File terlalu besar (maksimum ${maxSize}MB)`);
      if (onFileSelect) onFileSelect(null);
      return;
    }

    if (onFileSelect) onFileSelect(file);
  };

  const validateAndSetFiles = (newFiles: File[]) => {
    setError(null);
    if (newFiles.length === 0) {
      if (onFilesSelect) onFilesSelect([]);
      return;
    }

    // Combine with existing files if any
    const allFiles = [...selectedFiles, ...newFiles];

    // Check file count limit
    if (allFiles.length > maxFiles) {
      setError(`Maksimum ${maxFiles} file yang dapat dipilih`);
      return;
    }

    // Validate each new file
    const validFiles: File[] = [];
    let totalSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);

    for (const file of newFiles) {
      // Check for duplicates
      const isDuplicate = selectedFiles.some(
        (existingFile) =>
          existingFile.name === file.name && existingFile.size === file.size
      );

      if (isDuplicate) {
        setError(`File ${file.name} sudah dipilih`);
        return;
      }

      // Validate file type
      if (accept) {
        const acceptedTypes = accept.split(",").map((type) => type.trim());
        const fileType = `.${file.name.split(".").pop()?.toLowerCase()}`;
        if (!acceptedTypes.includes(fileType)) {
          setError(`File ${file.name} bukan tipe yang didukung (${accept})`);
          return;
        }
      }

      // Validate individual file size
      if (maxSize && file.size > maxSize * 1024 * 1024) {
        setError(`File ${file.name} terlalu besar (maksimum ${maxSize}MB)`);
        return;
      }
      validFiles.push(file);
      totalSize += file.size;
    }
    // Validate total size (50MB for multi-files)
    const maxTotalSizeBytes = maxTotalSize * 1024 * 1024;
    if (totalSize > maxTotalSizeBytes) {
      setError(`Total ukuran file terlalu besar (maksimum ${maxTotalSize}MB)`);
      return;
    }

    if (onFilesSelect) onFilesSelect([...selectedFiles, ...validFiles]);
  };
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files || []);

    if (multiple) {
      validateAndSetFiles(files);
    } else {
      const file = files[0] || null;
      validateAndSetFile(file);
    }
  };

  const removeFile = () => {
    if (onFileSelect) onFileSelect(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const removeFileFromMultiple = (indexToRemove: number) => {
    const newFiles = selectedFiles.filter(
      (_, index) => index !== indexToRemove
    );
    if (onFilesSelect) onFilesSelect(newFiles);
    if (inputRef.current) inputRef.current.value = "";
  };

  const clearAllFiles = () => {
    if (onFilesSelect) onFilesSelect([]);
    if (inputRef.current) inputRef.current.value = "";
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split(".").pop()?.toLowerCase();
    return extension === "csv"
      ? "/image/csv.png"
      : extension === "json"
      ? "/image/json.png"
      : extension === "xlsx"
      ? "/image/xlsx.png"
      : "/file-not-found.png";
  };

  const getTotalSize = () => {
    return selectedFiles.reduce((total, file) => total + file.size, 0);
  };

  const formatFileSize = (bytes: number) => {
    return (bytes / 1024 / 1024).toFixed(2);
  };

  return (
    <div className={cn("w-full", className)}>
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={handleFileChange}
        accept={accept}
        disabled={loading}
        multiple={multiple}
        {...props}
      />

      {/* Drag & drop area or selected file(s) info */}
      {(!multiple && !selectedFile) ||
      (multiple && selectedFiles.length === 0) ? (
        <div
          onClick={() => !loading && inputRef.current?.click()}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-all",
            isDragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/20 hover:border-muted-foreground/50",
            loading ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
            "h-32"
          )}
        >
          {loading ? (
            <Icons.spinner className="h-8 w-8 text-muted-foreground animate-spin" />
          ) : (
            <Icons.upload className="h-8 w-8 mb-2 text-muted-foreground" />
          )}
          <p className="text-sm text-muted-foreground text-center">
            {isDragging
              ? "Lepas file di sini"
              : multiple
              ? `Klik atau seret file ${accept.replace(/,/g, " atau ")} ke sini`
              : `Klik atau seret file ${accept.replace(
                  /,/g,
                  " atau "
                )} ke sini`}
          </p>
          <p className="text-xs text-muted-foreground mt-1 text-center">
            {multiple
              ? `Maksimum ${maxFiles} file, ${maxSize}MB per file, ${maxTotalSize}MB total`
              : `Maksimum ${maxSize}MB`}
          </p>
        </div>
      ) : multiple ? (
        // Multi-file display
        <div className="space-y-3">
          {/* Header with file count and total size */}
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium text-foreground">
              {selectedFiles.length} file terpilih (
              {formatFileSize(getTotalSize())} MB total)
            </div>
            {selectedFiles.length > 0 && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={clearAllFiles}
                disabled={loading}
                className="text-destructive hover:text-destructive"
              >
                <Icons.trash className="h-4 w-4 mr-1" />
                Hapus Semua
              </Button>
            )}
          </div>

          {/* File list */}
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {selectedFiles.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                className="flex items-center justify-between rounded-lg border p-3 bg-background"
              >
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  <div className="h-8 w-8 relative shrink-0">
                    <Image
                      src={getFileIcon(file.name)}
                      alt="File icon"
                      fill
                      sizes="32px"
                      className="object-contain"
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-foreground truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(file.size)} MB
                    </p>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => removeFileFromMultiple(index)}
                  disabled={loading}
                  className="shrink-0 ml-2"
                >
                  <Icons.closeX className="h-4 w-4 text-destructive" />
                </Button>
              </div>
            ))}
          </div>

          {/* Add more files button */}
          {selectedFiles.length < maxFiles && (
            <Button
              type="button"
              variant="outline"
              onClick={() => inputRef.current?.click()}
              disabled={loading}
              className="w-full"
            >
              <Icons.plus className="h-4 w-4 mr-2" />
              Tambah File Lainnya ({selectedFiles.length}/{maxFiles})
            </Button>
          )}
        </div>
      ) : (
        // Single file display (existing)
        <div className="flex items-center justify-between rounded-lg border p-3">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 relative shrink-0">
              <Image
                src={getFileIcon(selectedFile!.name)}
                alt="File icon"
                fill
                sizes="40px"
                className="object-contain"
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-foreground truncate">
                {selectedFile!.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {formatFileSize(selectedFile!.size)} MB
              </p>
            </div>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={removeFile}
            disabled={loading}
          >
            <Icons.trash className="h-4 w-4 text-destructive" />
          </Button>
        </div>
      )}

      {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
    </div>
  );
}
