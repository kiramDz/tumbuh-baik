"use client";

import * as React from "react";
import Image from "next/image";
import { Upload, FileCheck, Trash, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

interface FileUploaderProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  accept?: string;
  maxSize?: number; // in MB
  onFileSelect?: (file: File | null) => void;
  selectedFile?: File | null;
  className?: string;
  loading?: boolean;
}

export function FileUploader({
  accept = ".csv,.json,.xlsx",
  maxSize = 50,
  onFileSelect,
  selectedFile,
  className,
  loading = false,
  ...props
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    validateAndSetFile(file);
  };

  const validateAndSetFile = (file: File | null) => {
    setError(null);

    if (!file) {
      if (onFileSelect) onFileSelect(null);
      return;
    }

    // Validate file type
    if (accept) {
      const acceptedTypes = accept.split(",");
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
    const file = e.dataTransfer.files?.[0] || null;
    validateAndSetFile(file);
  };

  const removeFile = () => {
    if (onFileSelect) onFileSelect(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split(".").pop()?.toLowerCase();
    return extension === "csv"
      ? "/image/csv.png" // ✅ Updated path
      : extension === "json"
      ? "/image/json.png" // ✅ Updated path
      : extension === "xlsx"
      ? "/image/xlsx.png" // ✅ Added XLSX support with correct path
      : "/file-not-found.png"; // Keep fallback as is
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
        {...props}
      />

      {/* Drag & drop area or selected file info */}
      {!selectedFile ? (
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
            <Loader2 className="h-8 w-8 text-muted-foreground animate-spin" />
          ) : (
            <Upload className="h-8 w-8 mb-2 text-muted-foreground" />
          )}
          <p className="text-sm text-muted-foreground">
            {isDragging
              ? "Lepas file di sini"
              : `Klik atau seret file ${accept.replace(
                  /,/g,
                  " atau "
                )} ke sini`}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Maksimum {maxSize}MB
          </p>
        </div>
      ) : (
        <div className="flex items-center justify-between rounded-lg border p-3">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 relative shrink-0">
              <Image
                src={getFileIcon(selectedFile.name)}
                alt="File icon"
                fill
                sizes="40px"
                className="object-contain"
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-foreground truncate">
                {selectedFile.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
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
            <Trash className="h-4 w-4 text-destructive" />
          </Button>
        </div>
      )}

      {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
    </div>
  );
}
