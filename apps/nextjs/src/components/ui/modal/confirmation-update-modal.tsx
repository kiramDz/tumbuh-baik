import React from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Save } from "lucide-react";

interface ConfirmationUpdateModalProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  onConfirm: () => void;
  datasetName: string;
  isUpdating: boolean;
}

export function ConfirmationUpdateModal({
  isOpen,
  setIsOpen,
  onConfirm,
  datasetName,
  isUpdating,
}: ConfirmationUpdateModalProps) {
  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2">
            <Save className="h-5 w-5" />
            Konfirmasi Perubahan Dataset
          </AlertDialogTitle>
          <AlertDialogDescription>
            Apakah Anda yakin ingin menyimpan perubahan pada dataset{" "}
            <strong>{datasetName}</strong>?
          </AlertDialogDescription>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel disabled={isUpdating}>Batal</AlertDialogCancel>
          <AlertDialogAction
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
            disabled={isUpdating}
          >
            {isUpdating ? "Menyimpan..." : "Simpan Perubahan"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
