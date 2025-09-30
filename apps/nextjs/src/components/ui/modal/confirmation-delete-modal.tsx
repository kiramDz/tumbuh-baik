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
import { Trash } from "lucide-react";

interface ConfirmationDeleteModalProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  onConfirm: () => void;
  datasetName: string;
  collectionName: string;
  isDeleting: boolean;
}

export function ConfirmationDeleteModal({
  isOpen,
  setIsOpen,
  onConfirm,
  datasetName,
  collectionName,
  isDeleting,
}: ConfirmationDeleteModalProps) {
  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="text-destructive flex items-center gap-2">
            <Trash className="h-5 w-5" />
            Hapus Dataset
          </AlertDialogTitle>
          <AlertDialogDescription>
            Yakin ingin menghapus dataset <strong>{datasetName}</strong>?
          </AlertDialogDescription>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Batal</AlertDialogCancel>
          <AlertDialogAction
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            disabled={isDeleting}
          >
            {isDeleting ? "Menghapus..." : "Hapus Dataset"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
