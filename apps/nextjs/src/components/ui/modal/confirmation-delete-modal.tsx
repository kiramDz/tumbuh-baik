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
import { Icons } from "@/app/dashboard/_components/icons";

interface ConfirmationDeleteModalProps {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  onConfirm: () => void;
  datasetName: string;
  collectionName: string;
  isDeleting: boolean;
  type?: "soft" | "permanent";
}

export function ConfirmationDeleteModal({
  isOpen,
  setIsOpen,
  onConfirm,
  datasetName,
  collectionName,
  isDeleting,
  type = "soft",
}: ConfirmationDeleteModalProps) {
  const isPermanent = type === "permanent";

  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle
            className={`flex items-center gap-2 ${isPermanent ? "text-red-600" : "text-destructive"}`}
          >
            {isPermanent ? (
              <Icons.alertCircle className="h-5 w-5" />
            ) : (
              <Icons.trash className="h-5 w-5" />
            )}
            {isPermanent ? "Hapus Permanen Dataset" : "Hapus Dataset"}
          </AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-2">
              {isPermanent ? (
                <>
                  <div>
                    Dataset <strong>{datasetName}</strong> akan dihapus secara
                    permanen dari sistem.
                  </div>
                  <div className="text-sm">
                    Collection:{" "}
                    <code className="bg-muted px-1 py-0.5 rounded">
                      {collectionName}
                    </code>
                  </div>
                </>
              ) : (
                <div>
                  Dataset <strong>{datasetName}</strong> akan dipindahkan ke
                  Recycle Bin.
                </div>
              )}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>

        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Batal</AlertDialogCancel>
          <AlertDialogAction
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
            className={
              isPermanent
                ? "bg-red-600 text-white hover:bg-red-700"
                : "bg-destructive text-destructive-foreground hover:bg-destructive/90"
            }
            disabled={isDeleting}
          >
            {isDeleting
              ? "Menghapus..."
              : isPermanent
                ? "Hapus Permanen"
                : "Hapus"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
