"use client";
import { SafeImage } from "@/components/media/managed-image";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
export function ImageLightbox({
  src,
  alt,
  onClose,
}: {
  src: string;
  alt: string;
  onClose: () => void;
}) {
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent
        className="max-h-[calc(100vh-2rem)] overflow-hidden sm:max-w-5xl"
        aria-describedby={undefined}
      >
        <DialogHeader>
          <DialogTitle>{alt}</DialogTitle>
        </DialogHeader>
        <SafeImage
          src={src}
          alt={alt}
          className="mx-auto max-h-[75vh] max-w-full object-contain"
        />
      </DialogContent>
    </Dialog>
  );
}
