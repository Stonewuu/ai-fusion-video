"use client";

import { Download } from "lucide-react";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { ManagedVideo } from "@/components/media/managed-video";
import { ManagedAudio } from "@/components/media/managed-audio";
import { downloadMedia } from "@/lib/api/media";
import { toastApiError } from "@/lib/api/toast-api-error";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { CanvasDocumentPreview } from "./document-preview";

export function CanvasMediaPreviewDialog({
  media,
  returnFocus,
  onClose,
}: {
  media: CanvasMediaPreview;
  returnFocus?: HTMLElement;
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
        className="flex max-h-[calc(100vh-2rem)] sm:max-w-5xl flex-col overflow-hidden"
        finalFocus={
          returnFocus
            ? () => (returnFocus.isConnected ? returnFocus : false)
            : undefined
        }
      >
        <DialogHeader className="shrink-0">
          <DialogTitle>{media.title}</DialogTitle>
        </DialogHeader>
        <div className="min-h-0 overflow-y-auto rounded-lg bg-muted/30">
          {media.kind === "image" ? (
            <SafeImage
              src={media.source}
              alt={media.title}
              className="max-h-[70vh] w-full object-contain"
            />
          ) : media.kind === "video" ? (
            <ManagedVideo
              src={media.source}
              controls
              playsInline
              className="max-h-[70vh] w-full"
            />
          ) : media.kind === "audio" ? (
            <div className="p-6">
              <ManagedAudio src={media.source} controls className="w-full" />
            </div>
          ) : (
            <CanvasDocumentPreview
              source={media.source}
              pdf={media.kind === "document"}
            />
          )}
        </div>
        <div className="flex shrink-0 justify-end gap-2">
          <Button
            variant="outline"
            onClick={() =>
              void downloadMedia(media.source, media.title).catch((error) =>
                toastApiError(error, "下载失败"),
              )
            }
          >
            <Download />
            下载
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
