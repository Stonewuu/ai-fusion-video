"use client";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { SafeImage } from "@/components/media/managed-image";
import { ManagedVideo } from "@/components/media/managed-video";
import { ManagedAudio } from "@/components/media/managed-audio";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";

export function CanvasMediaCompare({
  media,
  onClose,
}: {
  media: CanvasMediaPreview[];
  onClose: () => void;
}) {
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-6xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>素材对比</DialogTitle>
        </DialogHeader>
        <div className="grid min-h-0 gap-4 overflow-y-auto md:grid-cols-2">
          {media.map((item) => (
            <section key={`${item.title}:${item.source}`} className="space-y-3">
              <h3 className="text-sm font-medium">{item.title}</h3>
              {item.kind === "image" && (
                <SafeImage
                  src={item.source}
                  alt={item.title}
                  className="max-h-[65vh] w-full object-contain"
                />
              )}
              {item.kind === "video" && (
                <ManagedVideo
                  src={item.source}
                  controls
                  playsInline
                  preload="metadata"
                  className="max-h-[65vh] w-full"
                />
              )}
              {item.kind === "audio" && (
                <ManagedAudio
                  src={item.source}
                  controls
                  preload="metadata"
                  className="w-full"
                />
              )}
            </section>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
