"use client";

import { Download, PlayCircle } from "lucide-react";
import { resolveMediaUrl } from "@/lib/api/client";

interface VideoPreviewDialogProps {
  open: boolean;
  title: string;
  videoUrl: string | null;
  onClose: () => void;
}

export function VideoPreviewDialog({
  open,
  title,
  videoUrl,
  onClose,
}: VideoPreviewDialogProps) {
  const resolvedVideoUrl = resolveMediaUrl(videoUrl) ?? videoUrl;

  if (!open || !resolvedVideoUrl) {
    return null;
  }

  return (
    <div
      onClick={onClose}
      className="modal-overlay fixed inset-0 z-50 flex items-center justify-center p-4"
    >
      <div
        onClick={(event) => event.stopPropagation()}
        className="modal-surface relative w-full max-w-4xl overflow-hidden rounded-2xl border border-border/40"
      >
        <div className="flex items-center justify-between border-b border-border/20 px-4 py-3">
          <h3 className="flex items-center gap-2 text-sm font-semibold">
            <PlayCircle className="h-4 w-4 text-emerald-500" />
            {title}
          </h3>
          <div className="flex items-center gap-2">
            <a
              href={resolvedVideoUrl}
              target="_blank"
              rel="noreferrer"
              download
              className="inline-flex items-center gap-1.5 rounded-lg border border-border/30 bg-muted/20 px-3 py-1.5 text-xs transition-colors hover:bg-muted/40"
            >
              <Download className="h-3.5 w-3.5" />
              下载视频
            </a>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-border/30 bg-muted/20 px-3 py-1.5 text-xs transition-colors hover:bg-muted/40"
            >
              关闭
            </button>
          </div>
        </div>
        <video
          src={resolvedVideoUrl}
          controls
          autoPlay
          playsInline
          className="w-full max-h-[75vh] bg-black"
        />
      </div>
    </div>
  );
}
