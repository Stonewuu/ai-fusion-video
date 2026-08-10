"use client";

import { FileText, Headphones, Video, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/ui/safe-image";
import type { AssistantAttachment } from "./assistant-multimodal";

interface AssistantAttachmentStripProps {
  attachments: AssistantAttachment[];
  disabled?: boolean;
  onRemove: (id: string) => void;
}

export function AssistantAttachmentStrip({
  attachments,
  disabled,
  onRemove,
}: AssistantAttachmentStripProps) {
  if (!attachments.length) return null;

  return (
    <div className="flex gap-2 overflow-x-auto border-b border-border/20 px-2 py-2">
      {attachments.map((attachment) => (
        <div
          key={attachment.id}
          className="group relative flex h-14 min-w-36 max-w-48 items-center gap-2 rounded-lg border border-border/30 bg-muted/35 p-1.5"
          title={`${attachment.name} · ${attachment.transport.toUpperCase()}`}
        >
          {attachment.inputType === "image" && attachment.previewUrl ? (
            <SafeImage
              src={attachment.previewUrl}
              alt=""
              className="size-11 shrink-0 rounded-md object-cover"
            />
          ) : (
            <span className="grid size-11 shrink-0 place-items-center rounded-md bg-background/70 text-muted-foreground">
              {attachment.inputType === "video" ? <Video className="size-4" /> : null}
              {attachment.inputType === "audio" ? <Headphones className="size-4" /> : null}
              {attachment.inputType === "file" ? <FileText className="size-4" /> : null}
            </span>
          )}
          <span className="min-w-0 flex-1 leading-tight">
            <span className="block truncate text-[10px] font-medium">{attachment.name}</span>
            <span className="mt-1 block text-[9px] text-muted-foreground">
              {formatFileSize(attachment.size)} · {attachment.transport.toUpperCase()}
            </span>
          </span>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="absolute right-0.5 top-0.5 opacity-70 group-hover:opacity-100"
            disabled={disabled}
            aria-label={`移除 ${attachment.name}`}
            title="移除附件"
            onClick={() => onRemove(attachment.id)}
          >
            <X />
          </Button>
        </div>
      ))}
    </div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
