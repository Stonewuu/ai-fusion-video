"use client";

import { Images, Music2 } from "lucide-react";
import { resolveMediaUrl } from "@/lib/api/client";
import { SafeImage } from "@/components/media/managed-image";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import { ManagedMediaLink } from "@/components/media/managed-media-link";
import type { GenerationTask, SimpleAttachment, WorkbenchMode } from "./generation-types";
import type { GenerationMediaPreview } from "./generation-media-preview-dialog";
import { taskReferences } from "./generation-reference-input";

interface GenerationHistoryReferencesProps {
  mode: WorkbenchMode;
  task: GenerationTask;
  onPreview: (preview: GenerationMediaPreview) => void;
}

export function GenerationHistoryReferences({ mode, task, onPreview }: GenerationHistoryReferencesProps) {
  let references: SimpleAttachment[];
  try {
    references = taskReferences(mode, task);
  } catch {
    return <p role="alert" className="text-xs text-destructive">参考素材记录无法读取</p>;
  }
  if (references.length === 0) {
    return null;
  }
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="flex shrink-0 items-center gap-1 text-[10px] text-muted-foreground">
        <Images className="size-3" />参考
      </span>
      <div className="flex min-w-0 items-center gap-2 overflow-x-auto p-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {references.map((reference, index) => {
          const url = resolveMediaUrl(reference.url);
          if (!url) {
            return null;
          }
          const className = "grid size-9 shrink-0 place-items-center overflow-hidden rounded-lg border border-border/40 bg-muted/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50";
          if (reference.mediaType === "audio") {
            return (
              <ManagedMediaLink key={`${reference.kind}-${index}`} href={reference.url} target="_blank" rel="noreferrer"
                title="播放参考音频" aria-label={`播放参考音频 ${index + 1}`} className={className}>
                <Music2 className="size-4 text-muted-foreground" />
              </ManagedMediaLink>
            );
          }
          const type = reference.mediaType;
          return (
            <button key={`${reference.kind}-${index}`} type="button" title={`查看${reference.label}`}
              aria-label={`查看${reference.label} ${index + 1}`} className={className}
              onClick={() => onPreview({ type, url, title: `${reference.label} ${index + 1}` })}>
              {type === "video" ? (
                <MediaThumbnail source={reference.url} kind="video" alt={reference.label} fit="cover" compact interactive={false} />
              ) : (
                <SafeImage src={url} alt={reference.label} className="h-full w-full object-cover" fallbackType="image" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
