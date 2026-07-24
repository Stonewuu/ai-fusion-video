"use client";

import { Images } from "lucide-react";
import { resolveMediaUrl } from "@/lib/api/client";
import { SafeImage } from "@/components/ui/safe-image";
import type { GenerationTask } from "./generation-types";
import type { GenerationMediaPreview } from "./generation-media-preview-dialog";

interface ReferenceImage {
  url: string;
  label: string;
}

interface GenerationHistoryReferencesProps {
  task: GenerationTask;
  onPreview: (preview: GenerationMediaPreview) => void;
}

function parseUrls(value?: string | null) {
  if (!value) return [];
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
      : [];
  } catch {
    return [];
  }
}

function collectReferenceImages(task: GenerationTask): ReferenceImage[] {
  const references: ReferenceImage[] = [];
  if ("refImageUrls" in task) {
    parseUrls(task.refImageUrls).forEach((url) =>
      references.push({ url, label: "参考图" }),
    );
  } else {
    if (task.firstFrameImageUrl) {
      references.push({ url: task.firstFrameImageUrl, label: "首帧图" });
    }
    if (task.lastFrameImageUrl) {
      references.push({ url: task.lastFrameImageUrl, label: "尾帧图" });
    }
    parseUrls(task.referenceImageUrls).forEach((url) =>
      references.push({ url, label: "参考图" }),
    );
  }

  const seen = new Set<string>();
  return references.filter(({ url }) => {
    const normalized = url.trim();
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

export function GenerationHistoryReferences({
  task,
  onPreview,
}: GenerationHistoryReferencesProps) {
  const references = collectReferenceImages(task);
  if (references.length === 0) return null;

  return (
    <div className="mt-2.5 flex items-center gap-2">
      <span className="flex shrink-0 items-center gap-1 text-[10px] text-muted-foreground">
        <Images className="h-3 w-3" />
        参考
      </span>
      <div className="flex min-w-0 items-center gap-1.5 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {references.map((reference, index) => {
          const resolvedUrl = resolveMediaUrl(reference.url);
          if (!resolvedUrl) return null;
          return (
            <button
              key={`${reference.url}-${index}`}
              type="button"
              onClick={() =>
                onPreview({
                  type: "image",
                  url: resolvedUrl,
                  title: `${reference.label} ${index + 1}`,
                })
              }
              title={`查看${reference.label}`}
              aria-label={`查看${reference.label} ${index + 1}`}
              className="h-9 w-9 shrink-0 overflow-hidden rounded-lg border border-border/40 bg-muted/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            >
              <SafeImage
                src={resolvedUrl}
                alt={reference.label}
                className="h-full w-full object-cover"
                fallbackType="image"
              />
            </button>
          );
        })}
      </div>
    </div>
  );
}
