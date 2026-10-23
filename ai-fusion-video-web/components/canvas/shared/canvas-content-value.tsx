"use client";
import { useState } from "react";
import { Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { ResourceType } from "@/lib/api/resource-events";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { historyFieldLabel, historyText } from "@/lib/canvas/history-presentation";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";

export function CanvasContentValue({
  type,
  field,
  value,
}: {
  type: ResourceType;
  field: string;
  value: unknown;
}) {
  const [expanded, setExpanded] = useState(false);
  const [media, setMedia] = useState<CanvasMediaPreview | null>(null);
  const content = historyText(value);
  const kind = RESOURCE_FIELDS[type].find((item) => item.key === field)?.kind;
  const mediaValue = typeof value === "string" && !!value && (kind === "image" || kind === "video" || kind === "audio");
  return (
    <div className="min-w-0 space-y-2">
      {!mediaValue && <p className="whitespace-pre-wrap break-words text-sm">{expanded ? content : content.slice(0, 4000)}</p>}
      {!mediaValue && content.length > 4000 && (
        <Button variant="link" size="xs" onClick={() => setExpanded(!expanded)}>
          {expanded ? "收起全文" : `显示全文（${content.length} 字符）`}
        </Button>
      )}
      {typeof value === "string" &&
        value &&
        (kind === "image" || kind === "video" || kind === "audio") && (
          <Button
            variant="outline"
            size="xs"
            onClick={() =>
              setMedia({
                kind,
                source: value,
                title: historyFieldLabel(type, field),
              })
            }
          >
            <Eye />
            预览
          </Button>
        )}
      {media && (
        <CanvasMediaPreviewDialog
          media={media}
          onClose={() => setMedia(null)}
        />
      )}
    </div>
  );
}
