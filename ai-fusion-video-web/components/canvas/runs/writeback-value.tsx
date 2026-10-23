"use client";

import { useState } from "react";
import { Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";
import type { CanvasWritebackField } from "@/lib/canvas/types";

export function CanvasWritebackValue({
  kind,
  value,
  title,
}: {
  kind: CanvasWritebackField["valueType"];
  value: string | null;
  title: string;
}) {
  const [open, setOpen] = useState(false);
  if (!value) return <p className="text-sm text-muted-foreground">尚无内容</p>;
  if (kind === "text")
    return (
      <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/20 bg-background/70 p-3 font-sans text-sm leading-relaxed">
        {value}
      </pre>
    );
  return (
    <div className="space-y-2">
      {kind === "image" && (
        <SafeImage
          src={value}
          alt={title}
          draggable={false}
          className="max-h-52 w-full rounded-lg object-contain"
        />
      )}
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Eye />
        预览{title}
      </Button>
      {open && (
        <CanvasMediaPreviewDialog
          media={{ kind, source: value, title }}
          onClose={() => setOpen(false)}
        />
      )}
    </div>
  );
}
