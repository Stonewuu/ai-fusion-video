"use client";

import { ExternalLink, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { ManagedVideo } from "@/components/media/managed-video";
import { ManagedAudio } from "@/components/media/managed-audio";
import { ManagedMediaLink } from "@/components/media/managed-media-link";
import type { ComfyUiStoredOutput } from "@/lib/api/comfyui-workflow";

const roles: Record<string, string> = {
  primary: "主结果",
  cover: "封面",
  auxiliary: "附加结果",
};
export function ComfyUiTestOutput({
  output,
  index,
}: {
  output: ComfyUiStoredOutput;
  index: number;
}) {
  const label = `${roles[output.role] ?? output.role} ${index + 1}`;
  return (
    <div className="overflow-hidden rounded-lg border border-border/20 bg-background/70">
      <div className="flex aspect-video items-center justify-center overflow-hidden bg-muted/30">
        {output.mediaType === "image" ? (
          <SafeImage
            src={output.url}
            alt={label}
            className="size-full object-contain"
          />
        ) : output.mediaType === "video" ? (
          <ManagedVideo
            src={output.url}
            controls
            preload="metadata"
            className="size-full object-contain"
            aria-label={label}
          />
        ) : output.mediaType === "audio" ? (
          <ManagedAudio
            src={output.url}
            controls
            preload="metadata"
            className="w-full"
            aria-label={label}
          />
        ) : (
          <FileText className="size-8 text-muted-foreground" />
        )}
      </div>
      <div className="flex items-center justify-between gap-2 p-3">
        <span className="text-xs">
          {label} · {(output.size / 1024).toFixed(1)} KB
        </span>
        <Button
          variant="ghost"
          size="icon-sm"
          nativeButton={false}
          aria-label={`下载${label}`}
          render={<ManagedMediaLink href={output.url} download />}
        >
          <ExternalLink />
        </Button>
      </div>
    </div>
  );
}
