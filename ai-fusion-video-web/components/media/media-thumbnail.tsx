"use client";
import { useEffect, useRef, useState } from "react";
import { ImageIcon, Loader2, RefreshCw, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage as Image } from "@/components/ui/safe-image";
import { mediaId } from "@/lib/api/media";
import { cn } from "@/lib/utils";
import { useMediaPreview } from "@/hooks/use-media-preview";
import { useMediaUrl } from "@/hooks/use-media-url";
import { SafeImage } from "./managed-image";

export function MediaThumbnail({
  source,
  kind,
  alt,
  fit = "contain",
  interactive = true,
  className,
}: {
  source: string;
  kind: "image" | "video";
  alt: string;
  fit?: "contain" | "cover";
  interactive?: boolean;
  className?: string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const managed = mediaId(source) !== null;
  const preview = useMediaPreview(source, visible && managed);
  const access = useMediaUrl(
    preview.view?.previewId ? `media://${preview.view.previewId}` : null,
  );
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const error =
    preview.error ??
    access.error ??
    (access.url && failedUrl === access.url
      ? "无法读取预览图片，请重试"
      : null);
  const loading = preview.loading || access.loading;
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: "200px" },
    );
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const Icon = kind === "video" ? Video : ImageIcon;
  const imageClass = cn(
    "h-full w-full",
    fit === "cover" ? "object-cover" : "object-contain",
  );
  return (
    <div
      ref={container}
      className={cn(
        "relative flex h-full w-full items-center justify-center",
        className,
      )}
      aria-busy={loading || undefined}
    >
      {!managed && kind === "image" ? (
        <SafeImage
          src={source}
          alt={alt}
          draggable={false}
          className={imageClass}
        />
      ) : preview.view?.status === "SUCCEEDED" && access.url && !error ? (
        <Image
          src={access.url}
          alt={alt}
          draggable={false}
          className={imageClass}
          width={preview.view.width!}
          height={preview.view.height!}
          onError={() => setFailedUrl(access.url)}
        />
      ) : error ? (
        <div
          className="flex flex-col items-center gap-2 px-3 text-center"
          title={error}
        >
          <Icon className="size-6 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">预览暂不可用</span>
          {interactive && (
            <Button
              variant="ghost"
              size="xs"
              className="nodrag nopan"
              onClick={(event) => {
                event.stopPropagation();
                setFailedUrl(null);
                preview.retry();
              }}
            >
              <RefreshCw />
              重试预览
            </Button>
          )}
        </div>
      ) : loading ? (
        <span
          role="status"
          className="flex items-center gap-2 text-xs text-muted-foreground"
        >
          <Loader2 className="size-4 animate-spin motion-reduce:animate-none" />
          准备预览
        </span>
      ) : (
        <Icon className="size-8 text-muted-foreground" />
      )}
      {kind === "video" &&
        preview.view?.status === "SUCCEEDED" &&
        access.url &&
        !error && (
          <span
            className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1 rounded-lg border border-border/30 bg-background/80 px-1.5 py-1 text-xs text-foreground"
            title={`预览帧取自 ${(preview.view.sampleAtMs / 1000).toFixed(2)} 秒`}
          >
            <Video className="size-3" />
            视频
          </span>
        )}
    </div>
  );
}
