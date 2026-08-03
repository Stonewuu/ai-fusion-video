"use client";

import type { SyntheticEvent } from "react";
import {
  ArrowDownToLine,
  Expand,
  ImagePlus,
  Loader2,
  PackagePlus,
  Play,
} from "lucide-react";
import { resolveMediaUrl } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/ui/safe-image";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { GenerationResultItem, WorkbenchMode } from "./generation-types";
import {
  getGenerationErrorSummary,
  getResultMediaUrl,
  getResultPreviewUrl,
} from "./generation-utils";
import type { GenerationMediaPreview } from "./generation-media-preview-dialog";

interface GenerationHistoryResultProps {
  mode: WorkbenchMode;
  item: GenerationResultItem;
  prompt: string;
  taskFailed: boolean;
  taskError?: string | null;
  onUseReference: (item: GenerationResultItem) => void;
  onAddAsset: (item: GenerationResultItem, prompt: string) => void;
  onPreview: (preview: GenerationMediaPreview) => void;
}

function seekToVideoPreviewFrame(event: SyntheticEvent<HTMLVideoElement>) {
  const video = event.currentTarget;
  const previewTime =
    Number.isFinite(video.duration) && video.duration > 0
      ? Math.min(0.05, video.duration / 2)
      : 0.01;
  try {
    video.currentTime = previewTime;
  } catch {
    // 某些流媒体源不支持 seek，浏览器仍会尝试显示默认首帧。
  }
}

export function GenerationHistoryResult({
  mode,
  item,
  prompt,
  taskFailed,
  taskError,
  onUseReference,
  onAddAsset,
  onPreview,
}: GenerationHistoryResultProps) {
  const mediaUrl = resolveMediaUrl(getResultMediaUrl(item));
  const previewUrl = resolveMediaUrl(getResultPreviewUrl(item));
  const resultReady = Boolean(mediaUrl || previewUrl);
  const displayUrl = mediaUrl || previewUrl;

  if (!resultReady) {
    const failed = taskFailed || item.status === 2;
    const failureMessage = item.errorMsg || taskError || "未生成结果";
    const placeholderContent = (
      <>
        <span
          className={cn(
            "grid size-9 shrink-0 place-items-center rounded-xl",
            failed
              ? "bg-destructive/8 text-destructive"
              : "bg-primary/8 text-primary",
          )}
        >
          {failed ? (
            <ImagePlus className="size-4" />
          ) : (
            <Loader2 className="size-4 animate-spin" />
          )}
        </span>
        <span className="min-w-0 text-left">
          <span className="block text-xs font-medium">
            {failed ? "生成失败" : "正在生成"}
          </span>
          <span className="mt-0.5 line-clamp-2 break-all text-[10px] text-muted-foreground">
            {failed
              ? getGenerationErrorSummary(failureMessage)
              : "完成后自动显示"}
          </span>
        </span>
      </>
    );

    if (failed) {
      return (
        <Tooltip>
          <TooltipTrigger
            render={
              <article
                tabIndex={0}
                aria-label="生成失败，悬停或聚焦查看完整错误"
                className={cn(
                  "flex min-h-20 w-full cursor-help items-center gap-3 rounded-2xl border border-dashed px-3 py-3",
                  "border-destructive/25 bg-destructive/4 transition-colors hover:bg-destructive/7",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                )}
              >
                {placeholderContent}
              </article>
            }
          />
          <TooltipContent className="max-w-[calc(100vw-2rem)] p-0 sm:max-w-lg">
            <div className="max-h-[50vh] min-w-0 overflow-y-auto whitespace-pre-wrap break-all px-3 py-1.5 text-left">
              {failureMessage}
            </div>
          </TooltipContent>
        </Tooltip>
      );
    }

    return (
      <article className="flex min-h-20 items-center gap-3 rounded-2xl border border-dashed border-border/45 bg-background/35 px-3 py-3">
        {placeholderContent}
      </article>
    );
  }

  return (
    <article className="group overflow-hidden rounded-2xl border border-border/45 bg-background/58 shadow-[0_12px_32px_-28px_rgba(15,23,42,.45)]">
      <div className="relative overflow-hidden bg-muted/22">
        {mode === "image" ? (
          <button
            type="button"
            onClick={() =>
              displayUrl &&
              onPreview({ type: "image", url: displayUrl, title: "生成图片" })
            }
            title="在当前页预览图片"
            className="relative block w-full cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            <SafeImage
              src={displayUrl || undefined}
              alt="生成图片"
              className="aspect-square w-full object-contain transition-transform duration-300 group-hover:scale-[1.015]"
              fallbackType="image"
            />
            <span className="absolute right-2 top-2 grid h-7 w-7 place-items-center rounded-lg bg-black/55 text-white opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
              <Expand className="h-3.5 w-3.5" />
            </span>
          </button>
        ) : mediaUrl ? (
          <button
            type="button"
            onClick={() =>
              onPreview({
                type: "video",
                url: mediaUrl,
                posterUrl: previewUrl || undefined,
                title: "生成视频",
              })
            }
            title="在当前页预览视频"
            className="relative block aspect-video w-full overflow-hidden bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            {previewUrl ? (
              <SafeImage
                src={previewUrl}
                alt="视频封面"
                className="h-full w-full object-contain"
                fallbackType="image"
              />
            ) : (
              <video
                src={mediaUrl}
                muted
                playsInline
                preload="metadata"
                aria-hidden="true"
                onLoadedMetadata={seekToVideoPreviewFrame}
                className="h-full w-full object-contain"
              />
            )}
            <span className="absolute inset-0 grid place-items-center bg-black/12">
              <span className="grid h-10 w-10 place-items-center rounded-full bg-black/60 text-white backdrop-blur-sm">
                <Play className="ml-0.5 h-4 w-4 fill-current" />
              </span>
            </span>
          </button>
        ) : previewUrl ? (
          <button
            type="button"
            onClick={() =>
              onPreview({
                type: "image",
                url: previewUrl,
                title: "视频封面",
              })
            }
            title="在当前页预览封面"
            className="block w-full cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            <SafeImage
              src={previewUrl}
              alt="视频封面"
              className="aspect-video w-full object-cover"
              fallbackType="image"
            />
          </button>
        ) : (
          <div className="grid aspect-video place-items-center text-xs text-muted-foreground">
            视频处理中
          </div>
        )}
      </div>

      <div className="flex items-center justify-between gap-1.5 px-2 py-1.5">
        <div className="flex min-w-0 items-center gap-0.5">
          <Button
            variant="ghost"
            size="xs"
            onClick={() => onUseReference(item)}
            disabled={!resultReady}
            title="作为下一次生成的参考"
          >
            <ImagePlus className="size-3" />
            作为参考
          </Button>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => onAddAsset(item, prompt)}
            disabled={!resultReady}
          >
            <PackagePlus className="size-3" />
            添加资产
          </Button>
        </div>
        {mediaUrl && (
          <a
            href={mediaUrl}
            download
            target="_blank"
            rel="noreferrer"
            title="下载"
            aria-label="下载结果"
            className="grid size-7 shrink-0 place-items-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <ArrowDownToLine className="h-3.5 w-3.5" />
          </a>
        )}
      </div>
    </article>
  );
}
