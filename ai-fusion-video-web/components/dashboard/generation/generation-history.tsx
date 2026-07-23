"use client";

import {
  ArrowDownToLine,
  Clock3,
  ImagePlus,
  Loader2,
  PackagePlus,
  RefreshCw,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import type { AiModel } from "@/lib/api/ai-model";
import { resolveMediaUrl } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/ui/safe-image";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type {
  GenerationHistoryEntry,
  GenerationResultItem,
  WorkbenchMode,
} from "./generation-types";
import {
  formatGenerationTime,
  generationSuggestions,
  getResultMediaUrl,
  getResultPreviewUrl,
  getTaskStatus,
} from "./generation-utils";

interface GenerationHistoryProps {
  mode: WorkbenchMode;
  entries: GenerationHistoryEntry[];
  models: AiModel[];
  loading: boolean;
  total: number;
  onRefresh: () => void;
  onLoadMore: () => void;
  onUseSuggestion: (prompt: string) => void;
  onReusePrompt: (prompt: string) => void;
  onUseReference: (item: GenerationResultItem) => void;
  onAddAsset: (item: GenerationResultItem, prompt: string) => void;
}

function HistoryResult({
  mode,
  item,
  prompt,
  taskFailed,
  taskError,
  onUseReference,
  onAddAsset,
}: {
  mode: WorkbenchMode;
  item: GenerationResultItem;
  prompt: string;
  taskFailed: boolean;
  taskError?: string | null;
  onUseReference: (item: GenerationResultItem) => void;
  onAddAsset: (item: GenerationResultItem, prompt: string) => void;
}) {
  const mediaUrl = resolveMediaUrl(getResultMediaUrl(item));
  const previewUrl = resolveMediaUrl(getResultPreviewUrl(item));
  const resultReady = Boolean(mediaUrl || previewUrl);
  const displayUrl = mediaUrl || previewUrl;

  if (!resultReady) {
    const failed = taskFailed || item.status === 3;
    const failureMessage = item.errorMsg || taskError || "未生成结果";
    const resultPlaceholder = (
      <article
        tabIndex={failed ? 0 : undefined}
        aria-label={failed ? `生成失败：${failureMessage}` : undefined}
        className={cn(
          "flex min-h-20 items-center gap-3 rounded-2xl border border-dashed border-border/45 bg-background/35 px-3 py-3",
          failed &&
            "cursor-help focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
        )}
      >
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
        <div className="min-w-0">
          <p className="text-xs font-medium">
            {failed ? "生成失败" : "正在生成"}
          </p>
          <p className="mt-0.5 line-clamp-1 text-[10px] text-muted-foreground">
            {failed ? failureMessage : "完成后自动显示"}
          </p>
        </div>
      </article>
    );

    if (!failed) return resultPlaceholder;

    return (
      <Tooltip>
        <TooltipTrigger render={resultPlaceholder} />
        <TooltipContent
          side="top"
          align="start"
          className="max-w-sm whitespace-pre-wrap break-words text-left leading-5"
        >
          {failureMessage}
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <article className="group overflow-hidden rounded-2xl border border-border/45 bg-background/58 shadow-[0_12px_32px_-28px_rgba(15,23,42,.45)]">
      <div className="relative overflow-hidden bg-muted/22">
        {mode === "image" ? (
          <a
            href={displayUrl || undefined}
            target="_blank"
            rel="noreferrer"
            title="打开原图"
            className="block"
          >
            <SafeImage
              src={displayUrl || undefined}
              alt="生成图片"
              className="aspect-square w-full object-contain transition-transform duration-300 group-hover:scale-[1.015]"
              fallbackType="image"
            />
          </a>
        ) : mediaUrl ? (
          <video
            controls
            preload="metadata"
            poster={previewUrl || undefined}
            src={mediaUrl}
            className="aspect-video w-full bg-black object-contain"
          />
        ) : previewUrl ? (
          <SafeImage
            src={previewUrl}
            alt="视频封面"
            className="aspect-video w-full object-cover"
            fallbackType="image"
          />
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

export function GenerationHistory({
  mode,
  entries,
  models,
  loading,
  total,
  onRefresh,
  onLoadMore,
  onUseSuggestion,
  onReusePrompt,
  onUseReference,
  onAddAsset,
}: GenerationHistoryProps) {
  const modelNames = new Map(models.map((model) => [model.id, model.name]));

  if (loading && entries.length === 0) {
    return (
      <div className="space-y-2.5">
        {[0, 1].map((item) => (
          <div
            key={item}
            className="h-28 animate-pulse rounded-2xl border border-border/35 bg-muted/20"
          />
        ))}
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="flex min-h-[320px] flex-col items-center justify-center px-4 text-center">
        <span className="grid h-14 w-14 place-items-center rounded-[20px] border border-border/40 bg-card text-primary shadow-sm">
          <Sparkles className="h-6 w-6" />
        </span>
        <h2 className="mt-4 text-lg font-semibold tracking-tight">从一句描述开始</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          {mode === "image" ? "描述画面，或加入参考图" : "描述场景、动作和镜头"}
        </p>
        <div className="mt-5 grid w-full max-w-2xl gap-2 sm:grid-cols-3">
          {generationSuggestions(mode).map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => onUseSuggestion(suggestion)}
              className="rounded-2xl border border-border/40 bg-card/55 px-3 py-3 text-left text-xs leading-5 text-muted-foreground transition-all hover:-translate-y-0.5 hover:border-primary/25 hover:bg-card hover:text-foreground hover:shadow-sm"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 px-1 py-0.5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">创作记录</h2>
          <span className="text-[10px] text-muted-foreground">{total}</span>
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={onRefresh}
          title="刷新记录"
          aria-label="刷新记录"
        >
          <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} />
        </Button>
      </div>

      {entries.map(({ task, items }) => {
        const status = getTaskStatus(task);
        const modelName = task.modelId ? modelNames.get(task.modelId) : undefined;
        const failureMessage =
          task.errorMsg ||
          items.find((item) => item.errorMsg)?.errorMsg ||
          "未返回失败原因";
        const statusBadge = (
          <span
            tabIndex={status.tone === "failed" ? 0 : undefined}
            aria-label={
              status.tone === "failed"
                ? `${status.label}：${failureMessage}`
                : undefined
            }
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5",
              status.tone === "success" &&
                "bg-emerald-500/10 text-emerald-600",
              status.tone === "failed" &&
                "cursor-help bg-destructive/10 text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
              status.tone === "running" && "bg-primary/10 text-primary",
              status.tone === "queued" && "bg-amber-500/10 text-amber-600",
            )}
          >
            {(status.tone === "running" || status.tone === "queued") && (
              <Clock3 className="h-2.5 w-2.5" />
            )}
            {status.label}
          </span>
        );

        return (
          <article
            key={task.id}
            className="rounded-[22px] border border-border/45 bg-card/68 p-3.5 shadow-[0_12px_36px_-32px_rgba(15,23,42,.48)]"
          >
            <header className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="line-clamp-2 text-[13px] leading-5 text-foreground/90">
                  {task.prompt}
                </p>
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
                  <span>{modelName || "生成模型"}</span>
                  <span>·</span>
                  <span>{formatGenerationTime(task.createTime)}</span>
                  {status.tone === "failed" ? (
                    <Tooltip>
                      <TooltipTrigger render={statusBadge} />
                      <TooltipContent
                        side="top"
                        align="start"
                        className="max-w-sm whitespace-pre-wrap break-words text-left leading-5"
                      >
                        {failureMessage}
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    statusBadge
                  )}
                </div>
              </div>
              <Button
                variant="ghost"
                size="xs"
                onClick={() => onReusePrompt(task.prompt)}
                title="复用提示词"
                className="shrink-0"
              >
                <RotateCcw className="h-3 w-3" />
                再次使用
              </Button>
            </header>

            {items.length > 0 ? (
              <div
                className={cn(
                  "mt-3 grid gap-2.5",
                  mode === "image"
                    ? "grid-cols-[repeat(auto-fill,minmax(180px,220px))]"
                    : "grid-cols-[repeat(auto-fill,minmax(260px,340px))]",
                )}
              >
                {items.map((item) => (
                  <HistoryResult
                    key={item.id}
                    mode={mode}
                    item={item}
                    prompt={task.prompt}
                    taskFailed={status.tone === "failed"}
                    taskError={task.errorMsg}
                    onUseReference={onUseReference}
                    onAddAsset={onAddAsset}
                  />
                ))}
              </div>
            ) : status.tone === "failed" ? (
              <p className="mt-2.5 rounded-xl bg-destructive/6 px-3 py-2 text-xs text-destructive">
                {task.errorMsg || "生成失败"}
              </p>
            ) : (
              <div className="mt-2.5 flex h-14 max-w-xs items-center gap-2.5 rounded-xl border border-dashed border-border/45 bg-background/28 px-3 text-xs text-muted-foreground">
                <Loader2 className="size-4 shrink-0 animate-spin" />
                {status.label}
              </div>
            )}
          </article>
        );
      })}

      {entries.length < total && (
        <div className="flex justify-center pb-2">
          <Button variant="outline" size="sm" onClick={onLoadMore} disabled={loading}>
            {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            加载更多
          </Button>
        </div>
      )}
    </div>
  );
}
