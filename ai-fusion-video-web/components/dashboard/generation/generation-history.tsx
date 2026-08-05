"use client";

import { useState } from "react";
import {
  Clock3,
  Loader2,
  RefreshCw,
  RotateCcw,
  Square,
  Sparkles,
} from "lucide-react";
import type { AiModel } from "@/lib/api/ai-model";
import { getModelDisplayParts } from "@/lib/model-display";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ModelVendorIcon } from "@/components/dashboard/model-vendor-icon";
import type {
  GenerationHistoryEntry,
  GenerationResultItem,
  WorkbenchMode,
} from "./generation-types";
import {
  formatGenerationDateTime,
  getGenerationErrorSummary,
  getGenerationParameterItems,
  generationSuggestions,
  getTaskStatus,
} from "./generation-utils";
import { GenerationHistoryResult } from "./generation-history-result";
import { GenerationHistoryReferences } from "./generation-history-references";
import {
  GenerationMediaPreviewDialog,
  type GenerationMediaPreview,
} from "./generation-media-preview-dialog";

function getParameterTone(label: string) {
  if (label === "比例" || label === "宽高比") {
    return "bg-sky-500/10 text-sky-700 dark:text-sky-300";
  }
  if (label === "分辨率" || label === "尺寸") {
    return "bg-violet-500/10 text-violet-700 dark:text-violet-300";
  }
  if (label === "数量") {
    return "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (label === "模式" || label === "时长") {
    return "bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  return "bg-primary/8 text-primary";
}

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
  onCancel: (taskId: string) => void;
  cancellingTaskIds: Set<string>;
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
  onCancel,
  cancellingTaskIds,
}: GenerationHistoryProps) {
  const [mediaPreview, setMediaPreview] = useState<GenerationMediaPreview | null>(null);
  const modelsById = new Map(models.map((model) => [model.id, model]));

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
        const model = task.modelId ? modelsById.get(task.modelId) : undefined;
        const modelDisplay = getModelDisplayParts(
          model,
          task.modelId ? `模型 #${task.modelId}` : "生成模型",
        );
        const parameterItems = getGenerationParameterItems(task, items);
        const failureMessage =
          task.errorMsg ||
          items.find((item) => item.errorMsg)?.errorMsg ||
          "未返回失败原因";
        const statusBadgeContent = (
          <>
            {(status.tone === "running" || status.tone === "queued") && (
              <Clock3 className="h-2.5 w-2.5" />
            )}
            {status.label}
          </>
        );
        const statusBadgeClassName = cn(
          "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5",
          status.tone === "success" &&
            "bg-emerald-500/10 text-emerald-600",
          status.tone === "failed" &&
            "cursor-pointer bg-destructive/10 text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
          status.tone === "running" && "bg-primary/10 text-primary",
          status.tone === "queued" && "bg-amber-500/10 text-amber-600",
        );
        const statusBadgeTrigger = (
          <span
            tabIndex={status.tone === "failed" ? 0 : undefined}
            aria-label={
              status.tone === "failed"
                ? "生成失败，悬停或聚焦查看完整错误"
                : undefined
            }
            className={statusBadgeClassName}
          >
            {statusBadgeContent}
          </span>
        );
        const statusBadge =
          status.tone === "failed" ? (
            <Tooltip>
              <TooltipTrigger render={statusBadgeTrigger} />
              <TooltipContent className="max-w-[calc(100vw-2rem)] p-0 sm:max-w-lg">
                <div className="max-h-[50vh] min-w-0 overflow-y-auto whitespace-pre-wrap break-all px-3 py-1.5 text-left">
                  {failureMessage}
                </div>
              </TooltipContent>
            </Tooltip>
          ) : (
            statusBadgeTrigger
          );

        return (
          <article
            key={task.id}
            className={cn(
              "rounded-[22px] border border-border/45 bg-card/68 p-3.5",
              "shadow-[0_12px_36px_-32px_rgba(15,23,42,.48)]",
            )}
          >
            <header className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="line-clamp-2 text-[13px] leading-5 text-foreground/90">
                  {task.prompt}
                </p>
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
                  {statusBadge}
                  <span aria-hidden="true">·</span>
                  <ModelVendorIcon source={model} className="size-3.5" />
                  <span className="font-medium text-foreground/75">
                    {modelDisplay.name}
                  </span>
                  {modelDisplay.code && (
                    <span className="max-w-full truncate font-mono text-muted-foreground/80">
                      {modelDisplay.code}
                    </span>
                  )}
                  {parameterItems.length > 0 && (
                    <>
                      <span aria-hidden="true">·</span>
                      {parameterItems.map((parameter) => (
                        <span
                          key={`${parameter.label}-${parameter.value}`}
                          className={cn(
                            "inline-flex items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5",
                            getParameterTone(parameter.label),
                          )}
                        >
                          <span className="opacity-65">{parameter.label}</span>
                          <span className="font-medium">
                            {parameter.value}
                          </span>
                        </span>
                      ))}
                    </>
                  )}
                  <span aria-hidden="true">·</span>
                  <span className="shrink-0 tabular-nums text-muted-foreground/70">
                    {formatGenerationDateTime(task.createTime)}
                  </span>
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {(status.tone === "queued" || (status.tone === "running" && model?.comfyuiWorkflowId)) && (
                  <Button
                    variant="destructive-ghost"
                    size="xs"
                    onClick={() => onCancel(task.taskId)}
                    disabled={cancellingTaskIds.has(task.taskId)}
                  >
                    {cancellingTaskIds.has(task.taskId) ? <Loader2 className="animate-spin" /> : <Square />}
                    取消
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => onReusePrompt(task.prompt)}
                  title="复用提示词"
                >
                  <RotateCcw className="h-3 w-3" />
                  再次使用
                </Button>
              </div>
            </header>

            <GenerationHistoryReferences
              task={task}
              onPreview={setMediaPreview}
            />

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
                  <GenerationHistoryResult
                    key={item.id}
                    mode={mode}
                    item={item}
                    prompt={task.prompt}
                    taskFailed={status.tone === "failed"}
                    taskError={task.errorMsg}
                    onUseReference={onUseReference}
                    onAddAsset={onAddAsset}
                    onPreview={setMediaPreview}
                  />
                ))}
              </div>
            ) : status.tone === "failed" ? (
              <Tooltip>
                <TooltipTrigger
                  render={
                    <p
                      tabIndex={0}
                      aria-label="生成失败，悬停或聚焦查看完整错误"
                      className="mt-2.5 line-clamp-2 w-full cursor-help break-all rounded-xl bg-destructive/6 px-3 py-2 text-left text-xs text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    >
                      {getGenerationErrorSummary(failureMessage)}
                    </p>
                  }
                />
                <TooltipContent className="max-w-[calc(100vw-2rem)] p-0 sm:max-w-lg">
                  <div className="max-h-[50vh] min-w-0 overflow-y-auto whitespace-pre-wrap break-all px-3 py-1.5 text-left">
                    {failureMessage}
                  </div>
                </TooltipContent>
              </Tooltip>
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

      <GenerationMediaPreviewDialog
        preview={mediaPreview}
        onOpenChange={(open) => {
          if (!open) setMediaPreview(null);
        }}
      />
    </div>
  );
}
