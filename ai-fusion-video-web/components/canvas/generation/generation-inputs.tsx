"use client";

import { Expand, FileText, Music2, Plus, X } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { canvasPortLabels } from "@/lib/canvas/labels";
import type { CanvasGenerationInputs } from "@/lib/canvas/generation";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { useCanvasStoreApi } from "../canvas-provider";

export function GenerationInputStrip({
  value,
  canEdit,
  onPreview,
  onFocus,
  onEditInputs,
  onError,
}: {
  value: CanvasGenerationInputs;
  canEdit: boolean;
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
  onFocus: (nodeId: string) => void;
  onEditInputs: () => void;
  onError: (message: string | null) => void;
}) {
  const store = useCanvasStoreApi();
  const removeInput = (edgeId: string) => {
    const edge = store
      .getState()
      .snapshot?.graph.edges.find((item) => item.id === edgeId);
    if (!edge) {
      return;
    }
    onError(null);
    void store
      .getState()
      .edit("断开输入连线", [
        {
          type: "canvas.edge.delete",
          edgeId: edge.id,
          expectedVersion: edge.version,
        },
      ])
      .catch((failure) => onError(getApiErrorMessage(failure)));
  };
  const media = value.inputs.flatMap((input) =>
    input.values.flatMap((item) => {
      if (!item.mediaId || !["image", "video", "audio"].includes(item.type)) {
        return [];
      }
      return [
        {
          key: `${input.edgeId}:${item.id}`,
          edgeId: input.edgeId,
          label: canvasPortLabels[input.port] ?? input.port,
          source: `media://${item.mediaId}`,
          kind: item.type as "image" | "video" | "audio",
          title: input.title,
        },
      ];
    }),
  );
  const textSources = value.inputs.filter((input) =>
    ["prompt", "context"].includes(input.port),
  );
  if (!media.length && !textSources.length) {
    return (
      <CanvasButton
        variant="ghost"
        size="sm"
        className="text-muted-foreground"
        onClick={onEditInputs}
      >
        <Plus />
        添加参考
      </CanvasButton>
    );
  }
  return (
    <div
      aria-label="已连接素材"
      className="flex min-w-0 items-center gap-2 overflow-x-auto px-1 py-1.5"
    >
      {media.map((item) => (
        <div key={item.key} className="group/item relative shrink-0">
          <button
            type="button"
            className="group/thumb relative flex size-9 items-center justify-center overflow-hidden rounded-[10px] border border-foreground/[0.1] bg-muted text-muted-foreground shadow-[0_1px_2px_rgba(0,0,0,0.06)] outline-none transition-[border-color,box-shadow] duration-150 hover:border-foreground/25 focus-visible:ring-[3px] focus-visible:ring-primary/30 dark:border-white/[0.14] motion-reduce:transition-none"
            aria-label={`预览${item.label}：${item.title}`}
            title={`${item.label} · ${item.title}`}
            onClick={(event) => onPreview(item, event.currentTarget)}
          >
            {item.kind === "audio" ? (
              <Music2 className="size-4" />
            ) : (
              <span className="absolute inset-0 overflow-hidden">
                <MediaThumbnail
                  source={item.source}
                  kind={item.kind}
                  alt={item.title}
                  fit="cover"
                  compact
                  interactive={false}
                />
              </span>
            )}
            <span
              aria-hidden
              className="absolute inset-0 flex items-center justify-center bg-black/35 text-white opacity-0 transition-opacity duration-150 group-hover/thumb:opacity-100 motion-reduce:transition-none"
            >
              <Expand className="size-3.5" />
            </span>
          </button>
          {canEdit && (
            <button
              type="button"
              aria-label={`断开${item.label}：${item.title}`}
              title={`断开“${item.title}”的连接`}
              className="absolute -right-1.5 -top-1.5 z-10 flex size-4 items-center justify-center rounded-full border border-background bg-foreground/80 text-background opacity-0 shadow-xs outline-none transition-[opacity,background-color,color] duration-150 hover:bg-destructive hover:text-white focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-primary/40 group-hover/item:opacity-100 group-focus-within/item:opacity-100 [@media(hover:none)]:opacity-100 motion-reduce:transition-none"
              onClick={(event) => {
                event.stopPropagation();
                removeInput(item.edgeId);
              }}
            >
              <X className="size-2.5" strokeWidth={2.5} />
            </button>
          )}
        </div>
      ))}
      {textSources.map((input) => (
        <div key={input.edgeId} className="group/item relative shrink-0">
          <button
            type="button"
            className="flex h-9 max-w-44 items-center gap-2 rounded-[10px] border border-foreground/[0.09] bg-background/90 px-2.5 text-[13px] font-medium text-foreground shadow-[0_1px_2px_rgba(0,0,0,0.04)] outline-none transition-[border-color,background-color,box-shadow] duration-150 hover:border-foreground/[0.18] hover:bg-background focus-visible:ring-[3px] focus-visible:ring-primary/30 dark:border-white/[0.1] dark:bg-white/[0.06] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_1px_2px_rgba(0,0,0,0.22)] dark:hover:border-white/[0.18] dark:hover:bg-white/[0.1] motion-reduce:transition-none"
            title={`定位到“${input.title}”`}
            onClick={() => onFocus(input.nodeId)}
          >
            <span className="grid size-5 shrink-0 place-items-center rounded-[6px] bg-violet-500/12 text-(--canvas-text)">
              <FileText className="size-3.5" aria-hidden />
            </span>
            <span className="truncate">{input.title}</span>
          </button>
          {canEdit && (
            <button
              type="button"
              aria-label={`断开文本：${input.title}`}
              title={`断开“${input.title}”的连接`}
              className="absolute -right-1.5 -top-1.5 z-10 flex size-4 items-center justify-center rounded-full border border-background bg-foreground/80 text-background opacity-0 shadow-xs outline-none transition-[opacity,background-color,color] duration-150 hover:bg-destructive hover:text-white focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-primary/40 group-hover/item:opacity-100 group-focus-within/item:opacity-100 [@media(hover:none)]:opacity-100 motion-reduce:transition-none"
              onClick={(event) => {
                event.stopPropagation();
                removeInput(input.edgeId);
              }}
            >
              <X className="size-2.5" strokeWidth={2.5} />
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

export function GenerationConnectedPrompt({ prompt }: { prompt: string }) {
  if (!prompt) {
    return null;
  }
  return (
    <div
      className="flex min-w-0 items-start gap-2.5 rounded-[12px] border border-foreground/[0.07] bg-foreground/[0.025] px-3 py-2 dark:border-white/[0.08] dark:bg-white/[0.03]"
      aria-label="上游提示词"
    >
      <span
        aria-hidden
        className="mt-1 h-3.5 w-0.5 shrink-0 rounded-full bg-primary/55"
      />
      <p className="line-clamp-2 min-w-0 flex-1 whitespace-pre-wrap break-words text-[13px] leading-5 text-foreground/75">
        {prompt}
      </p>
      <Popover>
        <PopoverTrigger
          render={
            <CanvasButton
              size="icon-xs"
              className="shrink-0 text-muted-foreground"
              aria-label="查看完整提示词"
              title="查看完整内容"
            />
          }
        >
          <Expand />
        </PopoverTrigger>
        <PopoverContent
          side="top"
          className="w-96 max-w-[calc(100vw-2rem)] rounded-[16px] border border-foreground/[0.1] p-3 shadow-xl dark:border-white/[0.12]"
        >
          <p className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words text-sm leading-6">
            {prompt}
          </p>
        </PopoverContent>
      </Popover>
    </div>
  );
}
