"use client";

import { useState, type CSSProperties } from "react";
import { Handle, Position } from "@xyflow/react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Expand,
  ImageIcon,
  Info,
  Loader2,
  Plus,
  Video,
} from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import { cn } from "@/lib/utils";
import { userFacingError } from "@/lib/user-facing-error";
import { runFinished, runLabels } from "@/lib/canvas/run-state";
import type { CanvasNodeActivity } from "@/lib/api/canvas-runs";
import type { CanvasNodeMediaGroup } from "@/lib/canvas/node-media";
import type { CanvasFlowNode } from "../editor/canvas-node";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeInlinePlayer } from "../editor/node-inline-player";
import { CanvasNodePorts } from "../editor/node-ports";
import { CanvasNodeTitle } from "../editor/node-title";

export function CanvasGenerationNode({
  data,
  selected,
  activity,
  groups,
}: {
  data: CanvasFlowNode["data"];
  selected: boolean;
  activity?: CanvasNodeActivity;
  groups: CanvasNodeMediaGroup[];
}) {
  const { node, capability, canEdit } = data;
  const [chosen, setChosen] = useState<string | null>(null);
  const items = groups.flatMap((group) => group.items);
  const index = Math.max(
    0,
    items.findIndex((item) => item.source === chosen),
  );
  const media = items[index];
  const active = !!activity && !runFinished(activity.node.status);
  const error = activity?.node.error;
  const stale = !!activity && activity.node.derivedState !== "CURRENT";
  const Icon = node.kind === "imageGeneration" ? ImageIcon : Video;
  const primary = capability.outputs[0];
  const showDetails = () => data.onShowNodeRun(activity?.runId);
  return (
    <div
      className="canvas-node-drag canvas-generation-node group relative"
      data-selected={selected || undefined}
      data-invalid={data.invalidInput || undefined}
      style={
        {
          "--canvas-tone": canvasNodeColor(node.kind),
          "--xy-handle-border-color": canvasNodeColor(node.kind),
          "--xy-handle-background-color": "var(--card)",
        } as CSSProperties
      }
    >
      <Handle
        type="source"
        position={Position.Right}
        id="annotation-out"
        isConnectable={false}
        className="pointer-events-none invisible"
      />
      <Handle
        type="target"
        position={Position.Left}
        id="annotation-in"
        isConnectable={false}
        className="pointer-events-none invisible"
      />
      <div className="mb-2 flex h-6 min-w-0 items-center gap-1.5 px-0.5 text-xs text-muted-foreground">
        <span className="canvas-node-badge-icon shrink-0">
          <Icon className="size-3" aria-hidden />
        </span>
        <CanvasNodeTitle node={node} canEdit={canEdit} />
        <span className="flex-1" />
        {active ? (
          <span
            role="status"
            className="flex items-center gap-1 rounded-[6px] bg-primary/10 px-1.5 py-0.5 text-[11px] font-medium text-primary"
          >
            <Loader2 className="size-3 animate-spin motion-reduce:animate-none" />
            {runLabels[activity.node.status]}
          </span>
        ) : error ? (
          <span className="rounded-[6px] bg-destructive/10 px-1.5 py-0.5 text-[11px] font-medium text-destructive">
            生成失败
          </span>
        ) : stale ? (
          <span className="rounded-[6px] bg-amber-500/12 px-1.5 py-0.5 text-[11px] font-medium text-amber-700 dark:text-amber-300">
            待更新
          </span>
        ) : activity?.node.status === "SUCCEEDED" ? (
          <Check
            className="size-3.5 text-(--canvas-flow)"
            aria-label="已完成"
          />
        ) : null}
        {activity && (
          <CanvasIconButton
            size="icon-xs"
            className="nodrag nopan"
            label="查看生成详情"
            onClick={(event) => {
              event.stopPropagation();
              showDetails();
            }}
          >
            <Info />
          </CanvasIconButton>
        )}
      </div>
      <div
        className="canvas-generation-frame relative aspect-video overflow-hidden rounded-xl border bg-muted"
        aria-label={media ? "生成结果" : "生成预览"}
      >
        {media ? (
          media.kind === "image" ? (
            <MediaThumbnail
              source={media.source}
              kind="image"
              alt={media.title}
            />
          ) : (
            <CanvasNodeInlinePlayer key={media.source} media={media} />
          )
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-2 text-muted-foreground/60">
            {active ? (
              <Loader2 className="size-8 animate-spin motion-reduce:animate-none" />
            ) : (
              <Icon className="size-10" strokeWidth={1.25} />
            )}
            {active && (
              <p role="status" className="text-xs">
                {runLabels[activity.node.status]}
              </p>
            )}
          </div>
        )}
        {media && (
          <div
            className={cn(
              "nodrag nopan absolute right-2 top-2 flex gap-2 opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 motion-reduce:transition-none",
              selected && "opacity-100",
            )}
          >
            <CanvasIconButton
              variant="overlay"
              size="icon-sm"
              label="放大预览"
              aria-label={`放大预览${media.title}`}
              onClick={(event) => {
                event.stopPropagation();
                data.onPreview(media, event.currentTarget);
              }}
            >
              <Expand />
            </CanvasIconButton>
          </div>
        )}
        {items.length > 1 && (
          <div className="nodrag nopan absolute bottom-2 right-2 flex items-center gap-1 rounded-[10px] border border-foreground/[0.1] bg-popover/92 p-0.5 text-xs tabular-nums shadow-[0_2px_8px_-1px_rgba(0,0,0,0.14)] backdrop-blur-md dark:border-white/[0.14]">
            <CanvasButton
              size="icon-xs"
              aria-label="上一项生成结果"
              disabled={index === 0}
              onClick={(event) => {
                event.stopPropagation();
                setChosen(items[index - 1].source);
              }}
            >
              <ChevronLeft />
            </CanvasButton>
            <span aria-live="polite">
              {index + 1} / {items.length}
            </span>
            <CanvasButton
              size="icon-xs"
              aria-label="下一项生成结果"
              disabled={index === items.length - 1}
              onClick={(event) => {
                event.stopPropagation();
                setChosen(items[index + 1].source);
              }}
            >
              <ChevronRight />
            </CanvasButton>
          </div>
        )}
      </div>
      {error && (
        <p
          role="alert"
          className="mt-2 line-clamp-2 px-1 text-xs text-destructive"
        >
          {userFacingError(error)}
        </p>
      )}
      <CanvasNodePorts node={node} capability={capability} canEdit={canEdit} />
      {canEdit && primary && (
        <div
          className={cn(
            "nodrag nopan absolute -right-10 opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100 motion-reduce:transition-none",
            // 多输出节点右侧会显示端口名称，将快捷加号放在右上外侧避开标签。
            capability.outputs.length > 1
              ? "top-9"
              : "top-1/2 -translate-y-1/2",
            selected && "opacity-100",
          )}
        >
          <CanvasIconButton
            variant="overlay"
            size="icon-sm"
            shape="round"
            side="right"
            label="添加后续节点"
            onClick={(event) => {
              event.stopPropagation();
              data.onAddAfter(node.id, primary.id, event.currentTarget);
            }}
          >
            <Plus />
          </CanvasIconButton>
        </div>
      )}
    </div>
  );
}
