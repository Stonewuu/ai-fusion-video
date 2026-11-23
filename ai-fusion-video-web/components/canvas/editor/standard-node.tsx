"use client";

import type { CSSProperties } from "react";
import { Handle, Position } from "@xyflow/react";
import { LockKeyhole, Plus } from "lucide-react";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { cn } from "@/lib/utils";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import type { CanvasNodeMediaGroup } from "@/lib/canvas/node-media";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { CanvasNodeActivity, CanvasNodeRun } from "@/lib/api/canvas-runs";
import { canRunCanvasNode } from "@/lib/canvas/node-execution";
import { hasCanvasGenerationComposer } from "@/lib/canvas/generation";
import { useCanvasResource } from "../canvas-provider";
import { useCanvasNodeRun } from "../runs/canvas-runs-provider";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import type { CanvasFlowNode } from "./canvas-node";
import { CanvasNodeQuickMedia } from "./node-quick-media";
import { CanvasNodeRunButton, CanvasNodeRunStatus } from "./node-run-controls";
import { CanvasNodePorts } from "./node-ports";
import { CanvasNodeTitle } from "./node-title";

/**
 * 普通节点与生成节点共用的外观：框外标题行 + 内容框 + 两侧连接点 + 右侧“添加后续”。
 */
export function CanvasStandardNode({
  data,
  selected,
  activity,
  run,
  childrenRuns,
  resource,
  groups,
  summary,
  runnable,
  starting,
  startError,
  onStart,
  onDismissError,
  onPreview,
}: {
  data: CanvasFlowNode["data"];
  selected: boolean;
  activity?: CanvasNodeActivity;
  run: ReturnType<typeof useCanvasNodeRun>;
  childrenRuns: readonly CanvasNodeRun[];
  resource: ReturnType<typeof useCanvasResource>;
  groups: CanvasNodeMediaGroup[];
  summary: string;
  runnable: boolean;
  starting: boolean;
  startError: string | null;
  onStart: () => void;
  onDismissError: () => void;
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
}) {
  const { node, capability, canEdit } = data;
  const tone = canvasNodeColor(node.kind);
  const primary = capability.outputs[0];
  const textual = ["text", "note", "story", "textFile", "textGeneration"].includes(
    node.kind,
  );
  return (
    <div
      className="canvas-node-drag group/node relative"
      data-selected={selected || undefined}
      style={
        {
          "--canvas-tone": tone,
          "--xy-handle-border-color": tone,
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
      <div className="mb-2 flex h-6 min-w-0 cursor-grab items-center gap-1.5 px-0.5 text-xs text-muted-foreground active:cursor-grabbing">
        <span className="canvas-node-badge-icon shrink-0">
          <CanvasNodeTypeIcon
            capability={capability}
            className="size-3"
            aria-hidden
          />
        </span>
        <CanvasNodeTitle node={node} canEdit={canEdit} />
        {node.title !== capability.label && (
          <span className="shrink-0 text-[11px] text-muted-foreground/80">
            {capability.label}
          </span>
        )}
        {node.layout.locked && (
          <LockKeyhole className="size-3.5 shrink-0" aria-label="位置已锁定" />
        )}
        {data.configurationIssue && (
          <span
            className={cn(
              "ml-auto shrink-0 rounded-[6px] px-1.5 py-0.5 text-[11px] font-medium",
              data.invalidInput
                ? "bg-destructive/10 text-destructive"
                : "bg-amber-500/12 text-amber-700 dark:text-amber-300",
            )}
            title={data.configurationIssue}
          >
            {data.invalidInput ? "检查输入" : "待配置"}
          </span>
        )}
      </div>
      <div
        className="canvas-node-surface flex min-h-24 flex-col overflow-hidden rounded-xl border bg-card text-card-foreground"
        data-selected={selected || undefined}
        data-invalid={data.invalidInput || undefined}
      >
        {runnable && (
          <CanvasNodeRunStatus
            activity={activity}
            starting={starting}
            startError={startError}
            childrenRuns={childrenRuns}
            onShow={() => data.onShowNodeRun(activity?.runId)}
            onDismiss={onDismissError}
          />
        )}
        <div
          className={cn(
            "flex min-w-0 flex-1 flex-col gap-3",
            // 只有媒体内容时让画面铺满内容框，不再嵌套第二层边框。
            groups.length && !summary && !resource.error
              ? "canvas-node-fill"
              : "p-3",
          )}
        >
          {!!run?.inputs?.values.context?.length && !run.contentUnavailable && (
            <p className="text-xs text-muted-foreground">
              内容设定 · {run.inputs.values.context.length} 项，按文字使用
            </p>
          )}
          {resource.error && (
            <p role="status" className="text-xs text-destructive">
              {resource.error}
            </p>
          )}
          <CanvasNodeQuickMedia
            node={node}
            resource={resource.data}
            canEdit={canEdit}
            groups={groups}
            onPreview={onPreview}
          />
          {summary && (
            <p
              className={cn(
                "whitespace-pre-wrap break-words",
                textual
                  ? "line-clamp-8 text-sm leading-6 text-foreground/85"
                  : "line-clamp-5 text-xs leading-5 text-foreground/75",
              )}
            >
              {summary}
            </p>
          )}
          {!summary && !groups.length && !resource.error && (
            <p className="flex flex-1 items-center justify-center py-4 text-xs text-muted-foreground">
              {resource.loading
                ? "正在读取内容…"
                : node.resource
                  ? "打开查看内容"
                  : node.kind === "textGeneration"
                    ? "点击开始写作"
                    : textual
                      ? "点击编写内容"
                      : "点击查看或修改设置"}
            </p>
          )}
        </div>
        {/* 带生成表单的节点由表单提供生成入口，节点内不再重复主按钮。 */}
        {runnable && canRunCanvasNode(capability) && !hasCanvasGenerationComposer(node) && (
          <div className="flex items-center justify-end px-3 pb-3">
            <CanvasNodeRunButton
              node={node}
              capability={capability}
              activity={activity}
              starting={starting}
              onRun={onStart}
            />
          </div>
        )}
      </div>
      <CanvasNodePorts node={node} capability={capability} canEdit={canEdit} />
      {canEdit && primary && (
        <div
          className={cn(
            "nodrag nopan absolute -right-10 opacity-0 transition-opacity duration-150 group-hover/node:opacity-100 group-focus-within/node:opacity-100 motion-reduce:transition-none [@media(hover:none)]:opacity-100",
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
