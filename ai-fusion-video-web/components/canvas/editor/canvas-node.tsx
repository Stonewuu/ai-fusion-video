"use client";

import { memo, useEffect, useRef, useState, type CSSProperties } from "react";
import { shallow } from "zustand/shallow";
import {
  Handle,
  Position,
  NodeResizer,
  useUpdateNodeInternals,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { LockKeyhole, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { canvasPortLabels, canvasTypeNames } from "@/lib/canvas/labels";
import { portTypes } from "@/lib/canvas/graph";
import {
  nodeSummary,
  type CanvasMediaPreview,
} from "@/lib/canvas/presentation";
import type {
  CanvasCapability,
  CanvasNode as DocumentNode,
  CanvasPort,
} from "@/lib/canvas/types";
import { useCanvasResource, useCanvasStoreApi } from "../canvas-provider";
import {
  useCanvasNodeRun,
  useCanvasNodeRunChildren,
  useCanvasNodeActivity,
} from "../runs/canvas-runs-provider";
import { canvasNodeMediaGroups } from "@/lib/canvas/node-media";
import { CanvasNodeQuickMedia } from "./node-quick-media";
import { CanvasNodeRunButton, CanvasNodeRunStatus } from "./node-run-controls";
import type { CanvasNodeStartResult } from "@/lib/canvas/node-execution";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";

export type CanvasFlowNode = Node<
  {
    node: DocumentNode;
    capability: CanvasCapability;
    canEdit: boolean;
    childCount: number;
    configurationIssue: string | null;
    invalidInput: boolean;
    onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
    onAddAfter: (id: string, portId: string, element: HTMLElement) => void;
    onRunNode: (id: string) => Promise<CanvasNodeStartResult>;
    onShowNodeRun: (runId?: string) => void;
  },
  "canvas"
>;

function Port({
  node,
  port,
  output,
  onAddAfter,
  canEdit,
}: {
  node: DocumentNode;
  port: CanvasPort;
  output: boolean;
  canEdit: boolean;
  onAddAfter: CanvasFlowNode["data"]["onAddAfter"];
}) {
  const types = portTypes(port, node.config);
  const label = canvasPortLabels[port.id] ?? port.label;
  return (
    <div
      role="group"
      aria-label={`${node.title} · ${label}${output ? "输出" : "输入"}`}
      className={cn(
        "relative flex min-h-8 items-center gap-2 px-3 text-xs",
        output && "justify-end",
      )}
    >
      <Handle
        type={output ? "source" : "target"}
        position={output ? Position.Right : Position.Left}
        id={port.id}
        isConnectable={canEdit}
        title={`${label} · ${types.length ? types.map((type) => canvasTypeNames[type]).join("、") : "待配置类型"}${port.collection ? "集合" : ""}`}
      />
      <span className="truncate text-muted-foreground">{label}</span>
      {output && canEdit && (
        <Button
          variant="ghost"
          size="icon-xs"
          className="nodrag nopan"
          aria-label={`从${label}添加后续节点`}
          onClick={(event) => onAddAfter(node.id, port.id, event.currentTarget)}
        >
          <Plus />
        </Button>
      )}
    </div>
  );
}

export const CanvasNodeRenderer = memo(
  function CanvasNodeRenderer({ data, selected }: NodeProps<CanvasFlowNode>) {
    const { node, capability, canEdit } = data;
    const run = useCanvasNodeRun(node.id);
    const children = useCanvasNodeRunChildren(node.id);
    const activity = useCanvasNodeActivity(node.id);
    const [starting, setStarting] = useState(false);
    const startingRef = useRef(false);
    const [startError, setStartError] = useState<string | null>(null);
    const start = async () => {
      if (startingRef.current) return;
      startingRef.current = true;
      setStarting(true);
      setStartError(null);
      try {
        const result = await data.onRunNode(node.id);
        setStartError(result.error);
      } finally { startingRef.current = false; setStarting(false); }
    };
    const resource = useCanvasResource(node.resource);
    const store = useCanvasStoreApi();
    const resizeVersion = useRef(node.layoutVersion);
    const updateNodeInternals = useUpdateNodeInternals();
    const portSignature = JSON.stringify([
      capability.inputs.map((port) => port.id),
      capability.outputs.map((port) => port.id),
    ]);
    const previousPorts = useRef(portSignature);
    useEffect(() => {
      if (previousPorts.current === portSignature) return;
      previousPorts.current = portSignature;
      updateNodeInternals(node.id);
    }, [node.id, portSignature, updateNodeInternals]);
    const isGroup = node.kind === "group";
    const mediaGroups = canvasNodeMediaGroups(
      node,
      resource.data,
      run,
      children,
    );
    const batchOperation =
      node.kind === "foreach"
        ? store
            .getState()
            .catalog?.nodes.find(
              (item) => item.kind === node.config.operationKind,
            )
        : undefined;
    const writebackField =
      node.kind === "writeback"
        ? store
            .getState()
            .catalog?.writebackFields.find(
              (item) => item.key === node.config.binding,
            )
        : undefined;
    const summary = (
      batchOperation
        ? `${batchOperation.label} · 最多 ${node.config.maxItems} 项\n每项独立执行，按原顺序汇总`
        : writebackField
          ? `${canvasTypeNames[writebackField.targetType]} · ${writebackField.label}\n查看改动后确认写入`
          : nodeSummary(node, resource.data)
    ).slice(0, 260);
    return (
      <div
        className={cn(
          "canvas-node-drag canvas-node-surface h-full overflow-visible rounded-xl border bg-card text-card-foreground",
          isGroup && "bg-muted/30",
          selected && "shadow-sm",
        )}
        data-selected={selected || undefined}
        data-invalid={data.invalidInput || undefined}
        style={{ "--canvas-tone": canvasNodeColor(node.kind) } as CSSProperties}
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
        {isGroup && (
          <NodeResizer
            minWidth={320}
            minHeight={160}
            isVisible={
              selected &&
              canEdit &&
              !node.layout.locked &&
              !node.layout.collapsed
            }
            color="var(--primary)"
            shouldResize={() => !store.getState().pending}
            onResizeStart={() => {
              resizeVersion.current = node.layoutVersion;
            }}
            onResizeEnd={(_event, params) => {
              void store
                .getState()
                .edit("调整分组大小", [
                  {
                    type: "canvas.node.layout",
                    nodeId: node.id,
                    expectedLayoutVersion: resizeVersion.current,
                    layout: {
                      ...node.layout,
                      x: params.x,
                      y: params.y,
                      width: params.width,
                      height: params.height,
                    },
                  },
                ])
                .catch(() => undefined);
            }}
          />
        )}
        <header className="canvas-node-drag canvas-node-heading flex cursor-grab items-center gap-3 rounded-t-xl border-b border-border/20 p-3 active:cursor-grabbing">
          <span className="canvas-node-icon flex size-8 shrink-0 items-center justify-center rounded-lg">
            <CanvasNodeTypeIcon
              capability={capability}
              className="size-4"
              aria-hidden
            />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium" title={node.title}>
              {node.title}
            </p>
            <p className="canvas-node-kind text-xs">
              {capability.label}
              {isGroup && ` · ${data.childCount} 个节点`}
            </p>
          </div>
          {node.layout.locked && (
            <LockKeyhole
              className="size-3.5 text-muted-foreground"
              aria-label="位置已锁定"
            />
          )}
          {data.configurationIssue && (
            <span
              className={cn(
                "shrink-0 rounded-full px-2 py-1 text-xs",
                data.invalidInput
                  ? "bg-destructive/10 text-destructive"
                  : "bg-amber-500/10 text-amber-700 dark:text-amber-300",
              )}
              title={data.configurationIssue}
            >
              {data.invalidInput ? "检查输入" : "待配置"}
            </span>
          )}
          <CanvasNodeRunButton node={node} capability={capability} activity={activity} starting={starting} onRun={() => void start()} />
        </header>
        {!isGroup && <CanvasNodeRunStatus activity={activity} starting={starting} startError={startError} childrenRuns={children}
          onShow={() => data.onShowNodeRun(activity?.runId)} onDismiss={() => setStartError(null)} />}
        {!isGroup && (
          <div className="space-y-3 p-3">
            {!!run?.inputs?.values.context?.length &&
              !run.contentUnavailable && (
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
              groups={mediaGroups}
              onPreview={(media, returnFocus) => {
                store.getState().select([node.id]);
                data.onPreview(media, returnFocus);
              }}
            />
            {summary && (
              <p className="line-clamp-5 whitespace-pre-wrap break-words text-xs leading-relaxed text-muted-foreground">
                {summary}
              </p>
            )}
            {!summary && !mediaGroups.length && !resource.error && (
              <p className="text-xs text-muted-foreground">
                {resource.loading
                  ? "正在读取内容…"
                  : node.resource
                    ? "打开查看内容"
                    : "点击节点查看或修改设置"}
              </p>
            )}
          </div>
        )}
        {(capability.inputs.length > 0 || capability.outputs.length > 0) && (
          <div className="grid grid-cols-2 border-t border-border/20 py-2">
            <div>
              {capability.inputs.map((port) => (
                <Port
                  key={port.id}
                  node={node}
                  port={port}
                  output={false}
                  canEdit={canEdit}
                  onAddAfter={data.onAddAfter}
                />
              ))}
            </div>
            <div>
              {capability.outputs.map((port) => (
                <Port
                  key={port.id}
                  node={node}
                  port={port}
                  output
                  canEdit={canEdit}
                  onAddAfter={data.onAddAfter}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    );
  },
  (previous, current) =>
    previous.selected === current.selected &&
    shallow(previous.data, current.data),
);
