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
import { LockKeyhole } from "lucide-react";
import { canvasTypeNames } from "@/lib/canvas/labels";
import {
  nodeSummary,
  type CanvasMediaPreview,
} from "@/lib/canvas/presentation";
import type {
  CanvasCapability,
  CanvasNode as DocumentNode,
} from "@/lib/canvas/types";
import { useCanvasResource, useCanvasStoreApi } from "../canvas-provider";
import {
  useCanvasNodeRun,
  useCanvasNodeRunChildren,
  useCanvasNodeActivity,
} from "../runs/canvas-runs-provider";
import { canvasNodeMediaGroups } from "@/lib/canvas/node-media";
import type { CanvasNodeStartResult } from "@/lib/canvas/node-execution";
import { canvasNodeColor } from "@/lib/canvas/node-appearance";
import { CanvasNodeTypeIcon } from "../shared/canvas-node-icon";
import { isCanvasGenerationNode } from "@/lib/canvas/generation";
import { CanvasGenerationNode } from "../generation/generation-node";
import { CanvasStandardNode } from "./standard-node";
import { CanvasNodeTitle } from "./node-title";

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
      if (startingRef.current) {
        return;
      }
      startingRef.current = true;
      setStarting(true);
      setStartError(null);
      try {
        const result = await data.onRunNode(node.id);
        setStartError(result.error);
      } finally {
        startingRef.current = false;
        setStarting(false);
      }
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
      if (previousPorts.current === portSignature) {
        return;
      }
      previousPorts.current = portSignature;
      updateNodeInternals(node.id);
    }, [node.id, portSignature, updateNodeInternals]);
    const source = ["constant", "resource"].includes(capability.executor);
    const generation = isCanvasGenerationNode(node);
    const mediaGroups = canvasNodeMediaGroups(
      node,
      resource.data,
      run,
      children,
    )
      .filter((group) => !generation || group.label !== "输入参考")
      .map((group) =>
        generation && group.label === "运行结果"
          ? {
              ...group,
              label: "生成结果",
              items: group.items
                .filter(
                  (item) =>
                    item.kind ===
                    (node.kind === "imageGeneration" ? "image" : "video"),
                )
                .map((item, index) => ({
                  ...item,
                  title: `${node.title} · 结果 · 第 ${index + 1} 项`,
                })),
            }
          : group,
      )
      .filter((group) => group.items.length > 0);
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
    // 生成文本完成后直接展示结果，未生成时展示写作内容。
    const generatedText =
      node.kind === "textGeneration" &&
      activity?.node.status === "SUCCEEDED" &&
      !activity.node.contentUnavailable
        ? (activity.node.outputs.values.text?.[0]?.text ?? "")
        : "";
    const summary = (
      generatedText ||
      (batchOperation
        ? `${batchOperation.label} · 最多 ${node.config.maxItems} 项\n每项独立执行，按原顺序汇总`
        : writebackField
          ? `${canvasTypeNames[writebackField.targetType]} · ${writebackField.label}\n查看改动后确认写入`
          : nodeSummary(node, resource.data))
    ).slice(0, 260);
    if (generation) {
      return (
        <CanvasGenerationNode
          data={data}
          selected={selected}
          activity={activity}
          groups={mediaGroups}
        />
      );
    }
    if (node.kind !== "group") {
      return (
        <CanvasStandardNode
          data={data}
          selected={selected}
          activity={activity}
          run={run}
          childrenRuns={children}
          resource={resource}
          groups={mediaGroups}
          summary={summary}
          runnable={!source}
          starting={starting}
          startError={startError}
          onStart={() => void start()}
          onDismissError={() => setStartError(null)}
          onPreview={(media, returnFocus) => {
            store.getState().select([node.id]);
            data.onPreview(media, returnFocus);
          }}
        />
      );
    }
    // 分组只承载布局：没有连接点、运行入口和内容摘要。
    return (
      <div
        className="canvas-node-drag canvas-node-surface flex h-full flex-col overflow-visible rounded-xl border bg-muted/30 text-card-foreground"
        data-selected={selected || undefined}
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
        <NodeResizer
          minWidth={320}
          minHeight={160}
          isVisible={
            selected && canEdit && !node.layout.locked && !node.layout.collapsed
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
        <header className="canvas-node-drag flex min-h-14 cursor-grab items-center gap-2 rounded-t-xl p-3 active:cursor-grabbing">
          <span className="canvas-node-icon flex size-7 shrink-0 items-center justify-center rounded-lg">
            <CanvasNodeTypeIcon
              capability={capability}
              className="size-4"
              aria-hidden
            />
          </span>
          <div className="flex min-w-0 flex-1 items-baseline gap-2">
            <CanvasNodeTitle node={node} canEdit={canEdit} />
            <p className="canvas-node-kind shrink-0 text-xs">
              {capability.label} · {data.childCount} 个节点
            </p>
          </div>
          {node.layout.locked && (
            <LockKeyhole
              className="size-3.5 text-muted-foreground"
              aria-label="位置已锁定"
            />
          )}
        </header>
      </div>
    );
  },
  (previous, current) =>
    previous.selected === current.selected &&
    shallow(previous.data, current.data),
);
