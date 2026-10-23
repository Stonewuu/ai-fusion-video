"use client";

import { useState } from "react";
import { Link2, Locate, Unlink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { canvasPortLabels } from "@/lib/canvas/labels";
import { canvasInputPurposes } from "@/lib/canvas/input-bindings";
import type { CanvasCapability, CanvasNode } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasInputPicker } from "./input-picker";

export function CanvasNodeInputs({
  node,
  capability,
  disabled,
  unsaved,
}: {
  node: CanvasNode;
  capability: CanvasCapability;
  disabled: boolean;
  unsaved: boolean;
}) {
  const store = useCanvasStoreApi();
  const graph = useCanvas((state) => state.snapshot!.graph);
  const [portId, setPortId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (!capability.inputs.length) return null;
  const nodes = new Map(graph.nodes.map((item) => [item.id, item]));
  return (
    <section
      className="flex flex-col gap-3 border-t border-border/20 pt-4"
      aria-label="节点输入与来源"
    >
      <details className="min-w-0" open={capability.inputs.some((port) => port.required && !Object.hasOwn(node.config, port.id) && !graph.edges.some((edge) => edge.targetNodeId === node.id && edge.targetPort === port.id)) || undefined}>
      <summary className="cursor-pointer rounded-lg text-sm font-medium outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">
        输入与来源 <span className="ml-2 text-xs font-normal text-muted-foreground">{graph.edges.filter((edge) => edge.kind === "execution" && edge.targetNodeId === node.id).length} 条连接</span>
      </summary>
      <div className="flex flex-col gap-1 pt-3">
        <p className="text-xs text-muted-foreground">
          按创作需要组合素材，每种输入的用途分别保存。
        </p>
        {unsaved && (
          <p className="text-xs text-muted-foreground">
            保存当前配置后可调整输入连接。
          </p>
        )}
      </div>
      <div className="divide-y divide-border/20">
        {capability.inputs.map((port) => {
          const edges = graph.edges
            .filter(
              (edge) =>
                edge.kind === "execution" &&
                edge.targetNodeId === node.id &&
                edge.targetPort === port.id,
            )
            .sort((a, b) => a.inputIndex! - b.inputIndex!);
          const manual = Object.hasOwn(node.config, port.id);
          const label = canvasPortLabels[port.id] ?? port.label;
          return (
            <div key={port.id} className="flex flex-col gap-2 py-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">{label}</span>
                <Button
                  variant="ghost"
                  size="xs"
                  disabled={
                    disabled ||
                    unsaved ||
                    (port.collection && edges.length >= port.maxConnections)
                  }
                  aria-label={`选择${label}来源`}
                  onClick={() => setPortId(port.id)}
                >
                  <Link2 />
                  {manual || (!port.collection && edges.length)
                    ? "更换"
                    : "连接"}
                </Button>
              </div>
              {canvasInputPurposes[port.id] && (
                <p className="text-xs text-muted-foreground">
                  {canvasInputPurposes[port.id]}
                </p>
              )}
              {manual && (
                <p className="line-clamp-3 whitespace-pre-wrap text-xs text-muted-foreground">
                  手填内容：{String(node.config[port.id])}
                </p>
              )}
              {edges.map((edge, index) => (
                <div key={edge.id} className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-xs">
                    {port.collection ? `${index + 1}. ` : ""}
                    {nodes.get(edge.sourceNodeId)?.title} ·{" "}
                    {canvasPortLabels[edge.sourcePort!] ?? edge.sourcePort}
                    {edge.output ? " · 固定结果" : ""}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    title="定位输入来源"
                    onClick={() => store.getState().focus([edge.sourceNodeId])}
                  >
                    <Locate />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    title="移除此输入连接"
                    disabled={disabled || unsaved}
                    onClick={() => {
                      setError(null);
                      void store
                        .getState()
                        .edit("移除创作输入", [
                          {
                            type: "canvas.edge.delete",
                            edgeId: edge.id,
                            expectedVersion: edge.version,
                          },
                        ])
                        .catch((failure) =>
                          setError(getApiErrorMessage(failure)),
                        );
                    }}
                  >
                    <Unlink />
                  </Button>
                </div>
              ))}
              {!manual && !edges.length && (
                <p className="text-xs text-muted-foreground">
                  {port.id === "prompt" &&
                  graph.edges.some(
                    (edge) =>
                      edge.kind === "execution" &&
                      edge.targetNodeId === node.id &&
                      edge.targetPort === "context",
                  )
                    ? "由已连接的内容设定组合，也可以补充本次创作要求。"
                    : port.required
                      ? "未连接 · 运行时需要内容"
                      : "未连接 · 按需添加"}
                </p>
              )}
            </div>
          );
        })}
      </div>
      </details>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {portId && (
        <CanvasInputPicker
          key={portId}
          nodeId={node.id}
          portId={portId}
          onClose={() => setPortId(null)}
        />
      )}
    </section>
  );
}
