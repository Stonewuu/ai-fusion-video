"use client";

import { Handle, Position, useConnection } from "@xyflow/react";
import { canvasPortLabels, canvasTypeNames } from "@/lib/canvas/labels";
import { portTypes } from "@/lib/canvas/graph";
import type { CanvasCapability, CanvasNode } from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";

/**
 * 节点两侧的连接点，按顺序分布在内容框高度内。已连接、正在连线和首个输出始终显示；
 * 其余连接点与端口名称在节点悬停或选中时出现，避免静止状态的视觉噪音。
 */
export function CanvasNodePorts({
  node,
  capability,
  canEdit,
}: {
  node: CanvasNode;
  capability: CanvasCapability;
  canEdit: boolean;
}) {
  const edges = useCanvas((state) => state.snapshot!.graph.edges);
  const connecting = useConnection((connection) => connection.inProgress);
  return [false, true].map((output) => {
    const ports = output ? capability.outputs : capability.inputs;
    return ports.map((port, index) => {
      const name = canvasPortLabels[port.id] ?? port.label;
      const types = portTypes(port, node.config);
      const label = `${node.title} · ${name}${output ? "输出" : "输入"}`;
      const connected = edges.some(
        (edge) =>
          edge.kind === "execution" &&
          (output
            ? edge.sourceNodeId === node.id && edge.sourcePort === port.id
            : edge.targetNodeId === node.id && edge.targetPort === port.id),
      );
      const pinned = connecting || connected || (output && index === 0);
      const top = `${25 + ((index + 1) / (ports.length + 1)) * 70}%`;
      return (
        <div key={`${output}:${port.id}`}>
          <Handle
            id={port.id}
            type={output ? "source" : "target"}
            position={output ? Position.Right : Position.Left}
            isConnectable={canEdit}
            className={pinned ? undefined : "canvas-port-idle"}
            title={`${name} · ${types.length ? types.map((type) => canvasTypeNames[type]).join("、") : "待配置类型"}${port.collection ? "集合" : ""}`}
            aria-label={label}
            style={{ top }}
          />
          {(!output || ports.length > 1) && <span
            aria-hidden
            className="canvas-port-label pointer-events-none absolute whitespace-nowrap rounded-[8px] border border-foreground/[0.09] bg-popover/95 px-1.5 py-0.5 text-[11px] font-medium text-foreground/80 shadow-xs backdrop-blur-md dark:border-white/[0.12]"
            data-side={output ? "right" : "left"}
            data-visible={connecting || undefined}
            style={{ top }}
          >
            {name}
          </span>}
        </div>
      );
    });
  });
}
