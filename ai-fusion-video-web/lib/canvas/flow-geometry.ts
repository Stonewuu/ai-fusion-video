import { Position, type NodeHandle } from "@xyflow/react";
import type { CanvasCapability, CanvasNode } from "./types";
import { isCanvasGenerationNode } from "./generation";

export { canvasGraphHierarchy } from "./hierarchy";

/**
 * Offscreen nodes need bounds and port anchors before their DOM exists. React Flow replaces
 * these initial dimensions/handles with ResizeObserver measurements when a node enters view.
 * No node body is simplified or collapsed as part of viewport culling.
 */
export function canvasNodeGeometry(
  node: CanvasNode,
  capability: CanvasCapability,
  collapsed: boolean,
  measured?: { width?: number; height?: number },
) {
  const width = Math.max(measured?.width ?? node.layout.width ?? 300, isCanvasGenerationNode(node) ? 360 : 0);
  const ports = [...capability.inputs, ...capability.outputs];
  const generation = isCanvasGenerationNode(node);
  const rich =
    !!node.resource ||
    ["image", "video", "audio", "document"].includes(node.kind);
  const height =
    node.kind === "group"
      ? collapsed
        ? 64
        : (measured?.height ?? node.layout.height ?? 400)
      : (measured?.height ?? (generation ? width * 9 / 16 + 24 : (rich ? 320 : 160) + ports.length * 32));
  const handle = (
    id: string,
    output: boolean,
    centerY: number,
  ): NodeHandle => ({
    id,
    type: output ? "source" : "target",
    position: output ? Position.Right : Position.Left,
    x: output ? width - 4 : -4,
    y: centerY - 4,
    width: 8,
    height: 8,
  });
  const handles = [
    handle("annotation-out", true, height / 2),
    handle("annotation-in", false, height / 2),
  ];
  if (node.kind !== "group" || !collapsed) {
    ports.forEach((port, index) =>
      handles.push(
        handle(
          port.id,
          index >= capability.inputs.length,
          generation
            ? height * (0.25 + 0.7 * ((index >= capability.inputs.length ? index - capability.inputs.length : index) + 1)
              / ((index >= capability.inputs.length ? capability.outputs.length : capability.inputs.length) + 1))
            : height - (ports.length - index) * 32 + 16,
        ),
      ),
    );
  }
  return {
    initialWidth: width,
    initialHeight: height,
    // Once mounted, retain React Flow's actual DOM handle bounds across graph updates.
    // Declared anchors are only needed before the first measurement.
    handles:
      measured?.width !== undefined && measured?.height !== undefined
        ? undefined
        : handles,
  };
}
