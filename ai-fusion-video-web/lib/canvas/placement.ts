import type { CanvasCatalog, CanvasNode, CanvasResourceRef } from "./types";
import type { ManagedMedia } from "@/lib/api/media";
import { worldPosition } from "./graph";

/** 新分支保留下一列，向下寻找空位，不覆盖同一端口已创建的后续节点。 */
export function freeNodePosition(
  nodes: CanvasNode[],
  preferred: { x: number; y: number },
  size: { width: number; height: number },
  measured: Record<string, { width?: number; height?: number } | undefined> = {},
): { x: number; y: number } {
  const bounds = nodes.map((node) => ({
    ...worldPosition(node, nodes),
    width: measured[node.id]?.width ?? node.layout.width ?? 300,
    height: measured[node.id]?.height ?? node.layout.height ?? 280,
  })).sort((left, right) => left.y - right.y);
  let y = preferred.y;
  for (const node of bounds) {
    if (preferred.x < node.x + node.width + 40 && preferred.x + size.width + 40 > node.x
      && y < node.y + node.height + 48 && y + size.height + 48 > node.y) {
      y = node.y + node.height + 48;
    }
  }
  return { x: preferred.x, y };
}

// Match the import job's three-column grid and leave room for the whole batch.
// Keep the source's horizontal chain while moving past occupied world bounds.
export function freeMediaPosition(
  nodes: CanvasNode[],
  preferred: { x: number; y: number },
  count = 1,
): { x: number; y: number } {
  const width = Math.min(3, Math.max(1, count)) * 360 - 60;
  const height = Math.ceil(Math.max(1, count) / 3) * 280;
  const bounds = nodes
    .map((node) => ({
      ...worldPosition(node, nodes),
      width: node.layout.width ?? 300,
      height: node.layout.height ?? 280,
    }))
    .sort((a, b) => a.x - b.x);
  let x = preferred.x;
  for (const node of bounds) {
    if (
      preferred.y < node.y + node.height + 40 &&
      preferred.y + height + 40 > node.y &&
      x < node.x + node.width + 60 &&
      x + width + 60 > node.x
    )
      x = node.x + node.width + 60;
  }
  return { x, y: preferred.y };
}

export const canvasMediaKinds = {
  image: "IMAGE",
  video: "VIDEO",
  audio: "AUDIO",
  textFile: "TEXT",
  document: "DOCUMENT",
} as const;
export function mediaNode(
  media: ManagedMedia,
  position: { x: number; y: number },
): CanvasNode {
  const kind = (
    Object.keys(canvasMediaKinds) as (keyof typeof canvasMediaKinds)[]
  ).find((key) => canvasMediaKinds[key] === media.content.kind);
  if (!kind || media.state !== "READY")
    throw new Error("媒体类型未登记或文件尚未就绪");
  return {
    id: crypto.randomUUID(),
    kind,
    schemaVersion: 1,
    title: media.name.slice(0, 200),
    resource: null,
    config: { resourceId: media.id },
    configVersion: 0,
    layoutVersion: 0,
    layout: {
      ...position,
      width: 300,
      height: null,
      parentId: null,
      locked: false,
      collapsed: false,
    },
  };
}

export function resourceNode(
  catalog: CanvasCatalog,
  ref: CanvasResourceRef,
  title: string,
  position: { x: number; y: number },
): CanvasNode {
  const capability = catalog.nodes.find(
    (item) => item.resourceType === ref.type,
  );
  if (!capability) throw new Error("此业务类型尚未注册画布节点");
  return {
    id: crypto.randomUUID(),
    kind: capability.kind,
    schemaVersion: capability.schemaVersion,
    title: title.slice(0, 200),
    resource: ref,
    config: {},
    configVersion: 0,
    layoutVersion: 0,
    layout: {
      ...position,
      width: 300,
      height: null,
      parentId: null,
      locked: false,
      collapsed: false,
    },
  };
}
