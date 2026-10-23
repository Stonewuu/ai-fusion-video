import type { CanvasNodeRun, CanvasRunValue } from "@/lib/api/canvas-runs";
import type { CanvasNode, CanvasResource } from "./types";
import { nodeMedia, type CanvasMediaPreview } from "./presentation";

export interface CanvasNodeMediaGroup {
  label: string;
  items: (CanvasMediaPreview & { field?: string })[];
}

function distinct<T extends CanvasMediaPreview & { field?: string }>(items: T[]): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = `${item.kind}:${item.source}:${item.field ?? ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function resourceMedia(
  type: string,
  data: Record<string, unknown>,
  title: string,
): (CanvasMediaPreview & { field?: string })[] {
  const fields: [string, CanvasMediaPreview["kind"], string][] =
    type === "shot"
      ? [
          ["firstFrameImageUrl", "image", "首帧"],
          ["lastFrameImageUrl", "image", "尾帧"],
          ["videoUrl", "video", "视频"],
          ["generatedVideoUrl", "video", "视频"],
        ]
      : type === "storyboardEpisode"
        ? [["composedVideoUrl", "video", "成片"]]
        : type === "assetItem"
          ? [["imageUrl", "image", "图片"]]
          : [["coverUrl", "image", "封面"]];
  return fields.flatMap(([field, kind, label]) => {
    const source = data[field];
    return typeof source === "string" && source
      ? [{ source, kind, title: `${title} · ${label}`, field }]
      : [];
  });
}

export function runValueMedia(
  values: CanvasRunValue[],
  title: string,
): CanvasMediaPreview[] {
  return distinct(
    values.flatMap((value, index) => {
      const itemTitle = `${title} · 第 ${index + 1} 项`;
      if (
        value.mediaId &&
        ["image", "video", "audio", "document", "textFile"].includes(value.type)
      ) {
        return [
          {
            source: `media://${value.mediaId}`,
            kind: value.type as CanvasMediaPreview["kind"],
            title: itemTitle,
          },
        ];
      }
      return value.resource && value.data
        ? resourceMedia(
            value.type,
            value.data as Record<string, unknown>,
            itemTitle,
          )
        : [];
    }),
  );
}

/** Use pinned inputs and authorized results; a preview never implies a downstream input was connected. */
export function canvasNodeMediaGroups(
  node: CanvasNode,
  resource: CanvasResource | null,
  run: CanvasNodeRun | undefined,
  children: readonly CanvasNodeRun[] = [],
): CanvasNodeMediaGroup[] {
  const groups: CanvasNodeMediaGroup[] = [];
  const sourceMedia = ["image", "video", "audio", "document", "textFile"].includes(node.kind);
  if (run && !run.contentUnavailable && !sourceMedia) {
    const inputs = runValueMedia(
      Object.entries(run.inputs?.values ?? {})
        .filter(([port]) => port !== "context")
        .flatMap(([, values]) => values),
      `${node.title} · 输入`,
    );
    if (inputs.length) groups.push({ label: "输入参考", items: inputs });
    const outputValues = Object.values(run.outputs.values).flat();
    for (const child of children) {
      if (!child.contentUnavailable)
        outputValues.push(...Object.values(child.outputs.values).flat());
    }
    const outputs = runValueMedia(outputValues, `${node.title} · 结果`);
    if (outputs.length) groups.push({ label: "运行结果", items: outputs });
  }
  if (!groups.some((group) => group.label === "运行结果")) {
    const ownMedia = nodeMedia(node, resource);
    const items = resource
      ? distinct(resourceMedia(node.kind, resource.data, node.title))
      : [];
    if (ownMedia && !items.some((item) => item.kind === ownMedia.kind && item.source === ownMedia.source)) items.push(ownMedia);
    if (items.length)
      groups.push({ label: "内容预览", items: distinct(items) });
  }
  return groups;
}
