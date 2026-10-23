"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { SafeImage } from "@/components/media/managed-image";
import { useCanvasMediaImports } from "../imports/media-imports-provider";
import { resourceMedia } from "@/lib/canvas/resource-media";
import { resourceTitle } from "@/lib/canvas/resource-presentation";
import { worldPosition } from "@/lib/canvas/graph";
import { freeMediaPosition } from "@/lib/canvas/placement";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasResource } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasMediaCompare } from "../preview/media-compare";

export function CanvasResourceMedia({
  resource,
  onPreview,
}: {
  resource: CanvasResource;
  onPreview: (media: CanvasMediaPreview) => void;
}) {
  const store = useCanvasStoreApi();
  const imports = useCanvasMediaImports();
  const snapshot = useCanvas((state) => state.snapshot!);
  const pending = useCanvas((state) => !!state.pending);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [compare, setCompare] = useState<CanvasMediaPreview[] | null>(null);
  const media = resourceMedia(resource);
  if (!media.length) return null;
  const add = async (fields: string[]) => {
    setBusy(true);
    setError(null);
    try {
      const graph = store.getState().snapshot!.graph;
      const source = graph.nodes.find(
        (node) =>
          node.resource?.type === resource.header.resourceType &&
          node.resource.id === resource.header.resourceId,
      );
      const origin = source
        ? worldPosition(source, graph.nodes)
        : { x: 64, y: 64 };
      await imports.start(
        media
          .filter((item) => fields.includes(item.field))
          .map((item) => ({
            title: `${resourceTitle(resource)} · ${item.title}`.slice(0, 200),
            source: {
              type: resource.header.resourceType,
              id: resource.header.resourceId,
              version: resource.header.version,
              field: item.field,
            },
          })),
        freeMediaPosition(
          graph.nodes,
          { x: origin.x + 380, y: origin.y },
          fields.length,
        ),
      );
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="border-b border-border/20 p-4">
      <details>
        <summary className="cursor-pointer text-sm font-medium">
          {resource.header.resourceType === "shot"
            ? "首尾帧与媒体"
            : "对象素材"}{" "}
          · {media.length}
        </summary>
        <div className="space-y-3 pt-3">
          {media.map((item) => (
            <div key={item.field} className="flex items-center gap-2">
              <Checkbox
                aria-label={`选择${item.title}`}
                checked={selected.includes(item.field)}
                onCheckedChange={(value) =>
                  setSelected((current) =>
                    value
                      ? [...current, item.field]
                      : current.filter((key) => key !== item.field),
                  )
                }
              />
              {item.kind === "image" && (
                <SafeImage
                  src={item.source}
                  alt={item.title}
                  className="size-12 rounded-lg object-contain"
                />
              )}
              <span className="min-w-0 flex-1 text-xs">{item.title}</span>
              <Button variant="ghost" size="xs" onClick={() => onPreview(item)}>
                预览
              </Button>
              <Button
                variant="outline"
                size="xs"
                disabled={
                  busy ||
                  !imports.ready ||
                  pending ||
                  !snapshot.permissions.canEdit ||
                  snapshot.document.archived
                }
                onClick={() => void add([item.field])}
              >
                放入画布
              </Button>
            </div>
          ))}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={
                !selected.length ||
                !imports.ready ||
                !snapshot.permissions.canEdit ||
                snapshot.document.archived
              }
              onClick={() => void add(selected)}
            >
              导入所选（{selected.length}）
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={selected.length < 2 || selected.length > 4}
              onClick={() =>
                setCompare(
                  media.filter((item) => selected.includes(item.field)),
                )
              }
            >
              对比所选素材（2–4 项）
            </Button>
          </div>
          {imports.error && (
            <p role="alert" className="text-xs text-destructive">
              {imports.error}
            </p>
          )}
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
        </div>
      </details>
      {compare && (
        <CanvasMediaCompare media={compare} onClose={() => setCompare(null)} />
      )}
    </section>
  );
}
