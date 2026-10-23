"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { canvasApi } from "@/lib/api/canvas";
import type { ImportSource } from "@/lib/api/media-imports";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { resourceMedia } from "@/lib/canvas/resource-media";
import { resourceTitle } from "@/lib/canvas/resource-presentation";
import { worldPosition } from "@/lib/canvas/graph";
import { freeMediaPosition } from "@/lib/canvas/placement";
import { useCanvasStoreApi } from "../canvas-provider";
import { useCanvasMediaImports } from "./media-imports-provider";

export function CanvasImportSources({ onClose }: { onClose: () => void }) {
  const imports = useCanvasMediaImports();
  const store = useCanvasStoreApi();
  const [sources, setSources] = useState<ImportSource[] | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [origin] = useState(() => {
    const state = store.getState();
    const nodes = state.snapshot?.graph.nodes ?? [];
    const points = nodes
      .filter((n) => state.selection.includes(n.id))
      .map((n) => worldPosition(n, nodes));
    return {
      x: points.length ? Math.max(...points.map((p) => p.x)) + 380 : 64,
      y: points.length ? Math.min(...points.map((p) => p.y)) : 64,
    };
  });
  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      try {
        const state = store.getState();
        const refs = state
          .snapshot!.graph.nodes.filter(
            (n) => state.selection.includes(n.id) && n.resource,
          )
          .map((n) => n.resource!);
        const unique = [
          ...new Map(
            refs.map((ref) => [`${ref.type}:${ref.id}`, ref]),
          ).values(),
        ];
        if (!unique.length)
          throw new Error("请先在画布中选择项目、资产、分镜或分集节点");
        if (unique.length > 50)
          throw new Error("一次最多读取 50 个内容节点，请减少选择");
        const all: ImportSource[] = [];
        for (const ref of unique) {
          const resource = await canvasApi.resource(
            state.canvasId,
            ref.type,
            ref.id,
            controller.signal,
          );
          all.push(
            ...resourceMedia(resource).map((item) => ({
              title: `${resourceTitle(resource)} · ${item.title}`.slice(0, 200),
              source: {
                type: ref.type,
                id: ref.id,
                version: resource.header.version,
                field: item.field,
              },
            })),
          );
        }
        if (!controller.signal.aborted) {
          setSources(all);
          setSelected(all.length <= 50 ? all.map((_, index) => index) : []);
        }
      } catch (failure) {
        if (!controller.signal.aborted) setError(getApiErrorMessage(failure));
      }
    };
    void load();
    return () => controller.abort();
  }, [store]);
  const start = async (place: boolean) => {
    if (!sources) return;
    const ok = await imports.start(
      sources.filter((_, index) => selected.includes(index)),
      place
        ? freeMediaPosition(
            store.getState().snapshot!.graph.nodes,
            origin,
            selected.length,
          )
        : null,
    );
    if (ok) onClose();
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>导入所选节点素材</DialogTitle>
          <DialogDescription>
            已选 {selected.length} 项，每批最多 50
            项。导入会固定当前来源版本，关闭页面后仍可继续。
          </DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-3 overflow-y-auto p-1">
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          {!sources && !error && (
            <p role="status" className="text-sm text-muted-foreground">
              正在读取来源素材…
            </p>
          )}
          {sources?.length === 0 && (
            <p className="text-sm text-muted-foreground">
              所选节点还没有可导入的素材
            </p>
          )}
          {sources?.map((source, index) => (
            <label key={index} className="flex items-start gap-3 py-2 text-sm">
              <Checkbox
                checked={selected.includes(index)}
                disabled={!selected.includes(index) && selected.length >= 50}
                onCheckedChange={(value) =>
                  setSelected((old) =>
                    value ? [...old, index] : old.filter((i) => i !== index),
                  )
                }
              />
              <span className="break-words">{source.title}</span>
            </label>
          ))}
          {imports.error && (
            <p role="alert" className="text-sm text-destructive">
              {imports.error}
            </p>
          )}
        </div>
        <DialogFooter className="shrink-0">
          <Button
            variant="outline"
            disabled={!selected.length || !imports.ready}
            onClick={() => void start(false)}
          >
            仅导入素材库
          </Button>
          <Button
            disabled={!selected.length || !imports.ready}
            onClick={() => void start(true)}
          >
            导入并放入画布
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
