"use client";

import { useEffect, useState } from "react";
import { Pin, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { http } from "@/lib/api/client";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasRunValue } from "@/lib/api/canvas-runs";
import type { CanvasEdge } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasRunResult } from "../runs/run-result";
import { CanvasResourceRead } from "../forms/resource-editor";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";

export function CanvasFixedOutput({ edge, editable }: { edge: CanvasEdge; editable: boolean }) {
  const store = useCanvasStoreApi();
  const canvasId = useCanvas((state) => state.snapshot!.document.id);
  const projectId = useCanvas((state) => state.snapshot!.document.projectId);
  const contentRevision = useCanvas((state) => state.contentRevision);
  const key = `${canvasId}:${edge.id}:${JSON.stringify(edge.output)}`;
  const [loaded, setLoaded] = useState<{ key: string; value: CanvasRunValue | null; error: string | null } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<CanvasMediaPreview | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let sequence = 0;
    const read = async () => {
      const current = ++sequence;
      try {
        const value = await http.get<never, CanvasRunValue>(`/api/canvases/${canvasId}/fixed-outputs/${edge.id}`, { signal: controller.signal });
        if (!controller.signal.aborted && current === sequence) setLoaded({ key, value, error: null });
      } catch (failure) {
        if (!controller.signal.aborted && current === sequence) setLoaded({ key, value: null, error: getApiErrorMessage(failure) });
      }
    };
    void read();
    const timer = setInterval(() => void read(), 10_000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [canvasId, edge.id, key, contentRevision]);
  const value = loaded?.key === key ? loaded.value : null;
  return (
    <section className="space-y-3 border-y border-border/20 py-4">
      <h3 className="flex items-center gap-2 text-sm font-medium"><Pin className="size-4 text-amber-700 dark:text-amber-300" />固定结果</h3>
      <p className="text-xs text-muted-foreground">后续步骤使用这份结果，前置步骤后续修改不会替换它。{edge.output?.snapshotId ? "此结果已保存在当前画布。" : "来源于已完成的运行。"}</p>
      {value?.type === "text" ? <details className="text-sm">
        <summary className="cursor-pointer py-1">查看固定正文</summary>
        <p className="whitespace-pre-wrap break-words py-3">{value.text || "（空文本）"}</p>
      </details> : value?.resource && value.data && value.resource.version !== null ? <details className="space-y-3">
        <summary className="cursor-pointer py-1 text-sm">查看固定项目内容 · 版本 {value.resource.version}</summary>
        <CanvasResourceRead resource={{ header: { resourceType: value.resource.type, resourceId: value.resource.id,
          projectId, version: value.resource.version, deleted: false }, data: value.data as Record<string, unknown>, canEdit: false }}
          library={[]} onPreview={setPreview} />
      </details> : value && <CanvasRunResult value={value} title="固定结果" sourceNodeId={edge.sourceNodeId} />}
      {loaded?.key !== key && <p className="text-xs text-muted-foreground">正在读取固定结果…</p>}
      {loaded?.key === key && loaded.error && <p role="alert" className="text-xs text-destructive">{loaded.error}</p>}
      <Button variant="outline" size="sm" disabled={!editable} onClick={() => {
        void store.getState().edit("使用最新输入", [{ type: "canvas.edge.replace", edge: { ...edge, output: null }, expectedVersion: edge.version }])
          .catch((failure) => setError(getApiErrorMessage(failure)));
      }}><RefreshCw />使用最新输入</Button>
      {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
      {preview && <CanvasMediaPreviewDialog media={preview} onClose={() => setPreview(null)} />}
    </section>
  );
}
