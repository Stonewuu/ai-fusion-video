"use client";
import { useEffect, useRef, useState } from "react";
import { useReactFlow } from "@xyflow/react";
import { toast } from "sonner";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { autoLayoutOperations } from "@/lib/canvas/layout/operations";
import { layoutRequestKey } from "@/lib/canvas/layout/request-key";
import { runLayoutWorker } from "@/lib/canvas/layout/worker-client";
import type { LayoutRelation, LayoutRequest, LayoutResult } from "@/lib/canvas/layout/types";
import type { CanvasStore } from "@/lib/store/canvas-store";
import { captureLayoutGeometry, captureLayoutLabels } from "./layout-geometry";

export function useCanvasAutoLayout({ store, hidden, relations, relationLoading, relationError, onResult }: {
  store: CanvasStore;
  hidden: Set<string>;
  relations: LayoutRelation[];
  relationLoading: boolean;
  relationError: string | null;
  onResult: (request: LayoutRequest, result: LayoutResult) => void;
}) {
  const flow = useReactFlow();
  const active = useRef<AbortController | null>(null);
  const [arranging, setArranging] = useState(false);
  useEffect(() => () => active.current?.abort(), []);
  useEffect(() => {
    active.current?.abort("relations-changed");
  }, [relations, relationLoading, relationError]);

  const arrange = async () => {
    if (active.current) {
      return;
    }
    const state = store.getState();
    const snapshot = state.snapshot!;
    if (state.pending || !snapshot.permissions.canEdit || snapshot.document.archived) {
      return;
    }
    if (relationLoading || relationError) {
      toast.error(relationLoading ? "正在读取项目关系，请稍后再整理" : `项目关系未能读取：${relationError}`);
      return;
    }
    const controller = new AbortController();
    active.current = controller;
    setArranging(true);
    let unsubscribe = () => {};
    try {
      const request: LayoutRequest = {
        graph: snapshot.graph, selection: state.selection,
        geometry: captureLayoutGeometry(snapshot.graph, state.catalog!, flow),
        relations, hidden: [...hidden],
        labels: captureLayoutLabels(flow),
      };
      const before = layoutRequestKey(request);
      unsubscribe = store.subscribe((next) => {
        if (next.pending || !next.snapshot || !next.snapshot.permissions.canEdit
          || next.snapshot.document.archived || next.snapshot.document.revision !== snapshot.document.revision
          || next.selection.join(",") !== state.selection.join(",")) {
          controller.abort("canvas-changed");
        }
      });
      const result = await runLayoutWorker(request, "arrange", controller.signal);
      const current = store.getState();
      const geometry = captureLayoutGeometry(current.snapshot!.graph, current.catalog!, flow);
      const nodes = new Map(current.snapshot!.graph.nodes.map((node) => [node.id, node]));
      const inMotion = flow.getNodes().some((node) => node.dragging || node.resizing
        || node.position.x !== nodes.get(node.id)!.layout.x || node.position.y !== nodes.get(node.id)!.layout.y);
      if (inMotion || before !== layoutRequestKey({ ...request, graph: current.snapshot!.graph, geometry })) {
        throw new Error("节点位置或尺寸已经变化，请结束当前操作后重新整理");
      }
      const operations = autoLayoutOperations(request, result);
      unsubscribe();
      if (operations.length) {
        await store.getState().edit("自动整理", operations);
      }
      const updates = new Map(operations.filter((operation) => operation.type === "canvas.node.layout")
        .map((operation) => [operation.nodeId, operation.layout]));
      onResult({ ...request, graph: { ...request.graph, nodes: request.graph.nodes.map((node) => ({
        ...node, layout: updates.get(node.id) ?? node.layout,
      })) } }, result);
      toast.success(operations.length ? `已整理 ${operations.length} 个节点，可撤销` : "当前布局已整理好");
    } catch (error) {
      if (!controller.signal.aborted) {
        toast.error(getApiErrorMessage(error));
      } else if (["canvas-changed", "relations-changed"].includes(controller.signal.reason)) {
        toast.info("画布已更改，本次整理已取消");
      }
    } finally {
      unsubscribe();
      if (active.current === controller) {
        active.current = null;
        setArranging(false);
      }
    }
  };
  return { arranging, arrange: () => void arrange(), cancel: () => active.current?.abort() };
}
