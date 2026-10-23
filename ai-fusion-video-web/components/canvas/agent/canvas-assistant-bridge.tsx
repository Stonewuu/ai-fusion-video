"use client";

import { useEffect, type RefObject } from "react";
import type { ReactFlowInstance } from "@xyflow/react";
import { canvasApi } from "@/lib/api/canvas";
import { useAuthStore } from "@/lib/store/auth-store";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { attachCanvasAssistant, saveCanvasAssistantDrafts } from "@/lib/canvas/assistant-session";
import { worldPosition } from "@/lib/canvas/graph";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import type { CanvasFlowNode } from "../editor/canvas-node";

export function CanvasAssistantBridge({ flow, area }: {
  flow: ReactFlowInstance<CanvasFlowNode>;
  area: RefObject<HTMLDivElement | null>;
}) {
  const store = useCanvasStoreApi();
  const userId = useAuthStore((state) => state.user?.id);
  const resourceScope = useResourceChangeStore((state) => state.scope);
  const canvasId = useCanvas((state) => state.canvasId);
  const title = useCanvas((state) => state.snapshot?.document.title ?? "");
  const projectId = useCanvas((state) => state.snapshot?.document.projectId ?? null);
  const count = useCanvas((state) => state.selection.length);
  useEffect(() => {
    if (!userId) return;
    return attachCanvasAssistant({
      canvasId, title, projectId, userId, selectedCount: count,
      focus: (ids) => {
        const present = new Set(store.getState().snapshot?.graph.nodes.map((node) => node.id));
        store.getState().focus(ids.filter((id) => present.has(id)));
      },
      prepare: async () => {
        const initial = store.getState();
        if (!initial.snapshot || initial.pending || initial.saving || initial.sync !== "connected")
          throw new Error("请先完成画布保存与同步，再发送给助手。");
        const selected = [...initial.selection];
        if (selected.length > 200) throw new Error("单次最多向助手提供 200 个选中节点，请缩小选区。");
        const relevant = initial.snapshot.graph.nodes.filter((node) => selected.includes(node.id));
        const keys = [...new Set(relevant.flatMap((node) => [
          `canvas-node-draft:${userId}:${canvasId}:${node.id}`,
          ...(node.resource ? [`canvas-draft:${userId}:${canvasId}:${node.resource.type}:${node.resource.id}`] : []),
        ]))];
        await saveCanvasAssistantDrafts(keys);
        await store.getState().load();
        if (useAuthStore.getState().user?.id !== userId || useResourceChangeStore.getState().scope !== resourceScope)
          throw new Error("账号或团队已经切换，请在当前范围重新发送。");
        const current = store.getState();
        if (!current.snapshot || current.pending || current.saving || current.sync !== "connected")
          throw new Error("画布尚未完成同步，消息与草稿已保留。");
        if (selected.some((id) => !current.snapshot!.graph.nodes.some((node) => node.id === id)))
          throw new Error("发送前选中节点发生了变化，请重新选择。");
        const snapshot = current.snapshot;
        const focused = selected.length === 1 ? snapshot.graph.nodes.find((node) => node.id === selected[0])?.resource : null;
        const resource = focused ? await canvasApi.resource(canvasId, focused.type, focused.id) : null;
        if (useAuthStore.getState().user?.id !== userId || useResourceChangeStore.getState().scope !== resourceScope)
          throw new Error("账号或团队已经切换，请在当前范围重新发送。");
        const viewport = flow.getViewport();
        const measured = new Map(flow.getNodes().map((node) => [node.id, node.measured]));
        const width = area.current?.clientWidth ?? 0, height = area.current?.clientHeight ?? 0;
        const visibleNodeIds = snapshot.graph.nodes.filter((node) => {
          const position = worldPosition(node, snapshot.graph.nodes);
          const x = position.x * viewport.zoom + viewport.x, y = position.y * viewport.zoom + viewport.y;
          return x < width && y < height
            && x + (measured.get(node.id)?.width ?? node.layout.width ?? 300) * viewport.zoom > 0
            && y + (measured.get(node.id)?.height ?? node.layout.height ?? 280) * viewport.zoom > 0;
        }).slice(0, 80).map((node) => node.id);
        return { title: snapshot.document.title, projectId: snapshot.document.projectId, context: {
          canvasId, selectedNodeIds: selected, visibleNodeIds, lastSequence: snapshot.lastSequence,
          focusedResource: resource ? { type: resource.header.resourceType, id: resource.header.resourceId, version: resource.header.version } : null,
        } };
      },
    });
  }, [area, canvasId, count, flow, projectId, resourceScope, store, title, userId]);
  return null;
}
