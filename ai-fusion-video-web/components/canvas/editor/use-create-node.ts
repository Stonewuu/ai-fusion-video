"use client";

import { useCallback, useRef } from "react";
import { useReactFlow } from "@xyflow/react";
import { toast } from "sonner";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { initialConfig } from "@/lib/canvas/graph";
import { freeNodePosition } from "@/lib/canvas/placement";
import type { CanvasAddIntent, CanvasNode, CanvasOperation } from "@/lib/canvas/types";
import { useCanvasStoreApi } from "../canvas-provider";
import type { CanvasNodeChoice } from "./node-menu";

export function useCanvasCreateNode(onInspect: (id: string) => void) {
  const store = useCanvasStoreApi();
  const flow = useReactFlow();
  const creating = useRef(false);
  return useCallback(async (intent: CanvasAddIntent, choice: CanvasNodeChoice) => {
    if (creating.current) return;
    const state = store.getState();
    if (!state.snapshot?.permissions.canEdit || state.snapshot.document.archived || state.pending) return;
    creating.current = true;
    try {
      const capability = choice.capability;
      const width = capability.kind === "group" ? 640 : 300;
      const position = intent.automatic
        ? freeNodePosition(state.snapshot.graph.nodes, intent.position,
            { width, height: Math.max(320, 128 + 32 * Math.max(capability.inputs.length, capability.outputs.length)) },
            Object.fromEntries(flow.getNodes().map((node) => [node.id, node.measured])))
        : intent.position;
      const node: CanvasNode = {
        id: crypto.randomUUID(), kind: capability.kind, schemaVersion: capability.schemaVersion,
        title: capability.label, config: choice.connection?.config ?? initialConfig(capability),
        resource: null, configVersion: 0, layoutVersion: 0,
        layout: { ...position, width, height: capability.kind === "group" ? 400 : null,
          parentId: null, locked: false, collapsed: false },
      };
      const operations: CanvasOperation[] = [];
      if (intent.source) operations.push({ type: "canvas.node.assert", nodeId: intent.source.nodeId,
        expectedConfigVersion: intent.source.configVersion });
      operations.push({ type: "canvas.node.create", node });
      if (intent.source && choice.connection) operations.push({ type: "canvas.edge.create", edge: {
        id: crypto.randomUUID(), kind: "execution", sourceNodeId: intent.source.nodeId,
        sourcePort: intent.source.portId, targetNodeId: node.id, targetPort: choice.connection.port.id,
        inputIndex: 0, output: null, version: 0,
      } });
      await state.edit(intent.source ? "添加并连接节点" : "添加节点", operations);
      store.getState().select([node.id]);
      store.getState().focus([node.id]);
      onInspect(node.id);
    } catch (error) {
      toast.error(getApiErrorMessage(error));
    } finally {
      creating.current = false;
    }
  }, [flow, onInspect, store]);
}
