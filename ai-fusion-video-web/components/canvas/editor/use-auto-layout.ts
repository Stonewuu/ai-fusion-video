"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useReactFlow } from "@xyflow/react";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { autoLayoutOperations } from "@/lib/canvas/layout/operations";
import { layoutRelationsKey, layoutRequestKey } from "@/lib/canvas/layout/request-key";
import { runLayoutWorker } from "@/lib/canvas/layout/worker-client";
import type { LayoutRelation, LayoutRequest } from "@/lib/canvas/layout/types";
import type { CanvasStore } from "@/lib/store/canvas-store";
import { captureLayoutGeometry, captureLayoutLabels } from "./layout-geometry";

export function useCanvasAutoLayout({ store, hidden, relations, relationLoading, relationError }: {
  store: CanvasStore;
  hidden: Set<string>;
  relations: LayoutRelation[];
  relationLoading: boolean;
  relationError: string | null;
}) {
  const flow = useReactFlow();
  const active = useRef<AbortController | null>(null);
  const [error, setError] = useState<string | null>(null);
  const relationsKey = useMemo(() => layoutRelationsKey(relations), [relations]);
  useEffect(() => () => active.current?.abort(), []);
  useEffect(() => {
    active.current?.abort("relations-changed");
  }, [relationsKey, relationLoading, relationError]);

  const arrange = async () => {
    const state = store.getState();
    const snapshot = state.snapshot!;
    if (state.pending || !snapshot.permissions.canEdit || snapshot.document.archived) {
      return;
    }
    if (relationLoading || relationError) {
      return;
    }
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setError(null);
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
      const result = await runLayoutWorker(request, controller.signal);
      controller.signal.throwIfAborted();
      const current = store.getState();
      const geometry = captureLayoutGeometry(current.snapshot!.graph, current.catalog!, flow);
      const nodes = new Map(current.snapshot!.graph.nodes.map((node) => [node.id, node]));
      const inMotion = flow.getNodes().some((node) => node.dragging || node.resizing
        || node.position.x !== nodes.get(node.id)!.layout.x || node.position.y !== nodes.get(node.id)!.layout.y);
      if (inMotion || before !== layoutRequestKey({ ...request, graph: current.snapshot!.graph, geometry })) {
        return;
      }
      const operations = autoLayoutOperations(request, result);
      unsubscribe();
      if (operations.length) {
        await store.getState().edit("自动整理", operations);
      }
    } catch (failure) {
      if (!controller.signal.aborted) {
        setError(getApiErrorMessage(failure));
      }
    } finally {
      unsubscribe();
      if (active.current === controller) {
        active.current = null;
      }
    }
  };
  return { error, arrange: () => void arrange() };
}
