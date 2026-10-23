"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { EdgeChange, NodeChange, ReactFlowProps } from "@xyflow/react";
import type { CanvasNode, CanvasOperation } from "@/lib/canvas/types";
import type { CanvasStore } from "@/lib/store/canvas-store";
import type { CanvasFlowNode } from "./canvas-node";

export type Transient = {
  position?: { x: number; y: number };
  width?: number;
  height?: number;
};
export type Measurements = Record<string, { width: number; height: number }>;

export function useCanvasFlowMotion(
  store: CanvasStore,
  perform: (label: string, operations: CanvasOperation[]) => void,
) {
  const [transient, setTransient] = useState<Record<string, Transient>>({});
  const [measurements, setMeasurements] = useState<Measurements>({});
  const dragStart = useRef(new Map<string, CanvasNode>());

  useEffect(
    () =>
      store.subscribe((state, previous) => {
        if (previous.pending && !state.pending) setTransient({});
      }),
    [store],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<CanvasFlowNode>[]) => {
      const selected = new Set(store.getState().selection);
      let selectionChanged = false;
      const positions: Record<string, Transient> = {};
      const measured: Measurements = {};
      for (const change of changes) {
        if (change.type === "select") {
          if (change.selected) selected.add(change.id);
          else selected.delete(change.id);
          selectionChanged = true;
        }
        if (change.type === "position" && change.position && !store.getState().pending)
          positions[change.id] = {
            ...positions[change.id],
            position: change.position,
          };
        if (change.type === "dimensions" && change.dimensions) {
          // React Flow needs measured dimensions on every controlled node. They are
          // view state, and must survive selection, saves and graph projections.
          measured[change.id] = change.dimensions;
          if (change.setAttributes && !store.getState().pending)
            positions[change.id] = {
              ...positions[change.id],
              ...change.dimensions,
            };
        }
      }
      if (selectionChanged)
        store.getState().select([...selected], store.getState().edgeSelection);
      if (Object.keys(measured).length)
        setMeasurements((current) => {
          if (
            Object.entries(measured).every(
              ([id, size]) =>
                current[id]?.width === size.width &&
                current[id]?.height === size.height,
            )
          )
            return current;
          return { ...current, ...measured };
        });
      if (Object.keys(positions).length)
        setTransient((current) => {
          const next = { ...current };
          for (const [id, value] of Object.entries(positions))
            next[id] = { ...next[id], ...value };
          return next;
        });
    },
    [store],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const selected = new Set(store.getState().edgeSelection);
      for (const change of changes) {
        if (change.type !== "select" || change.id.startsWith("relation:"))
          continue;
        if (change.selected) selected.add(change.id);
        else selected.delete(change.id);
      }
      store.getState().select(store.getState().selection, [...selected]);
    },
    [store],
  );

  const onNodeDragStart: NonNullable<
    ReactFlowProps<CanvasFlowNode>["onNodeDragStart"]
  > = (_event, _node, dragged) => {
    if (store.getState().pending) { dragStart.current.clear(); return; }
    dragStart.current = new Map(
      dragged.map((node) => [node.id, node.data.node]),
    );
  };
  const onNodeDragStop: NonNullable<
    ReactFlowProps<CanvasFlowNode>["onNodeDragStop"]
  > = (_event, _node, dragged) => {
    if (store.getState().pending) { dragStart.current.clear(); return; }
    const operations: CanvasOperation[] = dragged.flatMap((node) => {
      const before = dragStart.current.get(node.id);
      if (
        !before ||
        (before.layout.x === node.position.x &&
          before.layout.y === node.position.y)
      )
        return [];
      return [
        {
          type: "canvas.node.layout",
          nodeId: node.id,
          expectedLayoutVersion: before.layoutVersion,
          layout: { ...before.layout, x: node.position.x, y: node.position.y },
        },
      ];
    });
    perform("移动节点", operations);
    dragStart.current.clear();
  };

  return {
    transient,
    setTransient,
    measurements,
    onNodesChange,
    onEdgesChange,
    onNodeDragStart,
    onNodeDragStop,
  };
}
