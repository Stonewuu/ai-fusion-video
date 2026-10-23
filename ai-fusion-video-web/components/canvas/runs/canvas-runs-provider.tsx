"use client";
import {
  createContext,
  useContext,
  useLayoutEffect,
  useState,
  type ReactNode,
} from "react";
import { useStore } from "zustand";
import type { CanvasNodeRun } from "@/lib/api/canvas-runs";
import {
  createCanvasNodeRunStore,
  type CanvasNodeRunStore,
} from "@/lib/canvas/node-run-store";
import { useCanvasRunsController } from "./use-canvas-runs";

const Context = createContext<ReturnType<
  typeof useCanvasRunsController
> | null>(null);
const NodeContext = createContext<CanvasNodeRunStore | null>(null);
const EMPTY_CHILDREN: readonly CanvasNodeRun[] = [];
export function CanvasRunsProvider({
  userId,
  children,
}: {
  userId: number;
  children: ReactNode;
}) {
  const runtime = useCanvasRunsController(userId);
  const [nodes] = useState(createCanvasNodeRunStore);
  useLayoutEffect(() => {
    nodes.getState().updateActivities(runtime.activities);
  }, [nodes, runtime.activities]);
  useLayoutEffect(() => {
    nodes.getState().setReady(!runtime.loading && !runtime.busy && !runtime.pending && !runtime.corrupt);
  }, [nodes, runtime.loading, runtime.busy, runtime.pending, runtime.corrupt]);
  return (
    <NodeContext.Provider value={nodes}>
      <Context.Provider value={runtime}>{children}</Context.Provider>
    </NodeContext.Provider>
  );
}
export function useCanvasRuns() {
  const value = useContext(Context);
  if (!value) throw new Error("运行组件必须位于画布运行工作区内");
  return value;
}
export function useCanvasNodeRun(nodeId: string) {
  const nodes = useContext(NodeContext);
  if (!nodes) throw new Error("节点运行结果必须位于画布运行工作区内");
  return useStore(nodes, (state) => state.nodes.get(nodeId));
}
export function useCanvasNodeActivity(nodeId: string) {
  const nodes = useContext(NodeContext);
  if (!nodes) throw new Error("节点运行状态必须位于画布运行工作区内");
  return useStore(nodes, (state) => state.activities.get(nodeId));
}
export function useCanvasNodeRunReady() {
  const nodes = useContext(NodeContext);
  if (!nodes) throw new Error("节点运行操作必须位于画布运行工作区内");
  return useStore(nodes, (state) => state.ready);
}

export function useCanvasNodeRunChildren(nodeId: string) {
  const nodes = useContext(NodeContext);
  if (!nodes) throw new Error("节点运行结果必须位于画布运行工作区内");
  return useStore(
    nodes,
    (state) =>
      state.children.get(state.nodes.get(nodeId)?.id ?? "") ?? EMPTY_CHILDREN,
  );
}
