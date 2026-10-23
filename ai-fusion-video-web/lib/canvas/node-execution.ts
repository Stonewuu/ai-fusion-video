import type { CanvasNodeActivity, CanvasNodeRun, CanvasRunPlan } from "@/lib/api/canvas-runs";
import type { GenerationExecution } from "@/lib/api/generation-execution";
import { generationTerminalStates } from "@/lib/generation-execution-view";
import type { CanvasCapability, CanvasGraph } from "./types";
import { runFinished } from "./run-state";

export interface CanvasNodeStartResult {
  started: boolean;
  error: string | null;
  review?: { nodeIds: string[]; mode: "DEPENDENCIES"; reuse: Record<string, string> };
}

export function canRunCanvasNode(capability: CanvasCapability) {
  return !["none", "constant", "resource"].includes(capability.executor);
}

/** 固定输出已经有自己的快照，不沿该连线重新执行源节点。 */
export function canvasNodeDependencies(graph: CanvasGraph, nodeId: string): string[] {
  const incoming = new Map<string, string[]>();
  for (const edge of graph.edges) {
    if (edge.kind !== "execution" || edge.output) continue;
    incoming.set(edge.targetNodeId, [...(incoming.get(edge.targetNodeId) ?? []), edge.sourceNodeId]);
  }
  const visited = new Set<string>();
  const queue = [nodeId];
  while (queue.length) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    visited.add(current);
    queue.push(...(incoming.get(current) ?? []));
  }
  return [...visited];
}

export function reusableNodeDependencies(graph: CanvasGraph, nodeId: string, activities: CanvasNodeActivity[]) {
  const dependencies = new Set(canvasNodeDependencies(graph, nodeId));
  dependencies.delete(nodeId);
  return Object.fromEntries(activities.filter(({ node }) => dependencies.has(node.nodeId) && node.status === "SUCCEEDED"
    && node.derivedState === "CURRENT" && !node.contentUnavailable).map(({ node }) => [node.nodeId, node.id]));
}

export function activePlanNode(plan: CanvasRunPlan, activities: CanvasNodeActivity[]) {
  const executing = new Set(plan.nodes.filter((node) => node.action === "EXECUTE").map((node) => node.nodeId));
  return activities.find(({ node }) => executing.has(node.nodeId) && !runFinished(node.status));
}

export function generationNodeProgress(execution: GenerationExecution) {
  const finished = execution.items.filter((item) => generationTerminalStates.has(item.state)).length;
  const succeeded = execution.items.filter((item) => item.state === "SUCCEEDED").length;
  const active = execution.items.find((item) => !generationTerminalStates.has(item.state));
  const phases: Record<string, string> = {
    NEW: "排队中", PREPARING: "准备素材", SUBMITTING: "等待生成结果", POLLING: "正在生成", INGESTING: "保存生成结果", FINISHED: "已结束",
  };
  return { completed: finished, succeeded, total: execution.items.length,
    label: active ? phases[active.phase] ?? "正在处理" : "处理完成",
    error: execution.items.find((item) => item.errorMessage)?.errorMessage ?? null };
}

export function batchNodeProgress(node: CanvasNodeRun, children: readonly CanvasNodeRun[]) {
  const total = node.inputs?.values.items?.length ?? children.length;
  return { completed: children.filter((item) => runFinished(item.status)).length, total };
}
