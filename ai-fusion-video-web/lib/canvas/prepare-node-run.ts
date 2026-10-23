import { canvasRunApi } from "@/lib/api/canvas-runs";
import type { GenerationScope } from "@/lib/api/generation-execution";
import type { CanvasStore } from "@/lib/store/canvas-store";
import { saveCanvasAssistantDrafts } from "./assistant-session";
import { activePlanNode, canRunCanvasNode, canvasNodeDependencies, reusableNodeDependencies } from "./node-execution";
import { defaultRunBudget } from "./run-state";

/** 节点入口沿用正式预检与草稿保存，不读取选区来扩大执行范围。 */
export async function prepareCanvasNodeRun(store: CanvasStore, userId: number, scope: GenerationScope, nodeId: string) {
  const initial = store.getState();
  const canvasId = initial.canvasId;
  if (!initial.snapshot || initial.pending || initial.saving || initial.sync !== "connected")
    throw new Error("请等待画布保存并恢复同步后再运行。");
  if (!initial.snapshot.permissions.canExecute || initial.snapshot.document.archived)
    throw new Error("当前画布没有运行权限。");
  const relevant = new Set(canvasNodeDependencies(initial.snapshot.graph, nodeId));
  const keys = initial.snapshot.graph.nodes.filter((node) => relevant.has(node.id)).flatMap((node) => [
    `canvas-node-draft:${userId}:${canvasId}:${node.id}`,
    ...(node.resource ? [`canvas-draft:${userId}:${canvasId}:${node.resource.type}:${node.resource.id}`] : []),
  ]);
  await saveCanvasAssistantDrafts([...new Set(keys)], "请先保存或处理相关节点的草稿，再运行。");
  const current = store.getState();
  if (!current.snapshot || current.pending || current.saving || current.sync !== "connected")
    throw new Error("画布尚未完成同步，请稍后再试。");
  const target = current.snapshot.graph.nodes.find((node) => node.id === nodeId);
  const capability = current.catalog?.nodes.find((node) => node.kind === target?.kind);
  if (!target || !capability || !canRunCanvasNode(capability)) throw new Error("此节点不需要单独运行。");
  const latest = await canvasRunApi.nodeActivities(canvasId);
  const reuse = reusableNodeDependencies(current.snapshot.graph, nodeId, latest);
  const plan = await canvasRunApi.plan(canvasId, {
    expectedRevision: current.snapshot.document.revision,
    mode: "DEPENDENCIES", nodeIds: [nodeId], reuse, budget: defaultRunBudget, scope,
  });
  if (plan.issues.length) {
    const titles = new Map(plan.nodes.map((node) => [node.nodeId, node.title]));
    const issues = plan.issues.filter((issue) => issue.code !== "EMPTY_SCOPE" || plan.issues.length === 1);
    throw new Error(issues.map((issue) => `${issue.nodeId && issue.nodeId !== nodeId ? `${titles.get(issue.nodeId) ?? "前置节点"}：` : ""}${issue.message}`).join("；"));
  }
  const active = activePlanNode(plan, latest);
  if (active) throw new Error(`“${active.node.title}”正在运行，请等待完成。`);
  return { plan, reuse };
}
