import type { CanvasRun, CanvasRunCost, CanvasRunPlan, CanvasRunSummary } from "@/lib/api/canvas-runs";
import type { CanvasNode } from "./types";
import { runLabels } from "./run-state";

const externalKinds = new Set([
  "textGeneration", "imageGeneration", "videoGeneration", "comfyui",
  "scriptParse", "storyToScript", "episodeParse", "storyboardGenerate",
  "assetExtract", "assetImageGenerate", "frameGenerate", "agent",
  "musicGeneration", "sfxGeneration", "mcpTool",
]);

export function usesExternalCreation(plan: CanvasRunPlan, nodes: CanvasNode[]): boolean {
  return plan.nodes.some((step) => {
    if (step.action === "REUSE") return false;
    if (externalKinds.has(step.kind)) return true;
    const original = nodes.find((node) => node.id === step.nodeId);
    return step.kind === "foreach" && externalKinds.has(String(original?.config.operationKind));
  });
}

export function runUsageSummary(cost: CanvasRunCost): string {
  const items = [`最多 ${cost.nodes} 次处理`];
  if (cost.images) items.push(`${cost.images} 张图片`);
  if (cost.videoSeconds) items.push(`${cost.videoSeconds} 秒视频`);
  return items.join(" · ");
}

export function runHistoryOptions(history: CanvasRunSummary[], run: CanvasRun | null) {
  const current = run ? {
    id: run.id,
    status: run.status,
    createdAt: run.createdAt,
    nodes: run.nodes.filter((node) => node.parentNodeRunId === null).length,
  } : null;
  const entries = current && !history.some((item) => item.id === current.id)
    ? [current, ...history]
    : history.map((item) => current?.id === item.id ? current : item);
  return entries.map((item) => ({
    value: item.id,
    label: `${new Date(item.createdAt).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" })} · ${runLabels[item.status]} · ${item.nodes} 个节点`,
  }));
}
