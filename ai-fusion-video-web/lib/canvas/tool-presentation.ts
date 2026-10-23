import { userFacingError } from "@/lib/user-facing-error";
import { runLabels } from "./run-state";
import type { CanvasRunState } from "@/lib/api/canvas-runs";

const summaries: Record<string, string> = {
  get_canvas_context: "已读取当前画布与选中内容。",
  query_canvas_nodes: "已找到相关节点。",
  get_canvas_node_details: "已读取节点内容和连接关系。",
  inspect_canvas_media: "已查看所选素材，助手会据此继续处理。",
  apply_canvas_operations: "画布已更新。",
  edit_canvas_resource: "修改已保存到项目内容。",
  validate_canvas_run: "已检查运行范围和设置。",
  run_canvas_nodes: "运行已开始，可在“运行”面板查看进度和结果。",
  get_canvas_run: "已读取这次运行的进度。",
  await_canvas_run: "正在等待创作结果。",
  cancel_canvas_run: "已提交停止操作，已完成的结果会保留。",
  select_canvas_candidate: "已选择结果，后续步骤将使用这份内容。",
  focus_canvas_nodes: "已定位到相关节点。",
  create_canvas_checkpoint: "已保存画布版本。",
  undo_canvas_command: "已撤销这次画布编辑。",
  bind_canvas_output: "已将选定结果应用到项目。",
  confirm_canvas_writeback: "已处理这次内容更新。",
  arrange_canvas_nodes: "已整理节点布局。",
  get_canvas_models: "已查看可用的创作模型。",
  query_canvas_library: "已查找项目内容与素材。",
  get_canvas_library_resource: "已读取所选内容。",
  query_canvas_outputs: "已查找已有创作结果。",
  get_canvas_output: "已读取所选结果。",
  get_canvas_fixed_output: "已读取当前连线使用的历史结果。",
  import_canvas_sources: "已开始导入素材，可在“素材”面板查看进度。",
  get_canvas_import: "已读取素材导入进度。",
  await_canvas_import: "正在等待素材导入。",
  place_canvas_import: "素材已添加到画布。",
  control_canvas_import: "已更新素材导入任务。",
  get_canvas_writeback: "已读取待应用的内容变化。",
  review_canvas_batch: "已处理这组任务，进度可在“运行”面板查看。",
  get_canvas_mcp_operations: "已查看可用的外部工具。",
  get_canvas_mcp_call: "已读取外部工具的处理结果。",
};

export function isCanvasTool(name: string): boolean {
  return Object.hasOwn(summaries, name);
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

export interface CanvasToolPresentation {
  summary: string;
  tone: "info" | "success" | "warning" | "error";
  facts: string[];
  items: { title: string; detail?: string }[];
}

/** 显示提交后的结果摘要；不展示给模型的协议字段、标识或后续工具指令。 */
export function canvasToolPresentation(name: string, value: unknown): CanvasToolPresentation {
  const data = record(value);
  const run = record(data.run);
  const plan = record(data.plan);
  const state = String(data.state ?? run.status ?? data.status ?? "");
  const result: CanvasToolPresentation = { summary: summaries[name], tone: "success", facts: [], items: [] };
  if (!isCanvasTool(name)) throw new Error("Unknown canvas tool presentation");
  if (data.status === "error") {
    result.summary = userFacingError(typeof data.message === "string" ? data.message : null);
    result.tone = "error";
    return result;
  }
  const stateLabel = runLabels[state as CanvasRunState];
  if (stateLabel) {
    result.summary = stateLabel;
    result.tone = ["FAILED", "PARTIAL_FAILED", "BLOCKED"].includes(state) ? "error"
      : ["WAITING_SELECTION", "WAITING_CONFIRMATION", "UNKNOWN_SUBMISSION", "NEEDS_REVIEW"].includes(state) ? "warning"
        : state === "SUCCEEDED" ? "success" : "info";
  }
  if (state === "WAITING_SELECTION") result.facts.push("在运行面板选择一个候选，即可继续后续步骤。");
  if (state === "WAITING_CONFIRMATION") result.facts.push("请查看待确认的内容变化，再决定是否继续。");
  if (state === "UNKNOWN_SUBMISSION" || state === "NEEDS_REVIEW") result.facts.push("请先核对原任务状态，避免重复生成。");
  if (name === "get_canvas_context") {
    const document = record(data.document);
    if (typeof document.title === "string") result.facts.push(document.title);
    if (count(data.nodeCount) !== null) result.facts.push(`${data.nodeCount} 个节点`);
    if (count(data.edgeCount) !== null) result.facts.push(`${data.edgeCount} 条连线`);
    if (Array.isArray(data.selectedNodes)) result.facts.push(`已选 ${data.selectedNodes.length} 个节点`);
  }
  if (Array.isArray(data.affectedNodes) && data.affectedNodes.length) result.facts.push(`已更新 ${data.affectedNodes.length} 个节点`);
  if (Array.isArray(data.affectedEdges) && data.affectedEdges.length) result.facts.push(`已更新 ${data.affectedEdges.length} 条连线`);
  if (Array.isArray(data.deletedNodeIds) && data.deletedNodeIds.length) result.facts.push(`已移除 ${data.deletedNodeIds.length} 个节点`);
  if (count(data.total) !== null) result.facts.push(`找到 ${data.total} 项内容`);
  if (count(data.totalNodes) !== null) result.facts.push(`共 ${data.totalNodes} 个步骤`);
  if (name === "validate_canvas_run") {
    const issues = Array.isArray(plan.issues) ? plan.issues : [];
    result.summary = issues.length ? `还需完成 ${issues.length} 项设置` : "设置已检查，可以继续运行。";
    result.tone = issues.length ? "warning" : "success";
    result.items = issues.slice(0, 8).map((issue) => ({ title: userFacingError(String(record(issue).message ?? "")) }));
    const cost = record(plan.cost);
    if (count(cost.images)) result.facts.push(`最多 ${cost.images} 张图片`);
    if (count(cost.videoSeconds)) result.facts.push(`最多 ${cost.videoSeconds} 秒视频`);
  }
  const entries = [data.affectedNodes, data.nodes, data.items, data.models].find(Array.isArray);
  if (!result.items.length && Array.isArray(entries)) {
    result.items = entries.slice(0, 6).flatMap((entry) => {
      const item = record(entry);
      const title = item.title ?? item.name ?? item.label;
      if (typeof title !== "string") return [];
      const status = runLabels[String(item.status) as CanvasRunState];
      return [{ title, detail: status }];
    });
  }
  return result;
}
