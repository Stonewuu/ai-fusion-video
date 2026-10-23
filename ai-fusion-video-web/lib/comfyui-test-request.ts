import type {
  ComfyUiBindingValueType,
  ComfyUiTestRequest,
  ComfyUiTestRun,
} from "@/lib/api/comfyui-workflow";
import type { GenerationScope } from "@/lib/api/generation-execution";

export interface PendingComfyUiTest {
  schemaVersion: 1;
  createdAt: string;
  request: ComfyUiTestRequest;
}
export function comfyUiTestKey(scope: GenerationScope, versionId: number) {
  return `comfyui-test-request:${scope.userId}:${scope.teamId ?? "personal"}:${versionId}`;
}
export function readPendingComfyUiTest(
  raw: string | null,
  scope: GenerationScope,
  versionId: number,
): PendingComfyUiTest | null {
  if (raw === null) return null;
  const value = JSON.parse(raw) as PendingComfyUiTest;
  const request = value?.request;
  if (
    value?.schemaVersion !== 1 ||
    !Number.isFinite(Date.parse(value.createdAt)) ||
    request?.scope?.userId !== scope.userId ||
    (request.scope.teamId ?? null) !== (scope.teamId ?? null) ||
    request.versionId !== versionId ||
    !/^[A-Za-z0-9_.:-]{1,96}$/.test(request.requestId) ||
    !/^[a-f0-9]{64}$/.test(request.workflowHash) ||
    request.inputs === null ||
    typeof request.inputs !== "object" ||
    Array.isArray(request.inputs)
  ) {
    throw new Error(
      "本地试运行记录损坏，已阻止新提交。请先查看服务器任务历史。",
    );
  }
  return value;
}
export function comfyUiTestInput(
  raw: string,
  type: ComfyUiBindingValueType,
): unknown {
  const text = raw.trim();
  if (type === "integer" || type === "number") {
    const value = Number(text);
    if (
      !Number.isFinite(value) ||
      (type === "integer" && !Number.isSafeInteger(value))
    )
      throw new Error("请输入有效的数值，整数字段不能包含小数。");
    return value;
  }
  if (type === "boolean") {
    if (text !== "true" && text !== "false")
      throw new Error("开关值必须为 true 或 false。");
    return text === "true";
  }
  if (type === "string_list" || type.startsWith("uploaded_")) {
    // Data URI 与 URL 都可能含逗号；界面约定一行一项。
    return text
      .split(/\r?\n/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return text;
}
export const comfyUiTestActive = (run: ComfyUiTestRun) =>
  ["QUEUED", "WAITING", "RUNNING"].includes(run.status);
export function comfyUiTestStatus(run: ComfyUiTestRun) {
  if (run.status === "NEEDS_REVIEW") return "需要核实";
  if (run.status === "SUCCEEDED") return "试运行成功";
  if (run.status === "FAILED") return "试运行失败";
  if (run.status === "CANCELLED") return "已取消";
  if (run.cancelRequested) return "等待远端取消";
  if (
    run.status === "QUEUED" ||
    (run.status === "WAITING" && run.phase === "PREPARING")
  )
    return "等待试运行";
  return {
    PREPARING: "准备输入",
    SUBMITTING: "正在提交",
    POLLING: "远端处理中",
    INGESTING: "保存试运行结果",
  }[run.phase];
}
