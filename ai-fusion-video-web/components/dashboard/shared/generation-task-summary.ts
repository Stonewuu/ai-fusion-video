import { generationStateLabels } from "@/lib/generation-execution-view";

export function generationTaskSummary(state: string, accepted: boolean) {
  if (accepted) {
    return { label: "已受理，等待生成", description: "生成完成后继续处理。", tone: "pending" };
  }
  if (state === "SUCCEEDED") {
    return { label: "生成完成", description: "", tone: "success" };
  }
  if (state === "UNKNOWN_SUBMISSION" || state === "NEEDS_REVIEW") {
    return { label: "生成结果需要核实", description: "保留当前任务，核实后再处理。", tone: "warning" };
  }
  if (state === "PARTIAL_FAILED" || state === "FAILED" || state === "CANCELLED") {
    return { label: generationStateLabels[state], description: "", tone: "warning" };
  }
  return { label: generationStateLabels[state] ?? state, description: "尚未完成，等待生成结果。", tone: "pending" };
}
