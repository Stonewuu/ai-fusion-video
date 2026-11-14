import type { GenerationExecution, GenerationExecutionItem } from "@/lib/api/generation-execution";
import { generationStateLabels } from "@/lib/generation-execution-view";
import { formatGenerationDateTime } from "./generation-utils";

export function executionStatus(data: GenerationExecution) {
  if (data.state === "FAILED" && data.items.some((item) => item.state === "UNKNOWN_SUBMISSION")) {
    return "受理待确认";
  }
  return generationStateLabels[data.state] ?? "状态无法识别";
}

export function executionTime(value: string | null) {
  return value === null ? "—" : formatGenerationDateTime(value);
}

export function executionProviderTask(item: GenerationExecutionItem) {
  if (item.providerTaskId) {
    return item.providerTaskId;
  }
  if (item.attempts.some((attempt) => attempt.acknowledged)) {
    return "—";
  }
  return item.attempts.some((attempt) => attempt.submitted) ? "未返回" : "尚未提交";
}

export function executionGuidance(item: GenerationExecutionItem) {
  if (item.state === "UNKNOWN_SUBMISSION") {
    return item.providerTaskId
      ? "暂时无法确认生成结果，请用供应商任务编号核对任务状态。"
      : "未取得供应商任务编号，暂时无法查询结果。请先在供应商平台核对这次生成，再决定是否重新生成。";
  }
  if (item.availableActions.includes("RETRY_INGEST")) {
    return "生成结果已收到，保存尚未完成。可重新保存结果。";
  }
  if (item.state === "FAILED") {
    return item.attempts.some((attempt) => attempt.submitted)
      ? "本次生成未完成，请查看错误信息后调整输入或模型配置。"
      : "本次任务在提交前停止，请查看错误信息后调整输入或模型配置。";
  }
  if (item.state === "SUCCEEDED") {
    return "生成结果已保存，可在创作记录中查看。";
  }
  if (item.state === "CANCELLED") {
    return "本次任务已取消。";
  }
  if (item.cancelRequested) {
    return "正在确认取消结果。";
  }
  return null;
}

export function executionParameters(data: Pick<GenerationExecution, "kind" | "input">) {
  const p = data.input.parameters;
  const video = data.kind === "VIDEO";
  const parameters = [
    { label: "数量", value: `${p.count} ${video ? "段" : "张"}` },
    { label: "画面比例", value: p.ratio ?? "—" },
    { label: "分辨率", value: p.resolution ?? "—" },
  ];
  if (video) {
    parameters.push({ label: "时长", value: p.duration === -1 ? "自动" : p.duration == null ? "—" : `${p.duration} 秒` });
  }
  if (p.width != null && p.height != null) {
    parameters.push({ label: "尺寸", value: `${p.width} × ${p.height}` });
  }
  for (const [key, label] of [["generateAudio", "声音"], ["cameraFixed", "固定镜头"], ["watermark", "水印"]] as const) {
    if (p[key] != null) {
      parameters.push({ label, value: p[key] ? "开启" : "关闭" });
    }
  }
  if (p.seed != null) {
    parameters.push({ label: "随机种子", value: String(p.seed) });
  }
  return parameters;
}

export const executionReferenceLabels = {
  firstFrame: "首帧", lastFrame: "尾帧", image: "参考图片", video: "参考视频", audio: "参考音频",
};
