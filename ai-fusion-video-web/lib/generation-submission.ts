import type {
  ImageGenerationSubmitReq,
  VideoGenerationSubmitReq,
} from "./api/generation";
import type { GenerationScope } from "./api/generation-execution";

export interface PendingGenerationSubmission {
  schemaVersion: 1;
  mode: "image" | "video";
  modelName: string;
  createdAt: string;
  request: ImageGenerationSubmitReq | VideoGenerationSubmitReq;
}

export function generationSubmissionKey(
  mode: "image" | "video",
  scope: GenerationScope,
) {
  return `fusion-generation-pending:v1:${scope.userId}:${scope.teamId ?? "personal"}:${mode}`;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasVideoMedia(request: Record<string, unknown>): boolean {
  if ([request.firstFrameImageUrl, request.lastFrameImageUrl].some((value) => typeof value === "string" && !!value.trim())) return true;
  return [request.referenceImageUrls, request.referenceVideoUrls, request.referenceAudioUrls].some((value) => {
    if (typeof value !== "string" || !value) return false;
    const items: unknown = JSON.parse(value);
    return Array.isArray(items) && items.some((item) => typeof item === "string" && !!item.trim());
  });
}

export function readPendingGeneration(
  raw: string | null,
  mode: "image" | "video",
  scope: GenerationScope,
): PendingGenerationSubmission | null {
  if (raw === null) return null;
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error("本地提交记录无法解析，请先核对创作记录。");
  }
  if (
    !record(value) ||
    value.schemaVersion !== 1 ||
    value.mode !== mode ||
    typeof value.modelName !== "string" ||
    typeof value.createdAt !== "string" ||
    !Number.isFinite(Date.parse(value.createdAt)) ||
    !record(value.request)
  ) {
    throw new Error("本地提交记录的格式无效，请先核对创作记录。");
  }
  const request = value.request;
  if (
    !record(request.scope) ||
    request.scope.userId !== scope.userId ||
    request.scope.teamId !== scope.teamId ||
    typeof request.requestId !== "string" ||
    !/^[a-zA-Z0-9_.:-]{1,128}$/.test(request.requestId) ||
    typeof request.prompt !== "string" ||
    (!request.prompt.trim() && !(mode === "video" && hasVideoMedia(request))) ||
    request.prompt.length > 100_000 ||
    !Number.isSafeInteger(request.modelId) ||
    Number(request.modelId) <= 0 ||
    !Number.isSafeInteger(request.count) ||
    Number(request.count) < 1 ||
    Number(request.count) > 32
  ) {
    throw new Error(
      "本地提交记录与当前账号、团队或生成参数不一致，不能重新发送。",
    );
  }
  return value as unknown as PendingGenerationSubmission;
}

/** 只清理对应请求，迟到响应不能删除其他窗口的新提交。 */
export function clearPendingGeneration(
  storage: Pick<Storage, "getItem" | "removeItem">,
  key: string,
  requestId: string,
) {
  const raw = storage.getItem(key);
  if (raw === null) return;
  const value: unknown = JSON.parse(raw);
  if (
    !record(value) ||
    !record(value.request) ||
    value.request.requestId !== requestId
  ) {
    throw new Error("本地提交记录已变化，请重新读取状态。");
  }
  storage.removeItem(key);
}
