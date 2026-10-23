import type { GenerationScope } from "@/lib/api/generation-execution";
import type { ImportRequest, ImportStatus } from "@/lib/api/media-imports";

export const importLabels: Record<ImportStatus | "PENDING", string> = {
  QUEUED: "等待导入",
  WAITING: "等待继续",
  RUNNING: "正在导入",
  PENDING: "等待导入",
  SUCCEEDED: "已导入",
  PARTIAL_FAILED: "部分未完成",
  FAILED: "导入失败",
  CANCELLED: "已停止",
};
export const importFinished = (state: ImportStatus) =>
  ["SUCCEEDED", "PARTIAL_FAILED", "FAILED", "CANCELLED"].includes(state);
export interface PendingImport {
  version: 1;
  canvasId: string;
  createdAt: string;
  request: ImportRequest;
}
export function importRequestKey(canvasId: string, scope: GenerationScope) {
  return `canvas-media-import:v1:${scope.userId}:${scope.teamId ?? "personal"}:${canvasId}`;
}
export function parseImportRequest(
  raw: string,
  canvasId: string,
  scope: GenerationScope,
): PendingImport {
  const item = JSON.parse(raw) as Partial<PendingImport>;
  const request = item.request;
  if (
    item.version !== 1 ||
    item.canvasId !== canvasId ||
    !item.createdAt ||
    !Number.isFinite(Date.parse(item.createdAt)) ||
    !request ||
    !/^[A-Za-z0-9:._-]{1,128}$/.test(request.requestId) ||
    request.scope?.userId !== scope.userId ||
    request.scope.teamId !== scope.teamId ||
    !Array.isArray(request.sources) ||
    !request.sources.length ||
    request.sources.length > 50 ||
    request.sources.some(
      (s) =>
        !s ||
        typeof s.title !== "string" ||
        !s.title.trim() ||
        s.title.length > 200 ||
        !s.source ||
        ![
          "project",
          "asset",
          "assetItem",
          "shot",
          "storyboardEpisode",
          "imageItem",
          "videoItem",
        ].includes(s.source.type) ||
        !Number.isSafeInteger(s.source.id) ||
        s.source.id <= 0 ||
        !Number.isSafeInteger(s.source.version) ||
        s.source.version < 0 ||
        typeof s.source.field !== "string" ||
        !s.source.field ||
        s.source.field.length > 96,
    ) ||
    (request.position !== null &&
      (!request.position ||
        ![request.position.x, request.position.y].every(
          (n) => Number.isFinite(n) && Math.abs(n) <= 1_000_000,
        )))
  ) {
    throw new Error("本地导入请求已损坏或作用域不符，请先核对导入记录");
  }
  return item as PendingImport;
}
