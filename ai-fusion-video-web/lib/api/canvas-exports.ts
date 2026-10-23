import { http, resolveMediaUrl } from "./client";

export type CanvasExportFormat = "MARKDOWN" | "CSV" | "JSON" | "ZIP" | "MEDIA";
export type CanvasExportScope = "CANVAS" | "SELECTION" | "PROJECT";
export interface CanvasExportOptions {
  format: CanvasExportFormat;
  scope: CanvasExportScope;
  nodeIds: string[];
  expectedRevision: number;
}
export interface CanvasExportPreview {
  hash: string;
  documentRevision: number;
  nodes: number;
  businessObjects: number;
  mediaFiles: number;
  mediaBytes: number;
  warnings: string[];
}
export interface CanvasExportSummary {
  id: string;
  canvasId: string;
  format: CanvasExportFormat;
  status:
    | "QUEUED"
    | "RUNNING"
    | "CANCEL_REQUESTED"
    | "SUCCEEDED"
    | "FAILED"
    | "CANCELLED";
  filename: string;
  byteSize: number | null;
  contentHash: string | null;
  revision: number;
  attempts: number;
  error: string | null;
  createdAt: string;
  updatedAt: string;
  actorName: string;
  canCancel: boolean;
  mediaFiles: number;
  businessObjects: number;
  contentUnavailable: boolean;
}
export const canvasExportsApi = {
  preview: (
    id: string,
    options: CanvasExportOptions,
  ): Promise<CanvasExportPreview> =>
    http.post(`/api/canvases/${id}/exports/preview`, options),
  submit: (
    id: string,
    body: {
      requestId: string;
      options: CanvasExportOptions;
      previewHash: string;
    },
  ): Promise<CanvasExportSummary> =>
    http.post(`/api/canvases/${id}/exports`, body),
  list: (
    id: string,
    page: number,
    signal?: AbortSignal,
  ): Promise<CanvasExportSummary[]> =>
    http.get(`/api/canvases/${id}/exports`, { params: { page }, signal }),
  cancel: (canvasId: string, id: string): Promise<boolean> =>
    http.post(`/api/canvases/${canvasId}/exports/${id}/cancel`),
  download: async (canvasId: string, id: string) => {
    const issued = await http.post<never, { url: string; expiresAt: string }>(
      `/api/canvases/${canvasId}/exports/${id}/access`,
    );
    const url = resolveMediaUrl(issued.url);
    if (!url) throw new Error("未取得交付文件下载地址");
    const link = document.createElement("a");
    link.href = url;
    link.rel = "noopener noreferrer";
    link.click();
  },
};
