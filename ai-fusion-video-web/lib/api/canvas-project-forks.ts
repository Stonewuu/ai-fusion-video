import { http } from "./client";

export interface CanvasProjectForkPreview {
  digest: string;
  revision: number;
  nodes: number;
  businessObjects: number;
  historicalObjects: number;
  mediaFiles: number;
  mediaBytes: number;
  warnings: string[];
  ownership: { type: 1 | 2; id: number; name: string };
}
export interface CanvasProjectForkRequest {
  requestId: string;
  sourceCanvasId: string;
  expectedRevision: number;
  digest: string;
  canvasId: string;
  title: string;
}
export const canvasProjectForksApi = {
  preview: (canvas: string, expectedRevision: number): Promise<CanvasProjectForkPreview> =>
    http.get(`/api/canvases/${canvas}/project-forks/preview`, { params: { expectedRevision } }),
  copy: (request: CanvasProjectForkRequest): Promise<{ result: { id: number; canvasId: string; title: string } }> =>
    http.post(`/api/canvases/${request.sourceCanvasId}/project-forks`, request),
};
