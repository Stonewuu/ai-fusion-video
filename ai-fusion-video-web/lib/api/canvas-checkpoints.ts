import { http } from "./client";
import type { CanvasDocument, CanvasReceipt } from "@/lib/canvas/types";

export interface CanvasCheckpointPlan {
  checkpointId: string;
  title: string;
  expectedRevision: number;
  digest: string;
  nodes: {
    id: string;
    title: string;
    change: "RESTORED" | "MODIFIED" | "REMOVED";
    content: boolean;
    layout: boolean;
  }[];
  upsertedEdges: number;
  removedEdges: number;
  resources: {
    type: string;
    id: number;
    savedVersion: number;
    currentVersion: number;
  }[];
  hasChanges: boolean;
}
export const canvasCheckpointsApi = {
  preview: (canvas: string, id: string): Promise<CanvasCheckpointPlan> =>
    http.get(`/api/canvases/${canvas}/checkpoints/${id}/restore-preview`),
  restore: (
    canvas: string,
    id: string,
    body: { requestId: string; expectedRevision: number; digest: string },
  ): Promise<CanvasReceipt> =>
    http.post(`/api/canvases/${canvas}/checkpoints/${id}/restore`, body),
  fork: (
    canvas: string,
    id: string,
    body: { requestId: string; canvasId: string; title: string },
  ): Promise<CanvasDocument> =>
    http.post(`/api/canvases/${canvas}/checkpoints/${id}/fork`, body),
};
