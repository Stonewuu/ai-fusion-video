import { http } from "./client";
import type { CanvasRunStatistics } from "./canvas-runs";

export interface CanvasActivity extends CanvasRunStatistics {
  from: string;
  until: string;
  runs: number;
  states: Record<string, number>;
  deliveries: { total: number; states: Record<string, number>; bytes: number; unavailable: number };
}
export const canvasActivityApi = {
  get: (canvasId: string, from: string, until: string, signal?: AbortSignal) =>
    http.get<never, CanvasActivity>(`/api/canvases/${canvasId}/activity-statistics`, { params: { from, until }, signal }),
};
