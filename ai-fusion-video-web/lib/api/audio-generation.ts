import { http } from "./client";

export interface AudioGenerationInput {
  text: string;
  media: string[];
  voice?: string;
}

export interface AudioGenerationTask {
  id: string;
  modelId: number;
  modelName: string;
  state: string;
  result: { text: string; audio: string[] } | null;
  errorMessage: string | null;
  canRetrySave: boolean;
  createTime: string;
  updateTime: string;
}

export const AUDIO_GENERATION_STATES: Record<string, string> = {
  QUEUED: "排队中", RUNNING: "生成中", INGESTING: "正在保存", COMPLETED: "已完成",
  FAILED: "生成失败", NO_OUTPUT: "未生成音频", UNKNOWN: "结果待确认", CANCELLED: "已取消",
};

export function audioGenerationActive(task: AudioGenerationTask) {
  return ["QUEUED", "RUNNING", "INGESTING"].includes(task.state);
}

export const audioGenerationApi = {
  create: (modelId: number, requestId: string, input: AudioGenerationInput) =>
    http.post<never, AudioGenerationTask>("/api/generation/audio/submit", { modelId, requestId, input }),
  list: () => http.get<never, AudioGenerationTask[]>("/api/generation/audio"),
  get: (id: string) => http.get<never, AudioGenerationTask>(`/api/generation/audio/${id}`),
  retrySave: (id: string) => http.post<never, boolean>(`/api/generation/audio/${id}/retry-save`),
};
