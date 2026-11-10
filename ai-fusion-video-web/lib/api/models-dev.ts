import { http } from "./client";

export interface ModelsDevEntry {
  providerId: string;
  providerName: string;
  modelId: string;
  name: string;
  canonicalModelId: string | null;
  modelType: number;
  contextWindow: number | null;
  outputTokenLimit: number | null;
  inputModalities: string[] | null;
  outputModalities: string[] | null;
  supportReasoning: boolean | null;
  reasoningEffortLevels: string[] | null;
  toolCall: boolean | null;
  updatedAt: string | null;
  status: string | null;
}

export interface ModelsDevSearchResult {
  fetchedAt: string;
  total: number;
  models: ModelsDevEntry[];
  providers: { id: string; name: string }[];
}

export const modelsDevApi = {
  search: (query: string, modelType: number, apiConfigId?: number, providerId?: string) => {
    const params = new URLSearchParams({ query, modelType: String(modelType) });
    if (apiConfigId != null) {
      params.set("apiConfigId", String(apiConfigId));
    }
    if (providerId) {
      params.set("providerId", providerId);
    }
    return http.get<never, ModelsDevSearchResult>(`/api/ai/model/models-dev/search?${params}`);
  },
  refresh: () => http.post<never, boolean>("/api/ai/model/models-dev/refresh"),
};
