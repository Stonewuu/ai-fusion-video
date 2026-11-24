"use client";

import type { AiModel, ApiConfig } from "@/lib/api/ai-model";
import { filterApiConfigsAndModels } from "./ai-models-filter";

export function matchingApiConfigs(
  configs: ApiConfig[],
  models: AiModel[],
  query: string,
): ApiConfig[] {
  return filterApiConfigsAndModels(configs, models, {
    query,
    modelType: 0,
    status: "all",
  }).map((entry) => entry.config);
}
