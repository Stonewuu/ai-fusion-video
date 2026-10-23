"use client";

import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { PLATFORM_LABELS, type AiModel, type ApiConfig } from "@/lib/api/ai-model";

export function matchingApiConfigs(configs: ApiConfig[], models: AiModel[], query: string) {
  const key = query.trim().toLowerCase();
  if (!key) return configs;
  return configs.filter((config) => `${config.name} ${config.platform} ${PLATFORM_LABELS[config.platform ?? ""] ?? ""}`.toLowerCase().includes(key)
    || models.some((model) => model.apiConfigId === config.id && `${model.name} ${model.code}`.toLowerCase().includes(key)));
}

export function ApiConfigSearch({ query, onChange }: { query: string; onChange: (query: string) => void }) {
  return <div className="relative mb-4 max-w-md">
    <Search className="pointer-events-none absolute left-3 top-3 size-4 text-muted-foreground" />
    <Input aria-label="搜索服务或模型" placeholder="搜索服务名称、模型名称或模型 ID" value={query} onChange={(event) => onChange(event.target.value)} className="pl-9" />
  </div>;
}
