import {
  aiModelApi,
  apiConfigApi,
  type AiModel,
  type ApiConfig,
} from "@/lib/api/ai-model";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";

export function exportAiModelsConfigFile(
  configs: readonly ApiConfig[],
  models: readonly AiModel[],
  includeSecrets: boolean,
): void {
  const exportedConfigs = configs.map((config) => ({
    ...config,
    ...(includeSecrets
      ? {}
      : { apiKey: undefined, appSecret: undefined, proxyPassword: undefined }),
  }));
  const payload = {
    version: 1,
    exportedAt: new Date().toISOString(),
    apiConfigs: exportedConfigs,
    models,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `ai-fusion-config-${new Date().toISOString().slice(0, 10)}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function importAiModelsConfigFile(
  file: File,
  providers: readonly ProviderDefinition[],
): Promise<{ configurationsAwaitingKey: number }> {
  const parsed = JSON.parse(await file.text()) as {
    apiConfigs?: ApiConfig[];
    models?: AiModel[];
  };
  if (!Array.isArray(parsed.apiConfigs) || !Array.isArray(parsed.models)) {
    throw new Error("配置文件结构不正确");
  }

  const configIdMap = new Map<number, number>();
  let configurationsAwaitingKey = 0;

  for (const source of parsed.apiConfigs) {
    const requiresKey = providers.find(
      (provider) => provider.id === source.providerId,
    )?.quickSetup;
    const awaitingKey = Boolean(requiresKey && !source.apiKey?.trim());
    if (awaitingKey) {
      configurationsAwaitingKey++;
    }
    const createdId = await apiConfigApi.create({
      name: source.name,
      providerId: source.providerId || undefined,
      platform: source.platform || undefined,
      textProtocol: source.textProtocol || undefined,
      imageProtocol: source.imageProtocol || undefined,
      videoProtocol: source.videoProtocol || undefined,
      apiUrl: source.apiUrl || undefined,
      autoAppendV1Path: source.autoAppendV1Path,
      proxyType: source.proxyType || "none",
      proxyHost: source.proxyHost || undefined,
      proxyPort: source.proxyPort || undefined,
      proxyUsername: source.proxyUsername || undefined,
      proxyPassword: source.proxyPassword || undefined,
      apiKey: source.apiKey || undefined,
      appId: source.appId || undefined,
      appSecret: source.appSecret || undefined,
      status: awaitingKey ? 0 : source.status,
      remark: source.remark || undefined,
    });
    if (typeof source.id === "number") {
      configIdMap.set(source.id, createdId);
    }
  }

  for (const source of parsed.models) {
    const apiConfigId = source.apiConfigId
      ? configIdMap.get(source.apiConfigId)
      : undefined;
    if (!apiConfigId) {
      continue;
    }
    await aiModelApi.create({
      name: source.name,
      code: source.code,
      capabilityPresetCode: source.capabilityPresetCode || undefined,
      modelProtocol: source.modelProtocol || undefined,
      modelType: source.modelType,
      icon: source.icon || undefined,
      description: source.description || undefined,
      sort: source.sort,
      config: source.config || undefined,
      maxConcurrency: source.maxConcurrency || undefined,
      defaultModel: false,
      supportVision: source.supportVision,
      multimodalInputTypes: source.multimodalInputTypes,
      multimodalInputTransports: source.multimodalInputTransports,
      supportReasoning: source.supportReasoning,
      reasoningEffortLevels: source.reasoningEffortLevels ?? [],
      contextWindow: source.contextWindow || undefined,
      apiConfigId,
    });
  }

  return { configurationsAwaitingKey };
}
