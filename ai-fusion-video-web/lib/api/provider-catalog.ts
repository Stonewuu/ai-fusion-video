import { http } from "./client";
import type { ApiConfigSaveReq } from "./ai-model";

export interface ProviderDefinition {
  id: string;
  platform: string;
  label: string;
  url: string;
  textProtocol: string;
  imageProtocol: string;
  videoProtocol: string;
  appendV1: boolean;
  vendor?: string;
  hint?: string;
  sponsored: boolean;
  quickSetup: boolean;
  credentialUrl?: string;
  documentationUrl?: string;
  supportedModelTypes: number[];
  discovery: string;
  modelPresetCodes: string[];
}

export interface ProviderModelCandidate {
  id: string;
  name: string;
  modelType: number;
  modelProtocol: string;
  capabilityPresetCode: string | null;
  config: string;
  configurationHint: string | null;
}

export interface ProviderDiscoveryResult {
  providerId: string;
  keyVerified: boolean;
  notice: string;
  models: ProviderModelCandidate[];
}

export const providerCatalogApi = {
  list: () => http.get<never, ProviderDefinition[]>("/api/ai/providers"),
  preview: (configuration: ApiConfigSaveReq) =>
    http.post<never, ProviderDiscoveryResult>(
      "/api/ai/providers/preview",
      configuration,
    ),
  setup: (
    configuration: ApiConfigSaveReq,
    modelIds: string[],
    requestId: string,
  ) =>
    http.post<
      never,
      { apiConfigId: number; createdModels: number; existingModels: number }
    >("/api/ai/providers/setup", { configuration, modelIds, requestId }),
};
