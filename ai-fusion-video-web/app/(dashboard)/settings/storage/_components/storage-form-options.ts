import { getStorageProviderOption, renderEndpointTemplate, type StorageConfig as StorageConfigType } from "@/lib/api/storage";

export function getProviderOption(provider?: string) {
  return getStorageProviderOption(provider || "generic_s3");
}

export function getProviderDefaultOptions(provider?: string) {
  const option = getProviderOption(provider);
  return {
    pathStyleAccessEnabled: option.pathStyleAccessEnabled,
    chunkedEncodingEnabled: option.chunkedEncodingEnabled,
    checksumCalculation: "WHEN_REQUIRED" as const,
  };
}

export function buildEndpoint(provider?: string, region?: string) {
  return renderEndpointTemplate(provider || "generic_s3", region);
}

export function hasEndpointTemplate(provider?: string) {
  return !!getProviderOption(provider).endpointTemplate;
}

export function isTemplateEndpoint(provider?: string, region?: string, endpoint?: string) {
  const generated = buildEndpoint(provider, region);
  return !!generated && generated === endpoint;
}

export function endpointPlaceholder(provider?: string) {
  const option = getProviderOption(provider);
  return option.endpointTemplate || "https://s3.example.com";
}

export function normalizeStorageType(config: Pick<StorageConfigType, "type" | "provider" | "endpoint" | "bucketName">) {
  if (config.type === "local") return "local";
  if (config.type === "s3" || config.type === "aliyun_oss" || config.type === "tencent_cos") return "s3";
  if (config.provider || config.endpoint || config.bucketName) return "s3";
  return "local";
}

export function normalizeStorageProvider(config: Pick<StorageConfigType, "type" | "provider">) {
  if (config.provider) return config.provider;
  if (config.type === "aliyun_oss") return "aliyun_oss";
  if (config.type === "tencent_cos") return "tencent_cos";
  return "generic_s3";
}
