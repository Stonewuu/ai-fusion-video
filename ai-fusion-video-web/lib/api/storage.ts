import { http } from "./client";

// ============================================================
// 类型定义
// ============================================================

export interface StorageConfig {
  id: number;
  name: string;
  type: string;
  provider?: string;
  endpoint?: string;
  bucketName?: string;
  accessKey?: string;
  secretKey?: string;
  region?: string;
  basePath?: string;
  customDomain?: string;
  options?: StorageConfigOptions;
  isDefault?: boolean;
  status: number;
  remark?: string;
  createTime?: string;
  updateTime?: string;
}

export interface StorageConfigSaveReq {
  id?: number;
  name: string;
  type?: string;
  provider?: string;
  endpoint?: string;
  bucketName?: string;
  accessKey?: string;
  secretKey?: string;
  region?: string;
  basePath?: string;
  customDomain?: string;
  options?: StorageConfigOptions;
  isDefault?: boolean;
  status?: number;
  remark?: string;
}

export interface StorageConfigOptions {
  pathStyleAccessEnabled?: boolean;
  chunkedEncodingEnabled?: boolean;
  signingRegion?: string;
  checksumCalculation?: "WHEN_REQUIRED" | "WHEN_SUPPORTED";
}

export interface StorageConfigTestResult {
  success: boolean;
  message: string;
  publicUrl?: string;
}

// ============================================================
// 存储类型选项
// ============================================================

export const STORAGE_TYPE_OPTIONS = [
  { value: "local", label: "本地存储", description: "文件保存在服务器本地磁盘" },
  { value: "s3", label: "对象存储", description: "S3 兼容：阿里云 OSS / 腾讯 COS / 七牛 / 天翼云等" },
] as const;

export const STORAGE_TYPE_LABELS: Record<string, string> = {
  local: "本地存储",
  s3: "S3 兼容",
};

export interface StorageProviderOption {
  value: string;
  label: string;
  description: string;
  endpointTemplate?: string;
  defaultRegion?: string;
  pathStyleAccessEnabled: boolean;
  chunkedEncodingEnabled: boolean;
  endpointRequired: boolean;
}

export const STORAGE_PROVIDER_OPTIONS: StorageProviderOption[] = [
  {
    value: "generic_s3",
    label: "通用 S3",
    description: "适用于支持 AWS S3 协议的对象存储",
    defaultRegion: "us-east-1",
    pathStyleAccessEnabled: true,
    chunkedEncodingEnabled: false,
    endpointRequired: true,
  },
  {
    value: "aliyun_oss",
    label: "阿里云 OSS",
    description: "AWS S3 兼容访问端点 (Endpoint)",
    endpointTemplate: "https://s3.oss-{region}.aliyuncs.com",
    pathStyleAccessEnabled: false,
    chunkedEncodingEnabled: false,
    endpointRequired: false,
  },
  {
    value: "tencent_cos",
    label: "腾讯云 COS",
    description: "COS S3 兼容访问端点 (Endpoint)",
    endpointTemplate: "https://cos.{region}.myqcloud.com",
    pathStyleAccessEnabled: false,
    chunkedEncodingEnabled: false,
    endpointRequired: false,
  },
  {
    value: "qiniu_kodo",
    label: "七牛云 Kodo",
    description: "Kodo S3 兼容访问端点 (Endpoint)",
    endpointTemplate: "https://s3.{region}.qiniucs.com",
    pathStyleAccessEnabled: true,
    chunkedEncodingEnabled: false,
    endpointRequired: false,
  },
  {
    value: "ctyun_zos",
    label: "天翼云 ZOS",
    description: "天翼云对象存储 S3 协议",
    defaultRegion: "us-east-1",
    pathStyleAccessEnabled: true,
    chunkedEncodingEnabled: false,
    endpointRequired: true,
  },
  {
    value: "minio",
    label: "MinIO",
    description: "自托管 S3 兼容服务",
    defaultRegion: "us-east-1",
    pathStyleAccessEnabled: true,
    chunkedEncodingEnabled: false,
    endpointRequired: true,
  },
  {
    value: "aws_s3",
    label: "AWS S3",
    description: "AWS 官方 S3 服务",
    endpointTemplate: "https://s3.{region}.amazonaws.com",
    defaultRegion: "us-east-1",
    pathStyleAccessEnabled: false,
    chunkedEncodingEnabled: true,
    endpointRequired: false,
  },
];

export const STORAGE_PROVIDER_LABELS = Object.fromEntries(
  STORAGE_PROVIDER_OPTIONS.map((provider) => [provider.value, provider.label])
) as Record<string, string>;

export function getStorageProviderOption(provider?: string) {
  return STORAGE_PROVIDER_OPTIONS.find((item) => item.value === provider)
    ?? STORAGE_PROVIDER_OPTIONS[0];
}

export function renderEndpointTemplate(provider?: string, region?: string) {
  const option = getStorageProviderOption(provider);
  if (!option.endpointTemplate || !region?.trim()) return "";
  return option.endpointTemplate.replace("{region}", region.trim());
}

// ============================================================
// API
// ============================================================

export const storageConfigApi = {
  async create(data: StorageConfigSaveReq): Promise<number> {
    return http.post("/api/storage/config/create", data);
  },

  async update(data: StorageConfigSaveReq): Promise<boolean> {
    return http.put("/api/storage/config/update", data);
  },

  async delete(id: number): Promise<boolean> {
    return http.delete("/api/storage/config/delete", { params: { id } });
  },

  async get(id: number): Promise<StorageConfig> {
    return http.get("/api/storage/config/get", { params: { id } });
  },

  async list(): Promise<StorageConfig[]> {
    return http.get("/api/storage/config/list");
  },

  async setDefault(id: number): Promise<boolean> {
    return http.put("/api/storage/config/set-default", null, { params: { id } });
  },

  async test(data: StorageConfigSaveReq): Promise<StorageConfigTestResult> {
    return http.post("/api/storage/config/test", data);
  },
};

export async function uploadFile(
  file: File,
  subDir: string = "uploads"
): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("subDir", subDir);

  return http.post("/api/storage/upload", formData, { timeout: 0 });
}

export async function uploadAssistantInput(
  file: File,
  modelId: number,
  transport: "url" | "base64",
): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("modelId", String(modelId));
  formData.append("transport", transport);

  return http.post("/api/storage/assistant-upload", formData, { timeout: 0 });
}
