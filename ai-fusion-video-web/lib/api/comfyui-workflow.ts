import { http } from "./client";
import type { PageResult } from "./types";

export type ComfyUiWorkflowModelType = 2 | 3;
export type ComfyUiBindingValueType =
  | "string"
  | "integer"
  | "number"
  | "boolean"
  | "string_list"
  | "uploaded_image"
  | "uploaded_video"
  | "uploaded_audio";
export type ComfyUiOutputMediaType = "image" | "video" | "audio" | "file";
export type ComfyUiOutputRole = "primary" | "cover" | "auxiliary";

export interface ComfyUiInputBinding {
  nodeId: string;
  inputName: string;
  valueType: ComfyUiBindingValueType;
  index?: number;
}

export type ComfyUiInputBindings = Record<string, ComfyUiInputBinding[]>;

export interface ComfyUiOutputBinding {
  nodeId: string;
  mediaType: ComfyUiOutputMediaType;
  role: ComfyUiOutputRole;
}

export interface ComfyUiWorkflow {
  id: number;
  apiConfigId: number;
  name: string;
  code: string;
  modelType: ComfyUiWorkflowModelType;
  description: string | null;
  activeVersionId: number | null;
  status: number;
  createTime: string;
  updateTime: string;
}

export interface ComfyUiWorkflowVersion {
  id: number;
  workflowId: number;
  versionNo: number;
  uiWorkflowJson: string | null;
  apiWorkflowJson: string;
  inputBindingsJson: string;
  outputBindingsJson: string;
  requiredNodesJson: string;
  workflowHash: string;
  validationStatus: number;
  validationMessage: string | null;
  testStatus: number;
  testMessage: string | null;
  lastTestTime: string | null;
  published: boolean;
  createTime: string;
  updateTime: string;
}

export interface ComfyUiWorkflowSaveReq {
  id?: number;
  apiConfigId: number;
  name: string;
  code: string;
  modelType: ComfyUiWorkflowModelType;
  description?: string;
  status?: number;
}

export interface ComfyUiWorkflowVersionSaveReq {
  id?: number;
  workflowId: number;
  uiWorkflowJson?: string;
  apiWorkflowJson: string;
  inputBindingsJson: string;
  outputBindingsJson: string;
}

export interface ComfyUiWorkflowValidationResult {
  valid: boolean;
  checkedNodeCount: number;
  missingNodeClasses: string[];
  invalidModelInputs: string[];
  message: string;
}

export interface ComfyUiStoredOutput {
  mediaType: string;
  role: string;
  url: string;
  size: number;
}

export interface ComfyUiWorkflowTestResult {
  passed: boolean;
  promptId: string | null;
  durationMillis: number;
  outputs: ComfyUiStoredOutput[];
  message: string;
}

export interface ComfyUiConnectionResult {
  connected: boolean;
  jobsApiSupported: boolean;
  version: string | null;
  systemStats: unknown;
  features: unknown;
}

export const comfyUiWorkflowApi = {
  page: (pageNo = 1, pageSize = 100) =>
    http.get<never, PageResult<ComfyUiWorkflow>>(
      `/api/ai/comfyui/workflow/page?pageNo=${pageNo}&pageSize=${pageSize}`,
    ),
  list: (apiConfigId?: number, modelType?: number) => {
    const query = new URLSearchParams();
    if (apiConfigId !== undefined) query.set("apiConfigId", String(apiConfigId));
    if (modelType !== undefined) query.set("modelType", String(modelType));
    const suffix = query.size > 0 ? `?${query.toString()}` : "";
    return http.get<never, ComfyUiWorkflow[]>(`/api/ai/comfyui/workflow/list${suffix}`);
  },
  get: (id: number) =>
    http.get<never, ComfyUiWorkflow>(`/api/ai/comfyui/workflow/get?id=${id}`),
  create: (data: ComfyUiWorkflowSaveReq) =>
    http.post<never, number>("/api/ai/comfyui/workflow/create", data),
  update: (data: ComfyUiWorkflowSaveReq) =>
    http.put<never, boolean>("/api/ai/comfyui/workflow/update", data),
  delete: (id: number) =>
    http.delete<never, boolean>(`/api/ai/comfyui/workflow/delete?id=${id}`),
  versions: (workflowId: number) =>
    http.get<never, ComfyUiWorkflowVersion[]>(
      `/api/ai/comfyui/workflow/versions?workflowId=${workflowId}`,
    ),
  getVersion: (id: number) =>
    http.get<never, ComfyUiWorkflowVersion>(`/api/ai/comfyui/workflow/version/get?id=${id}`),
  createVersion: (data: ComfyUiWorkflowVersionSaveReq) =>
    http.post<never, number>("/api/ai/comfyui/workflow/version/create", data),
  updateVersion: (data: ComfyUiWorkflowVersionSaveReq) =>
    http.put<never, boolean>("/api/ai/comfyui/workflow/version/update", data),
  deleteVersion: (id: number) =>
    http.delete<never, boolean>(`/api/ai/comfyui/workflow/version/delete?id=${id}`),
  validateVersion: (versionId: number) =>
    http.post<never, ComfyUiWorkflowValidationResult>(
      `/api/ai/comfyui/workflow/version/validate?versionId=${versionId}`,
    ),
  testVersion: (versionId: number, inputs: Record<string, unknown>) =>
    http.post<never, ComfyUiWorkflowTestResult>("/api/ai/comfyui/workflow/version/test", {
      versionId,
      inputs,
    }),
  publish: (workflowId: number, versionId: number) =>
    http.post<never, boolean>(
      `/api/ai/comfyui/workflow/publish?workflowId=${workflowId}&versionId=${versionId}`,
    ),
  testConnection: (apiConfigId: number) =>
    http.post<never, ComfyUiConnectionResult>(
      `/api/ai/api-config/test-comfyui-connectivity?id=${apiConfigId}`,
    ),
};
