import { http } from "./client";
import type { CanvasGraph } from "@/lib/canvas/types";
import type { ResourceType } from "./resource-events";
import type { ManagedMedia } from "./media";

export type TemplateProviderKind = "TEXT_MODEL" | "IMAGE_MODEL" | "VIDEO_MODEL" | "COMFYUI_MODEL" | "AUDIO_API" | "WORKFLOW" | "MCP";
export interface TemplateUse { nodeId: string; batch: boolean; field: string }
export interface TemplateProvider { id: string; kind: TemplateProviderKind; label: string; uses: TemplateUse[] }
export interface TemplateMedia { id: string; name: string; content: ManagedMedia["content"] & { contentHash: string } }
export interface CanvasTemplate {
  schema: "afv.canvas.template"; schemaVersion: 1; title: string; graph: CanvasGraph;
  objects: { type: ResourceType; id: number; data: Record<string, unknown> }[];
  primaryScript: number | null; primaryStoryboard: number | null;
  media: TemplateMedia[]; providers: TemplateProvider[]; fixedOutputs: Record<string, unknown>;
}
export interface TemplateBinding { id: string; version?: number | null }
export interface TemplateBindings { media: Record<string, TemplateBinding>; providers: Record<string, TemplateBinding> }
export interface TemplatePreview {
  hash: string; nodes: number; businessObjects: number; mediaFiles: number; notices: string[];
  providerNames: Record<string, string>; ownership: { type: 1 | 2; id: number; name: string };
}
export interface TemplateImportRequest {
  requestId: string; canvasId: string; title: string; templateJson: string; bindings: TemplateBindings; previewHash: string;
}
export const canvasTemplatesApi = {
  export: (canvasId: string, expectedRevision: number): Promise<CanvasTemplate> =>
    http.get(`/api/canvases/${canvasId}/template`, { params: { expectedRevision } }),
  inspect: (templateJson: string): Promise<CanvasTemplate> => http.post("/api/canvas-templates/inspect", { templateJson }),
  preview: (templateJson: string, bindings: TemplateBindings): Promise<TemplatePreview> =>
    http.post("/api/canvas-templates/preview", { templateJson, bindings }),
  import: (request: TemplateImportRequest): Promise<{ result: { id: number; canvasId: string; title: string } }> =>
    http.post("/api/canvas-templates/import", request),
};
