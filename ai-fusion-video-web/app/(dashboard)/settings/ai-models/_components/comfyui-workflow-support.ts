import type {
  ComfyUiInputBinding,
  ComfyUiInputBindings,
  ComfyUiOutputBinding,
  ComfyUiWorkflowModelType,
} from "@/lib/api/comfyui-workflow";

export interface ComfyUiNodeOption {
  id: string;
  classType: string;
  title?: string;
  inputNames: string[];
}

export interface FlatComfyUiInputBinding extends ComfyUiInputBinding {
  businessField: string;
}

export const COMFYUI_COMMON_INPUTS = ["prompt", "negativePrompt", "seed", "count"] as const;
export const COMFYUI_IMAGE_INPUTS = ["width", "height", "referenceImages"] as const;
export const COMFYUI_VIDEO_INPUTS = [
  "width",
  "height",
  "duration",
  "fps",
  "firstFrame",
  "lastFrame",
  "referenceImages",
  "referenceVideos",
  "referenceAudios",
  "generateAudio",
] as const;

export const COMFYUI_VALUE_TYPE_OPTIONS = [
  { value: "string", label: "文本" },
  { value: "integer", label: "整数" },
  { value: "number", label: "数值" },
  { value: "boolean", label: "布尔" },
  { value: "string_list", label: "文本列表" },
  { value: "uploaded_image", label: "上传图片" },
  { value: "uploaded_video", label: "上传视频" },
  { value: "uploaded_audio", label: "上传音频" },
] as const;

export const COMFYUI_MEDIA_TYPE_OPTIONS = [
  { value: "image", label: "图片" },
  { value: "video", label: "视频" },
  { value: "audio", label: "音频" },
  { value: "file", label: "文件" },
] as const;

export const COMFYUI_OUTPUT_ROLE_OPTIONS = [
  { value: "primary", label: "主结果" },
  { value: "cover", label: "视频封面" },
  { value: "auxiliary", label: "附加结果" },
] as const;

export function getComfyUiBusinessFields(modelType: ComfyUiWorkflowModelType): string[] {
  return [
    ...COMFYUI_COMMON_INPUTS,
    ...(modelType === 2 ? COMFYUI_IMAGE_INPUTS : COMFYUI_VIDEO_INPUTS),
  ];
}

export function parseComfyUiNodes(apiWorkflowJson: string): ComfyUiNodeOption[] {
  if (!apiWorkflowJson.trim()) return [];
  const parsed = JSON.parse(apiWorkflowJson) as Record<string, unknown>;
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("API-format 工作流必须是 JSON 对象");
  }
  if ("nodes" in parsed || "links" in parsed) {
    throw new Error("检测到 UI-format，请在 ComfyUI 中使用 Export (API) 导出");
  }
  return Object.entries(parsed).map(([id, rawNode]) => {
    const node = rawNode as { class_type?: unknown; inputs?: unknown; _meta?: unknown };
    if (!node || typeof node !== "object" || typeof node.class_type !== "string") {
      throw new Error(`节点 ${id} 缺少 class_type`);
    }
    if (!node.inputs || Array.isArray(node.inputs) || typeof node.inputs !== "object") {
      throw new Error(`节点 ${id} 缺少 inputs 对象`);
    }
    const metadata = node._meta && !Array.isArray(node._meta) && typeof node._meta === "object"
      ? node._meta as { title?: unknown }
      : undefined;
    const title = typeof metadata?.title === "string" ? metadata.title.trim() : "";
    return {
      id,
      classType: node.class_type,
      title: title || undefined,
      inputNames: Object.keys(node.inputs as Record<string, unknown>),
    };
  });
}

export function formatComfyUiNodeLabel(node: ComfyUiNodeOption): string {
  return node.title
    ? `${node.id} · ${node.title}（${node.classType}）`
    : `${node.id} · ${node.classType}`;
}

export function flattenInputBindings(bindings: ComfyUiInputBindings): FlatComfyUiInputBinding[] {
  return Object.entries(bindings).flatMap(([businessField, items]) =>
    items.map(item => ({ ...item, businessField })),
  );
}

export function groupInputBindings(rows: FlatComfyUiInputBinding[]): ComfyUiInputBindings {
  return rows.reduce<ComfyUiInputBindings>((result, row) => {
    const { businessField, ...binding } = row;
    result[businessField] = [...(result[businessField] || []), binding];
    return result;
  }, {});
}

export function parseInputBindingsJson(value: string): ComfyUiInputBindings {
  if (!value.trim()) return {};
  const parsed = JSON.parse(value) as ComfyUiInputBindings;
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error("输入绑定必须是 JSON 对象");
  }
  return parsed;
}

export function parseOutputBindingsJson(value: string): ComfyUiOutputBinding[] {
  if (!value.trim()) return [];
  const parsed = JSON.parse(value) as ComfyUiOutputBinding[];
  if (!Array.isArray(parsed)) throw new Error("输出绑定必须是 JSON 数组");
  return parsed;
}
