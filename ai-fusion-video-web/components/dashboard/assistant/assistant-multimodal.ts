import type { AiModel, MultimodalInputType } from "@/lib/api/ai-model";
import type {
  AiMultimodalInput,
  AiMultimodalInputTransport,
} from "@/lib/api/ai-assistant";
import { resolveMediaUrl } from "@/lib/api/client";
import { uploadAssistantInput } from "@/lib/api/storage";

const MAX_INPUT_COUNT = 8;
const MAX_BASE64_FILE_SIZE = 10 * 1024 * 1024;
const MAX_TOTAL_BASE64_SIZE = 20 * 1024 * 1024;
const MAX_URL_FILE_SIZE = 100 * 1024 * 1024;

const MIME_BY_EXTENSION: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  mp4: "video/mp4",
  webm: "video/webm",
  mov: "video/quicktime",
  mpeg: "video/mpeg",
  mp3: "audio/mpeg",
  m4a: "audio/mp4",
  wav: "audio/wav",
  ogg: "audio/ogg",
  flac: "audio/flac",
  aac: "audio/aac",
  pdf: "application/pdf",
  txt: "text/plain",
  md: "text/markdown",
  csv: "text/csv",
  json: "application/json",
};

const ALLOWED_MIME_TYPES = new Set(Object.values(MIME_BY_EXTENSION));

export interface AssistantAttachment extends AiMultimodalInput {
  previewUrl?: string;
}

export async function prepareAssistantAttachments(
  files: File[],
  model: AiModel,
  existing: AssistantAttachment[],
): Promise<AssistantAttachment[]> {
  if (existing.length + files.length > MAX_INPUT_COUNT) {
    throw new Error(`单次最多添加 ${MAX_INPUT_COUNT} 个附件`);
  }

  const existingKeys = new Set(existing.map((item) => `${item.name}:${item.size}`));
  const preparedFiles = files.filter((file) => {
    const key = `${file.name}:${file.size}`;
    if (existingKeys.has(key)) return false;
    existingKeys.add(key);
    return true;
  });
  let totalBase64Size = existing
    .filter((item) => item.transport === "base64")
    .reduce((sum, item) => sum + item.size, 0);

  const plans = preparedFiles.map((file) => {
    const mimeType = resolveMimeType(file);
    const inputType = inputTypeForMime(mimeType);
    const supportedTransports = model.multimodalInputTransports[inputType] ?? [];
    if (!model.multimodalInputTypes.includes(inputType) || !supportedTransports.length) {
      throw new Error(`${model.name} 不支持${inputTypeLabel(inputType)}输入`);
    }
    if (file.size <= 0 || file.size > MAX_URL_FILE_SIZE) {
      throw new Error(`${file.name} 的大小必须在 1B 到 100MB 之间`);
    }

    let transport: AiMultimodalInputTransport | null = null;
    if (
      supportedTransports.includes("base64")
      && file.size <= MAX_BASE64_FILE_SIZE
      && totalBase64Size + file.size <= MAX_TOTAL_BASE64_SIZE
    ) {
      transport = "base64";
      totalBase64Size += file.size;
    } else if (supportedTransports.includes("url")) {
      transport = "url";
    }
    if (!transport) {
      throw new Error(`${file.name} 超出 Base64 限制，且当前模型未启用 URL 输入`);
    }
    return { file, mimeType, inputType, transport };
  });

  const results = await Promise.allSettled(plans.map(async ({ file, mimeType, inputType, transport }) => {
    const id = createAttachmentId();
    if (transport === "base64") {
      const data = await readFileAsBase64(file);
      const resourceUrl = await uploadAssistantInput(file, model.id, transport);
      return {
        id,
        name: file.name,
        inputType,
        mimeType,
        transport,
        data,
        resourceUrl,
        size: file.size,
        previewUrl: inputType === "image" ? resolveMediaUrl(resourceUrl) ?? undefined : undefined,
      };
    }

    const url = await uploadAssistantInput(file, model.id, transport);
    return {
      id,
      name: file.name,
      inputType,
      mimeType,
      transport,
      url,
      resourceUrl: url,
      size: file.size,
      previewUrl: inputType === "image" ? resolveMediaUrl(url) ?? undefined : undefined,
    };
  }));

  const failure = results.find((result): result is PromiseRejectedResult => result.status === "rejected");
  if (failure) {
    throw failure.reason;
  }
  return results.map((result) => (result as PromiseFulfilledResult<AssistantAttachment>).value);
}

export function attachmentCompatibilityError(
  model: AiModel | null,
  attachments: AssistantAttachment[],
): string | null {
  if (!attachments.length) return null;
  if (!model) return "请先选择对话模型";
  for (const attachment of attachments) {
    const transports = model.multimodalInputTransports[attachment.inputType] ?? [];
    if (!model.multimodalInputTypes.includes(attachment.inputType)
      || !transports.includes(attachment.transport)) {
      return `${model.name} 不支持附件“${attachment.name}”的 ${attachment.transport.toUpperCase()} 输入`;
    }
  }
  return null;
}

export function multimodalCapabilitySummary(model: AiModel | null): string {
  if (!model || !model.multimodalInputTypes.length) return "当前模型仅支持文本输入";
  const capabilities = model.multimodalInputTypes.map((type) => {
    const transports = model.multimodalInputTransports[type] ?? [];
    const transportLabel = transports.map((transport) => transport === "url" ? "URL" : "Base64").join("/");
    return `${inputTypeLabel(type)}${transportLabel ? `（${transportLabel}）` : ""}`;
  });
  return `添加${capabilities.join("、")}`;
}

function resolveMimeType(file: File): string {
  const declared = file.type.toLowerCase().split(";", 1)[0].trim();
  const extension = file.name.toLowerCase().split(".").pop() ?? "";
  const inferred = MIME_BY_EXTENSION[extension];
  const mimeType = ALLOWED_MIME_TYPES.has(declared) ? declared : inferred;
  if (!mimeType || !ALLOWED_MIME_TYPES.has(mimeType)) {
    throw new Error(`${file.name} 不是支持的图片、视频、音频、PDF 或文本文件`);
  }
  return mimeType;
}

function inputTypeForMime(mimeType: string): MultimodalInputType {
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("video/")) return "video";
  if (mimeType.startsWith("audio/")) return "audio";
  return "file";
}

function inputTypeLabel(type: MultimodalInputType): string {
  return { image: "图片", video: "视频", audio: "音频", file: "文件" }[type];
}

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`读取 ${file.name} 失败`));
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const separator = result.indexOf(",");
      if (separator < 0) {
        reject(new Error(`${file.name} 无法转换为 Base64`));
        return;
      }
      resolve(result.slice(separator + 1));
    };
    reader.readAsDataURL(file);
  });
}

function createAttachmentId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `attachment-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
