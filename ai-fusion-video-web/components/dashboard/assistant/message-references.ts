import type {
  AiMultimodalInputTransport,
  AiMultimodalInputType,
  AssistantMcpToolReferenceOption,
  AssistantSkillReferenceOption,
} from "@/lib/api/ai-assistant";
import type { AssistantMessageProjectReference } from "@/lib/store/assistant-types";

export interface AssistantMessageAttachmentReference {
  id: string;
  name: string;
  inputType: AiMultimodalInputType;
  mimeType: string;
  transport: AiMultimodalInputTransport;
  resourceUrl?: string;
  url?: string;
  size: number;
}

export interface ParsedAssistantMessageReferences {
  version: number | null;
  projectId: number | null;
  project: AssistantMessageProjectReference | null;
  skills: AssistantSkillReferenceOption[];
  mcpTools: AssistantMcpToolReferenceOption[];
  attachments: AssistantMessageAttachmentReference[];
}

const INPUT_TYPES = new Set<AiMultimodalInputType>(["image", "video", "audio", "file"]);
const TRANSPORTS = new Set<AiMultimodalInputTransport>(["url", "base64"]);

export function parseAssistantMessageReferences(
  referencesJson?: string,
): ParsedAssistantMessageReferences {
  if (!referencesJson) return emptyReferences();

  try {
    const value: unknown = JSON.parse(referencesJson);
    if (!isRecord(value)) return emptyReferences();

    const version = Number.isInteger(value.version) ? value.version as number : null;
    if (version !== null && version !== 1 && version !== 2) {
      return emptyReferences();
    }
    const project = parseProject(value.project);
    const explicitProjectId = isPositiveInteger(value.projectId) ? value.projectId : null;
    return {
      version,
      projectId: explicitProjectId ?? project?.id ?? null,
      project,
      skills: Array.isArray(value.skills)
        ? value.skills.flatMap((item) => {
            const skill = parseSkillReference(item, version);
            return skill ? [skill] : [];
          })
        : [],
      mcpTools: Array.isArray(value.mcpTools) ? value.mcpTools.filter(isMcpToolReference) : [],
      attachments: Array.isArray(value.attachments)
        ? value.attachments.flatMap((item) => {
            const attachment = parseAttachment(item);
            return attachment ? [attachment] : [];
          })
        : [],
    };
  } catch {
    return emptyReferences();
  }
}

function emptyReferences(): ParsedAssistantMessageReferences {
  return {
    version: null,
    projectId: null,
    project: null,
    skills: [],
    mcpTools: [],
    attachments: [],
  };
}

function parseProject(value: unknown): AssistantMessageProjectReference | null {
  if (!isRecord(value) || !isPositiveInteger(value.id) || !isNonEmptyString(value.name)) {
    return null;
  }
  return {
    id: value.id,
    name: value.name,
    ...(typeof value.description === "string" ? { description: value.description } : {}),
  };
}

function parseSkillReference(
  value: unknown,
  version: number | null,
): AssistantSkillReferenceOption | null {
  if (!isRecord(value)
    || !isNonEmptyString(value.id)
    || !isNonEmptyString(value.name)
    || typeof value.description !== "string"
    || !isNonEmptyString(value.source)) {
    return null;
  }
  const displayName = version === 2
    ? (isNonEmptyString(value.displayName) ? value.displayName : null)
    : value.name;
  if (displayName === null) return null;
  return {
    id: value.id,
    name: value.name,
    displayName,
    description: value.description,
    source: value.source,
  };
}

function isMcpToolReference(value: unknown): value is AssistantMcpToolReferenceOption {
  return isRecord(value)
    && isNonEmptyString(value.serverName)
    && isNonEmptyString(value.toolName)
    && typeof value.description === "string"
    && typeof value.readOnly === "boolean";
}

function parseAttachment(value: unknown): AssistantMessageAttachmentReference | null {
  if (!isRecord(value)
    || !isNonEmptyString(value.id)
    || !isNonEmptyString(value.name)
    || !isInputType(value.inputType)
    || !isNonEmptyString(value.mimeType)
    || !isTransport(value.transport)
    || !isNonNegativeNumber(value.size)) {
    return null;
  }

  return {
    id: value.id,
    name: value.name,
    inputType: value.inputType,
    mimeType: value.mimeType,
    transport: value.transport,
    size: value.size,
    ...(isNonEmptyString(value.resourceUrl) ? { resourceUrl: value.resourceUrl } : {}),
    ...(isNonEmptyString(value.url) ? { url: value.url } : {}),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function isNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isInputType(value: unknown): value is AiMultimodalInputType {
  return typeof value === "string" && INPUT_TYPES.has(value as AiMultimodalInputType);
}

function isTransport(value: unknown): value is AiMultimodalInputTransport {
  return typeof value === "string" && TRANSPORTS.has(value as AiMultimodalInputTransport);
}
