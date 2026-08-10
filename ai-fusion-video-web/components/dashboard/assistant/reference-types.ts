import type {
  AssistantMcpToolReferenceOption,
  AssistantSkillReferenceOption,
} from "@/lib/api/ai-assistant";

export interface AssistantProjectReference {
  id: number;
  name: string;
  description?: string;
}

export type AssistantReferencePickerMode = "capability" | "project";

export interface AssistantReferenceTrigger {
  mode: AssistantReferencePickerMode;
  query: string;
  start: number;
  end: number;
}

export type AssistantCapabilityReference =
  | { kind: "skill"; value: AssistantSkillReferenceOption }
  | { kind: "mcp"; value: AssistantMcpToolReferenceOption };

export type AssistantReferencePickerItem =
  | AssistantCapabilityReference
  | { kind: "project"; value: AssistantProjectReference };
