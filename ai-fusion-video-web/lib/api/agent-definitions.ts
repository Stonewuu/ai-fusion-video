import { http } from "./client";

export interface AgentSkillOption {
  name: string;
  displayName: string | null;
  description: string;
  source: string;
}

export interface AgentDefinitionValues {
  name: string;
  systemPrompt: string;
  instructionTemplate: string;
  defaultUserMessage: string;
  toolsEnabled: boolean;
  toolNames: string[] | null;
  subAgentToolNames: string[];
  mcpToolNames: string[] | null;
  defaultSkills: string[];
  maxIters: number;
}
export type AgentDefinitionOverrides = { [K in keyof AgentDefinitionValues]?: AgentDefinitionValues[K] | null };
export interface AgentDefinitionSummary {
  type: string;
  builtinName: string;
  name: string;
  customized: boolean;
  revision: number;
  toolsEnabled: boolean;
  defaultSkills: string[];
  maxIters: number;
}
export interface AgentToolOption {
  name: string;
  label: string;
  description: string;
  kind: "platform" | "agent" | "mcp";
  readOnly: boolean;
  category: string;
}
export interface AgentDefinitionDetail {
  configuration: {
    type: string;
    revision: number;
    overrides: AgentDefinitionOverrides;
    defaults: AgentDefinitionValues;
    effective: AgentDefinitionValues;
  };
  tools: AgentToolOption[];
  skills: AgentSkillOption[];
}

const path = "/api/ai/agent-config/definitions";
export const agentDefinitionsApi = {
  list: (): Promise<AgentDefinitionSummary[]> => http.get(path),
  get: (type: string): Promise<AgentDefinitionDetail> => http.get(`${path}/${encodeURIComponent(type)}`),
  save: (type: string, expectedRevision: number, overrides: AgentDefinitionOverrides): Promise<AgentDefinitionDetail> =>
    http.put(`${path}/${encodeURIComponent(type)}`, { expectedRevision, overrides }),
};

export function compactAgentOverrides(value: AgentDefinitionOverrides): AgentDefinitionOverrides {
  return Object.fromEntries(Object.entries(value).filter(([, field]) => field != null).sort(([left], [right]) => left.localeCompare(right)));
}

export function effectiveAgentValues(defaults: AgentDefinitionValues, overrides: AgentDefinitionOverrides): AgentDefinitionValues {
  return { ...defaults, ...compactAgentOverrides(overrides) } as AgentDefinitionValues;
}
