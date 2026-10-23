import { getAssistantReferenceOptions, type AssistantAgentType, type AssistantReferenceOptions } from "@/lib/api/ai-assistant";
import { aiModelApi, type AiModel } from "@/lib/api/ai-model";
import { projectApi, type Project } from "@/lib/api/project";

const TTL_MS = 30_000;
interface Entry<T> { value?: T; loadedAt: number; request?: Promise<T> }
let account: number | null = null;
const entries = new Map<string, Entry<unknown>>();

export function invalidateAssistantReferences(userId: number) {
  if (account !== userId) return;
  for (const key of entries.keys()) if (key.startsWith("references:")) entries.delete(key);
}

function load<T>(userId: number, key: string, fetchValue: () => Promise<T>): Promise<T> {
  if (!Number.isSafeInteger(userId) || userId <= 0) throw new Error("助手目录缺少登录身份");
  if (account !== userId) {
    account = userId;
    entries.clear();
  }
  let entry = entries.get(key) as Entry<T> | undefined;
  if (!entry) { entry = { loadedAt: 0 }; entries.set(key, entry); }
  if (entry.value !== undefined && Date.now() - entry.loadedAt < TTL_MS) return Promise.resolve(entry.value);
  if (entry.request) return entry.request;
  const current = entry;
  current.request = fetchValue().then((value) => {
    if (account === userId && entries.get(key) === current) {
      current.value = value;
      current.loadedAt = Date.now();
    }
    return value;
  }).finally(() => { current.request = undefined; });
  return current.request;
}

export function hasCachedAssistantModels(userId: number) {
  const entry = account === userId ? entries.get("models") : undefined;
  return entry?.value !== undefined && Date.now() - entry.loadedAt < TTL_MS;
}

export function loadAssistantModels(userId: number): Promise<AiModel[]> {
  return load(userId, "models", () => aiModelApi.listByType(1));
}

export function loadAssistantProjects(userId: number): Promise<Project[]> {
  return load(userId, "projects", () => projectApi.list());
}

export function loadAssistantReferenceOptions(userId: number, agentType: AssistantAgentType): Promise<AssistantReferenceOptions> {
  return load(userId, `references:${agentType}`, () => getAssistantReferenceOptions(agentType));
}
