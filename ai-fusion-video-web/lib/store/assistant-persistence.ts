import type { AssistantStoreState, PersistedAssistantState } from "./assistant-types";
import {
  clampDockWidth,
  clampLauncherPosition,
  clampRect,
  getDefaultLauncherPosition,
  getDefaultNormalRect,
} from "@/components/dashboard/assistant/geometry";
import type { AssistantMode } from "./assistant-types";
import type { ToolExecutionMode } from "@/lib/api/ai-assistant";

export const ASSISTANT_STORAGE_SCHEMA_VERSION = 1;
const ASSISTANT_GEOMETRY_VERSION = 2;

let persistTimer: ReturnType<typeof setTimeout> | null = null;

function storageKey(userId: number) {
  return `fusion-assistant:${userId}:v${ASSISTANT_STORAGE_SCHEMA_VERSION}`;
}

function isMode(value: unknown): value is AssistantMode {
  return value === "collapsed" || value === "floating" || value === "docked" || value === "maximized";
}

function isOpenMode(value: unknown): value is Exclude<AssistantMode, "collapsed"> {
  return value === "floating" || value === "docked" || value === "maximized";
}

function isRestoreMode(value: unknown): value is Exclude<AssistantMode, "collapsed" | "maximized"> {
  return value === "floating" || value === "docked";
}

function finiteNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function safeRecord(value: unknown): Record<string, string> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const result: Record<string, string> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (typeof entry === "string" && entry.trim()) result[key] = entry;
  }
  return result;
}

function safeNumberRecord(value: unknown): Record<string, number> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const result: Record<string, number> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (typeof entry === "number" && Number.isFinite(entry) && entry >= 0) {
      result[key] = Math.floor(entry);
    }
  }
  return result;
}

function isToolExecutionMode(value: unknown): value is ToolExecutionMode {
  return value === "DEFAULT"
    || value === "ALWAYS_ASK"
    || value === "ALWAYS_ALLOW"
    || value === "FULL_ACCESS";
}

function safeToolExecutionModes(value: unknown): Record<string, ToolExecutionMode> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const result: Record<string, ToolExecutionMode> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (isToolExecutionMode(entry)) result[key] = entry;
  }
  return result;
}

function readRaw(userId: number): Partial<PersistedAssistantState> {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(storageKey(userId));
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return {};
    const value = parsed as Partial<PersistedAssistantState>;
    return value.schemaVersion === ASSISTANT_STORAGE_SCHEMA_VERSION ? value : {};
  } catch {
    return {};
  }
}

export function defaultPersistedState(userId: number): PersistedAssistantState {
  const value = readRaw(userId);
  return {
    schemaVersion: ASSISTANT_STORAGE_SCHEMA_VERSION,
    geometryVersion: ASSISTANT_GEOMETRY_VERSION,
    mode: isMode(value.mode) ? value.mode : "collapsed",
    lastOpenMode: isOpenMode(value.lastOpenMode) && value.lastOpenMode !== "maximized"
      ? value.lastOpenMode
      : "floating",
    restoreMode: isRestoreMode(value.restoreMode) ? value.restoreMode : "floating",
    launcherPosition: value.launcherPosition && typeof value.launcherPosition === "object"
      ? clampLauncherPosition(value.launcherPosition)
      : getDefaultLauncherPosition(),
    normalRect: value.geometryVersion === ASSISTANT_GEOMETRY_VERSION
      && value.normalRect
      && typeof value.normalRect === "object"
      ? clampRect(value.normalRect)
      : getDefaultNormalRect(),
    dockWidth: Math.max(0, finiteNumber(value.dockWidth, 520)),
    selectedConversationId: typeof value.selectedConversationId === "string"
      ? value.selectedConversationId
      : null,
    selectedModelId: typeof value.selectedModelId === "number" && Number.isSafeInteger(value.selectedModelId)
      ? value.selectedModelId
      : null,
    drafts: safeRecord(value.drafts),
    runIds: safeRecord(value.runIds),
    lastSequences: safeNumberRecord(value.lastSequences),
    toolExecutionModes: safeToolExecutionModes(value.toolExecutionModes),
    newToolExecutionMode: isToolExecutionMode(value.newToolExecutionMode)
      ? value.newToolExecutionMode
      : "DEFAULT",
  };
}

export function writeAssistantPersisted(state: AssistantStoreState) {
  if (typeof window === "undefined" || !state.hydratedUserId) return;

  // Unloaded pages and an interrupted first render must not erase saved conversations.
  const previous = defaultPersistedState(state.hydratedUserId);
  const drafts: Record<string, string> = { ...previous.drafts, __new__: state.newDraft };
  const runIds: Record<string, string> = { ...previous.runIds };
  const lastSequences: Record<string, number> = { ...previous.lastSequences };
  const toolExecutionModes: Record<string, ToolExecutionMode> = { ...previous.toolExecutionModes };

  for (const [conversationId, runtime] of Object.entries(state.conversationStates)) {
    toolExecutionModes[conversationId] = runtime.toolExecutionMode;
    delete drafts[conversationId];
    delete runIds[conversationId];
    delete lastSequences[conversationId];
    if (runtime.draft) drafts[conversationId] = runtime.draft;

    const advertisedRunId = runtime.knownRunId || runtime.pipeline.runId;
    if (advertisedRunId) runIds[conversationId] = advertisedRunId;

    const cursorRunId = runtime.pipeline.runId;
    const cursor = Math.max(0, Math.floor(runtime.pipeline.lastSequence));
    // Never persist a cursor without the run identity it belongs to. If the
    // status endpoint has already announced a different run, the coordinator
    // must replay that run from zero after the next confirmation.
    if (cursor > 0 && cursorRunId && cursorRunId === advertisedRunId) {
      lastSequences[conversationId] = cursor;
    }
  }

  const value: PersistedAssistantState = {
    schemaVersion: ASSISTANT_STORAGE_SCHEMA_VERSION,
    geometryVersion: ASSISTANT_GEOMETRY_VERSION,
    mode: state.mode,
    lastOpenMode: state.lastOpenMode,
    restoreMode: state.restoreMode,
    launcherPosition: state.launcherPosition,
    normalRect: state.normalRect,
    dockWidth: state.dockWidth,
    selectedConversationId: state.restorationPending ? previous.selectedConversationId : state.selectedConversationId,
    selectedModelId: state.selectedModelId,
    drafts,
    runIds,
    lastSequences,
    toolExecutionModes,
    newToolExecutionMode: state.newToolExecutionMode,
  };

  savePersisted(state.hydratedUserId, value);
}

export function forgetAssistantConversation(userId: number, conversationId: string) {
  const value = defaultPersistedState(userId);
  delete value.drafts[conversationId];
  delete value.runIds[conversationId];
  delete value.lastSequences[conversationId];
  delete value.toolExecutionModes[conversationId];
  if (value.selectedConversationId === conversationId) value.selectedConversationId = null;
  savePersisted(userId, value);
}

function savePersisted(userId: number, value: PersistedAssistantState) {
  try {
    localStorage.setItem(storageKey(userId), JSON.stringify(value));
  } catch {
    // A full/private storage must never break the assistant UI.
  }
}

export function scheduleAssistantPersist(get: () => AssistantStoreState) {
  if (persistTimer) return;
  persistTimer = setTimeout(() => {
    persistTimer = null;
    writeAssistantPersisted(get());
  }, 250);
}

export function clearAssistantPersistTimer() {
  if (persistTimer) clearTimeout(persistTimer);
  persistTimer = null;
}

export function commitAssistantPersist(get: () => AssistantStoreState) {
  writeAssistantPersisted(get());
}

export function clampPersistedDockWidth(width: number, availableWidth: number) {
  return clampDockWidth(width, availableWidth);
}
