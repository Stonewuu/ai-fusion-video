import type {
  AgentConversation,
  AgentMessage,
  AssistantMcpToolReferenceOption,
  AssistantSkillReferenceOption,
  AiMultimodalInput,
} from "@/lib/api/ai-assistant";
import type { AiChatReq, AiChatStreamEvent } from "@/lib/api/ai-pipeline";
import type { AgentPipelineState } from "@/components/dashboard/agent-pipeline/types";
import type { AssistantPoint, AssistantRect } from "@/components/dashboard/assistant/geometry";

export const ASSISTANT_CATEGORY = "assistant";
export const NEW_ASSISTANT_DRAFT_KEY = "__new__";

export type AssistantMode = "collapsed" | "floating" | "docked" | "maximized";
export type AssistantConnectionMode = "start" | "reconnect";

export interface AssistantMessageProjectReference {
  id: number;
  name: string;
  description?: string;
}

export interface AssistantMessageReferences {
  project?: AssistantMessageProjectReference | null;
  skills: AssistantSkillReferenceOption[];
  mcpTools: AssistantMcpToolReferenceOption[];
  multimodalInputs?: AiMultimodalInput[];
}

export interface AssistantConnection {
  conversationId: string;
  runId?: string;
  controller: AbortController;
  connectionGeneration: number;
  connectionMode: AssistantConnectionMode;
}

export interface AssistantConversationRuntime {
  conversation: AgentConversation;
  messages: AgentMessage[];
  pipeline: AgentPipelineState;
  draft: string;
  status: string;
  /** True only after a server status response or a durable SSE event. */
  statusConfirmed: boolean;
  /** The run id currently advertised by the server for this conversation. */
  knownRunId?: string;
  /** Server metadata only; never used as the local replay cursor. */
  remoteLastSequence?: number;
  messagesLoaded: boolean;
  messagesLoading: boolean;
  messagesError?: string;
  connectionError?: string;
  unread: boolean;
}

export interface PersistedAssistantState {
  schemaVersion: number;
  geometryVersion: number;
  mode: AssistantMode;
  lastOpenMode: Exclude<AssistantMode, "collapsed">;
  restoreMode: Exclude<AssistantMode, "collapsed" | "maximized">;
  launcherPosition: AssistantPoint;
  normalRect: AssistantRect;
  dockWidth: number;
  selectedConversationId: string | null;
  selectedModelId: number | null;
  drafts: Record<string, string>;
  /** A cursor is valid only when its conversation entry has the same run id. */
  runIds: Record<string, string>;
  lastSequences: Record<string, number>;
}

export interface AssistantStoreState {
  hydratedUserId: number | null;
  initialized: boolean;
  mode: AssistantMode;
  lastOpenMode: Exclude<AssistantMode, "collapsed">;
  restoreMode: Exclude<AssistantMode, "collapsed" | "maximized">;
  launcherPosition: AssistantPoint;
  normalRect: AssistantRect;
  dockWidth: number;
  selectedConversationId: string | null;
  selectedModelId: number | null;
  conversations: AgentConversation[];
  conversationStates: Record<string, AssistantConversationRuntime>;
  newDraft: string;
  drawerOpen: boolean;
  conversationsLoading: boolean;
  conversationsError?: string;
  hasMoreConversations: boolean;
  conversationPage: number;
  connection: AssistantConnection | null;
  connectionGeneration: number;

  initializeForUser: (userId: number) => void;
  resetForUser: () => void;
  loadMoreConversations: () => void;
  selectConversation: (conversationId: string | null) => void;
  startNewConversation: () => void;
  setDraft: (conversationId: string | null, draft: string) => void;
  setSelectedModelId: (modelId: number | null) => void;
  sendMessage: (
    message: string,
    modelId: number | null,
    reasoningEffort: string | null,
    projectId?: number | null,
    references?: AssistantMessageReferences,
  ) => Promise<void>;
  stopGeneration: () => Promise<void>;
  markConversationRead: (conversationId: string) => void;
  deleteConversation: (conversationId: string, id: number) => Promise<void>;
  setMode: (mode: AssistantMode, canDock?: boolean) => void;
  openAssistant: (canDock?: boolean) => void;
  closeAssistant: () => void;
  setDrawerOpen: (open: boolean) => void;
  updateLauncherPosition: (position: AssistantPoint, commit?: boolean) => void;
  updateNormalRect: (rect: AssistantRect, commit?: boolean) => void;
  updateDockWidth: (width: number, availableWidth: number, commit?: boolean) => void;
  clampViewportGeometry: (availableWidth?: number) => void;
  loadMessagesIfNeeded: (conversationId: string) => Promise<void>;
  ensureContentConnection: () => void;
}

export type AssistantStreamEvent = AiChatStreamEvent;
export type AssistantChatRequest = AiChatReq;
