import type { AgentConversation, AgentMessage } from "@/lib/api/ai-assistant";
import type { AiChatStreamEvent, PipelineRunStatus } from "@/lib/api/ai-pipeline";
import { messagesToTimeline } from "@/components/dashboard/notification-panel/history";
import {
  createInitialPipelineState,
  createPendingPipelineState,
  reducePipelineEvent,
} from "@/components/dashboard/agent-pipeline/state";
import type { AgentPipelineState } from "@/components/dashboard/agent-pipeline/types";
import type { TimelineItem } from "@/lib/store/pipeline-store";
import type { AssistantConversationRuntime } from "./assistant-types";

export function statusIsRunning(status: string | undefined) {
  return status === "running"
    || status === "pending"
    || status === "RUNNING"
    || status === "WAITING_CONFIRMATION"
    || status === "WAITING_EXTERNAL"
    || status === "CANCEL_REQUESTED";
}

export function statusFromPipeline(status: PipelineRunStatus | string): string {
  switch (status) {
    case "COMPLETED": return "completed";
    case "FAILED": return "failed";
    case "CANCELLED": return "cancelled";
    case "ERROR": return "failed";
    default: return "running";
  }
}

export function normalizeTitle(value: string) {
  const normalized = value.trim().replace(/\s+/g, " ");
  return normalized.slice(0, 50) || "新对话";
}

function conversationStatus(conversation: AgentConversation) {
  return conversation.status || "completed";
}

export function makeRuntime(
  conversation: AgentConversation,
  drafts: Record<string, string>,
  runIds: Record<string, string>,
  lastSequences: Record<string, number>,
  scrollPositions: Record<string, number>,
): AssistantConversationRuntime {
  const conversationId = conversation.conversationId;
  const knownRunId = runIds[conversationId];
  // A sequence without its run id is unsafe after a refresh. It is discarded
  // here and the connection coordinator will confirm the current run first.
  const lastSequence = knownRunId
    ? Math.max(0, Math.floor(Number.isFinite(lastSequences[conversationId]) ? lastSequences[conversationId] : 0))
    : 0;
  return {
    conversation,
    messages: [],
    pipeline: {
      ...createInitialPipelineState(),
      conversationId,
      runId: knownRunId,
      lastSequence,
    },
    draft: drafts[conversationId] ?? "",
    status: conversationStatus(conversation),
    statusConfirmed: !statusIsRunning(conversation.status),
    knownRunId,
    messagesLoaded: false,
    messagesLoading: false,
    unread: false,
    scrollTop: Math.max(0, Number.isFinite(scrollPositions[conversationId]) ? scrollPositions[conversationId] : 0),
  };
}

export function messageKey(message: AgentMessage) {
  if (message.role === "user" && message.messageOrder > 0) {
    return `user:${message.conversationId}:${message.messageOrder}:${message.content ?? ""}`;
  }
  if (message.id > 0) return `id:${message.id}`;
  if (message.runId && message.projectionKey) return `projection:${message.runId}:${message.projectionKey}`;
  return [message.role, message.messageOrder, message.content ?? "", message.toolCallId ?? ""].join(":");
}

export function mergeMessages(current: AgentMessage[], incoming: AgentMessage[]) {
  const merged = new Map<string, AgentMessage>();
  for (const message of [...current, ...incoming]) merged.set(messageKey(message), message);
  return [...merged.values()].sort((a, b) => (a.messageOrder ?? 0) - (b.messageOrder ?? 0));
}

export function timelineForMessages(messages: AgentMessage[]): TimelineItem[] {
  return messagesToTimeline(messages.filter((message) => message.role !== "user"));
}

export function hasTerminal(event: AiChatStreamEvent) {
  return !event.parentToolCallId
    && !event.agentName
    && (event.outputType === "DONE" || event.outputType === "ERROR" || event.outputType === "CANCELLED");
}

export function terminalStatusForEvent(event: AiChatStreamEvent) {
  if (event.outputType === "DONE") return "completed";
  if (event.outputType === "ERROR") return "failed";
  return "cancelled";
}

export function reduceAssistantEvent(
  pipeline: AgentPipelineState,
  event: AiChatStreamEvent,
) {
  return reducePipelineEvent(pipeline, event);
}

export function pendingPipelineForNextRun(
  conversationId: string,
): AgentPipelineState {
  return {
    ...createPendingPipelineState(),
    conversationId,
    lastSequence: 0,
  };
}

export function uniqueConversations(
  current: AgentConversation[],
  incoming: AgentConversation[],
) {
  const byId = new Map(current.map((conversation) => [conversation.conversationId, conversation]));
  for (const conversation of incoming) byId.set(conversation.conversationId, conversation);
  return [...byId.values()].sort((a, b) => {
    const left = new Date(a.lastMessageTime ?? a.createTime ?? 0).getTime();
    const right = new Date(b.lastMessageTime ?? b.createTime ?? 0).getTime();
    return right - left;
  });
}
