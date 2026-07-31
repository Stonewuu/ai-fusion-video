import { triggerToolInvalidation, usePipelineStore } from "./pipeline-store";
import type { StoreApi } from "zustand";
import {
  getPipelineStatus,
  pipelineStream,
  reconnectPipelineStream,
  type AiChatReq,
  type AiChatStreamEvent,
  type PipelineRunStatusResponse,
} from "@/lib/api/ai-pipeline";
import type {
  AssistantConnectionMode,
  AssistantConversationRuntime,
  AssistantStoreState,
} from "./assistant-types";
import {
  hasTerminal,
  reduceAssistantEvent,
  statusFromPipeline,
  statusIsRunning,
  terminalStatusForEvent,
} from "./assistant-runtime";

type SetState = StoreApi<AssistantStoreState>["setState"];
type GetState = StoreApi<AssistantStoreState>["getState"];

export interface AssistantConnectionCoordinator {
  invalidateConnection: () => number;
  clearPolling: () => void;
  reset: () => void;
  scheduleStatusPolling: () => void;
  ensureConnection: () => void;
  startConnection: (conversationId: string, request: AiChatReq) => void;
}

export function createAssistantConnectionCoordinator(
  set: SetState,
  get: GetState,
): AssistantConnectionCoordinator {
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let pollInFlightGeneration: number | null = null;
  let pollFailureCount = 0;
  let lifecycleGeneration = 0;
  let ensureGeneration = 0;
  let pendingEnsure: { conversationId: string; generation: number } | null = null;
  let ensureRetryTimer: ReturnType<typeof setTimeout> | null = null;
  const eventKeysByRun = new Map<string, Set<string>>();
  const metadataGenerations = new Map<string, number>();

  const beginMetadataRequest = (conversationId: string) => {
    const generation = (metadataGenerations.get(conversationId) ?? 0) + 1;
    metadataGenerations.set(conversationId, generation);
    return generation;
  };

  const invalidateMetadataRequests = (conversationId: string) => {
    metadataGenerations.set(
      conversationId,
      (metadataGenerations.get(conversationId) ?? 0) + 1,
    );
  };

  const isCurrentMetadataRequest = (conversationId: string, generation: number) =>
    metadataGenerations.get(conversationId) === generation;

  const updateRuntime = (
    conversationId: string,
    updater: (runtime: AssistantConversationRuntime) => AssistantConversationRuntime,
  ) => {
    set((state) => {
      const runtime = state.conversationStates[conversationId];
      if (!runtime) return state;
      return {
        conversationStates: {
          ...state.conversationStates,
          [conversationId]: updater(runtime),
        },
      };
    });
  };

  const invalidatePendingEnsure = () => {
    ensureGeneration += 1;
    pendingEnsure = null;
    if (ensureRetryTimer) clearTimeout(ensureRetryTimer);
    ensureRetryTimer = null;
  };

  const scheduleEnsureRetry = (delay = 5000) => {
    if (ensureRetryTimer) return;
    const generation = ensureGeneration;
    ensureRetryTimer = setTimeout(() => {
      ensureRetryTimer = null;
      if (generation === ensureGeneration) ensureConnection();
    }, delay);
  };

  const invalidateConnection = () => {
    invalidatePendingEnsure();
    const current = get().connection;
    if (current) current.controller.abort();
    const nextGeneration = get().connectionGeneration + 1;
    set({ connection: null, connectionGeneration: nextGeneration });
    scheduleStatusPolling();
    return nextGeneration;
  };

  const clearConnection = (generation: number) => {
    const current = get().connection;
    if (!current || current.connectionGeneration !== generation) return;
    set({ connection: null, connectionGeneration: generation + 1 });
  };

  const clearPolling = () => {
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
  };

  const pollingCandidates = () => {
    const state = get();
    const activeConversationId = state.connection?.conversationId;
    return state.conversations.filter((conversation) => {
      if (conversation.conversationId === activeConversationId) return false;
      if (conversation.conversationId === pendingEnsure?.conversationId) return false;
      const runtime = state.conversationStates[conversation.conversationId];
      return statusIsRunning(runtime?.status ?? conversation.status);
    });
  };

  const applyPolledStatus = (
    conversationId: string,
    response: PipelineRunStatusResponse,
    requestGeneration: number,
    metadataGeneration: number,
    expectedRunId?: string,
  ) => {
    if (requestGeneration !== lifecycleGeneration
      || !isCurrentMetadataRequest(conversationId, metadataGeneration)) return;
    const nextStatus = statusFromPipeline(response.status);
    const terminal = !statusIsRunning(nextStatus);
    let applied = false;
    set((state) => {
      const runtime = state.conversationStates[conversationId];
      // Background status polling queries discrete conversation pipeline status.
      // Do not overwrite metadata if a live SSE stream is active on this exact conversation.
      if (!runtime
        || state.connection?.conversationId === conversationId
        || !isCurrentMetadataRequest(conversationId, metadataGeneration)) return state;
      if (expectedRunId
        && runtime.statusConfirmed
        && statusIsRunning(runtime.status)
        && runtime.knownRunId !== expectedRunId) {
        return state;
      }

      const runChanged = !!runtime.knownRunId
        && !!response.runId
        && runtime.knownRunId !== response.runId;
      const nextConversation = { ...runtime.conversation, status: nextStatus };
      applied = true;
      return {
        conversations: state.conversations.map((item) => item.conversationId === conversationId
          ? nextConversation
          : item),
        conversationStates: {
          ...state.conversationStates,
          [conversationId]: {
            ...runtime,
            conversation: nextConversation,
            status: nextStatus,
            statusConfirmed: true,
            knownRunId: response.runId || runtime.knownRunId,
            remoteLastSequence: response.lastSequence,
            connectionError: terminal ? undefined : runtime.connectionError,
            // A background completion makes the persisted transcript stale,
            // but it still must not trigger a content request while closed.
            messagesLoaded: terminal || runChanged ? false : runtime.messagesLoaded,
            unread: terminal
              && (state.mode === "collapsed" || state.selectedConversationId !== conversationId)
              ? true
              : runtime.unread,
          },
        },
      };
    });
    if (!applied) return;

    const latest = get();
    if (terminal) {
      if (latest.mode !== "collapsed" && latest.selectedConversationId === conversationId) {
        void latest.loadMessagesIfNeeded(conversationId);
      }
    }
  };

  const scheduleStatusPolling = () => {
    if (pollTimer || pollInFlightGeneration !== null || pollingCandidates().length === 0) return;
    const hidden = typeof document !== "undefined" && document.visibilityState === "hidden";
    //如果有运行中的对话，页面可见时固定每隔 1 秒 (1000ms) 查询一次对话状态，以便实时更新状态
    const baseDelay = hidden ? 5000 : 1000;
    const backoff = Math.min(5000, baseDelay * (2 ** pollFailureCount));
    const delay = hidden ? Math.max(5000, backoff) : backoff;
    pollTimer = setTimeout(() => {
      pollTimer = null;
      void pollStatuses();
    }, delay);
  };

  const pollStatuses = async () => {
    if (pollInFlightGeneration !== null) return;
    const requestGeneration = lifecycleGeneration;
    const candidates = pollingCandidates().map((conversation) => ({
      conversation,
      expectedRunId: get().conversationStates[conversation.conversationId]?.knownRunId,
      metadataGeneration: beginMetadataRequest(conversation.conversationId),
    }));
    if (candidates.length === 0) {
      clearPolling();
      return;
    }

    pollInFlightGeneration = requestGeneration;
    let failures = 0;
    try {
      await Promise.all(candidates.map(async ({
        conversation,
        expectedRunId,
        metadataGeneration,
      }) => {
        try {
          const response = await getPipelineStatus({ conversationId: conversation.conversationId });
          applyPolledStatus(
            conversation.conversationId,
            response,
            requestGeneration,
            metadataGeneration,
            expectedRunId,
          );
        } catch {
          failures += 1;
          // A transient metadata failure never becomes a Pipeline failure.
        }
      }));
    } finally {
      if (requestGeneration !== lifecycleGeneration
        || pollInFlightGeneration !== requestGeneration) return;
      pollInFlightGeneration = null;
      pollFailureCount = failures === candidates.length
        ? Math.min(pollFailureCount + 1, 4)
        : 0;
      if (pollingCandidates().length > 0) scheduleStatusPolling();
      else clearPolling();
    }
  };

  const alignCursorWithRun = (
    conversationId: string,
    runId: string,
  ) => {
    let afterSequence = 0;
    set((state) => {
      const runtime = state.conversationStates[conversationId];
      if (!runtime) return state;
      const sameRun = runtime.pipeline.runId === runId;
      // Server status metadata never advances or rewinds the local replay
      // cursor. A cursor is reusable only when it belongs to this exact run.
      afterSequence = sameRun ? runtime.pipeline.lastSequence : 0;
      const cursorUnchanged = sameRun && afterSequence === runtime.pipeline.lastSequence;
      return {
        conversationStates: {
          ...state.conversationStates,
          [conversationId]: {
            ...runtime,
            knownRunId: runId,
            pipeline: cursorUnchanged
              ? runtime.pipeline
              : {
                  ...runtime.pipeline,
                  runId,
                  conversationId,
                  lastSequence: afterSequence,
                  reasoningText: "",
                  reasoningStartTime: undefined,
                  reasoningDurationMs: undefined,
                  error: undefined,
                },
          },
        },
      };
    });
    return afterSequence;
  };

  const connect = (
    conversationId: string,
    connectionMode: AssistantConnectionMode,
    request?: AiChatReq,
    reconnectRunId?: string,
    reconnectAfterSequence = 0,
  ) => {
    const state = get();
    if (state.mode === "collapsed" || state.selectedConversationId !== conversationId) return;
    const runtime = state.conversationStates[conversationId];
    if (!runtime || !statusIsRunning(runtime.status)) return;
    if (state.connection?.conversationId === conversationId) return;
    if (connectionMode === "reconnect" && !reconnectRunId) return;

    const generation = invalidateConnection();
    invalidateMetadataRequests(conversationId);
    updateRuntime(conversationId, (runtimeValue) => ({
      ...runtimeValue,
      connectionError: undefined,
    }));
    const callbacks = {
      onEvent: (event: AiChatStreamEvent) => {
        const current = get().connection;
        if (!current
          || current.connectionGeneration !== generation
          || current.conversationId !== conversationId) return;
        if (current.runId && event.runId !== current.runId) return;
        invalidateMetadataRequests(conversationId);

        const runKey = `${conversationId}:${event.runId}`;
        const eventSet = eventKeysByRun.get(runKey) ?? new Set<string>();
        eventKeysByRun.set(runKey, eventSet);
        const eventKey = `${event.sequence}:${event.messageId ?? event.outputType}`;
        if (eventSet.has(eventKey)) return;
        eventSet.add(eventKey);

        const currentRuntime = get().conversationStates[conversationId];
        if (!currentRuntime
          || (currentRuntime.pipeline.runId && currentRuntime.pipeline.runId !== event.runId)
          || event.sequence <= currentRuntime.pipeline.lastSequence) return;

        try {
          const nextPipeline = reduceAssistantEvent(currentRuntime.pipeline, event);
          const terminal = hasTerminal(event);
          const nextStatus = terminal
            ? terminalStatusForEvent(event)
            : currentRuntime.status === "CANCEL_REQUESTED"
              ? "CANCEL_REQUESTED"
              : event.outputType === "USER_CONFIRMATION_REQUIRED"
                ? "WAITING_CONFIRMATION"
                : event.outputType === "EXTERNAL_EXECUTION_REQUIRED"
                  ? "WAITING_EXTERNAL"
                  : "running";
          if (event.outputType === "TOOL_FINISHED" && event.toolName && event.toolStatus !== "error") {
            triggerToolInvalidation(event.toolName);
          } else if (terminal) {
            const pStore = usePipelineStore.getState();
            pStore.triggerInvalidation("scripts");
            pStore.triggerInvalidation("storyboards");
            pStore.triggerInvalidation("assets");
          }

          updateRuntime(conversationId, (runtimeValue) => ({
            ...runtimeValue,
            pipeline: {
              ...nextPipeline,
              runId: event.runId,
              conversationId,
              lastSequence: event.sequence,
            },
            status: nextStatus,
            statusConfirmed: true,
            knownRunId: event.runId,
            remoteLastSequence: Math.max(runtimeValue.remoteLastSequence ?? 0, event.sequence),
            connectionError: undefined,
            messagesLoaded: terminal ? false : runtimeValue.messagesLoaded,
            conversation: { ...runtimeValue.conversation, status: nextStatus },
            unread: terminal
              && (get().mode === "collapsed" || get().selectedConversationId !== conversationId),
          }));
          set((storeState) => {
            const indexedConversation = storeState.conversations.find(
              (item) => item.conversationId === conversationId,
            );
            return {
              conversations: indexedConversation?.status === nextStatus
                ? storeState.conversations
                : storeState.conversations.map((item) => item.conversationId === conversationId
                    ? { ...item, status: nextStatus }
                    : item),
              connection: storeState.connection?.connectionGeneration === generation
                ? { ...storeState.connection, runId: event.runId }
                : storeState.connection,
            };
          });
        } catch (error) {
          updateRuntime(conversationId, (runtimeValue) => ({
            ...runtimeValue,
            connectionError: error instanceof Error ? error.message : "无法处理助手事件",
          }));
        }
      },
      onError: (error: Error) => {
        const current = get().connection;
        if (!current
          || current.connectionGeneration !== generation
          || current.conversationId !== conversationId) return;
        const startRejected = current.connectionMode === "start" && !current.runId;
        updateRuntime(conversationId, (runtimeValue) => ({
          ...runtimeValue,
          connectionError: error.message,
          status: startRejected ? "failed" : runtimeValue.status,
          statusConfirmed: startRejected ? true : runtimeValue.statusConfirmed,
          conversation: startRejected
            ? { ...runtimeValue.conversation, status: "failed" }
            : runtimeValue.conversation,
        }));
        if (startRejected) {
          set((state) => ({
            conversations: state.conversations.map((conversation) =>
              conversation.conversationId === conversationId
                ? { ...conversation, status: "failed" }
                : conversation),
          }));
        }
        clearConnection(generation);
        if (startRejected) return;
        scheduleEnsureRetry();
        scheduleStatusPolling();
      },
      onComplete: () => {
        const current = get().connection;
        if (!current
          || current.connectionGeneration !== generation
          || current.conversationId !== conversationId) return;
        clearConnection(generation);
        const latest = get();
        const latestRuntime = latest.conversationStates[conversationId];
        if (latest.mode !== "collapsed"
          && latest.selectedConversationId === conversationId
          && latestRuntime
          && !statusIsRunning(latestRuntime.status)) {
          void latest.loadMessagesIfNeeded(conversationId);
        }
        scheduleStatusPolling();
      },
    };

    let controller: AbortController;
    try {
      if (connectionMode === "start") {
        if (!request) throw new Error("缺少助手请求");
        controller = pipelineStream(request, callbacks);
      } else {
        controller = reconnectPipelineStream(
          reconnectRunId!,
          reconnectAfterSequence,
          callbacks,
        );
      }
    } catch (error) {
      updateRuntime(conversationId, (runtimeValue) => ({
        ...runtimeValue,
        connectionError: error instanceof Error ? error.message : "无法连接助手",
      }));
      scheduleEnsureRetry();
      scheduleStatusPolling();
      return;
    }

    set({
      connection: {
        conversationId,
        runId: reconnectRunId,
        controller,
        connectionGeneration: generation,
        connectionMode,
      },
    });
  };

  const confirmStatusAndReconnect = (conversationId: string) => {
    if (pendingEnsure?.conversationId === conversationId) return;
    const generation = ++ensureGeneration;
    const requestConnectionGeneration = get().connectionGeneration;
    const metadataGeneration = beginMetadataRequest(conversationId);
    pendingEnsure = { conversationId, generation };

    void getPipelineStatus({ conversationId })
      .then((response) => {
        if (generation !== ensureGeneration
          || get().connectionGeneration !== requestConnectionGeneration
          || !isCurrentMetadataRequest(conversationId, metadataGeneration)) return;
        const nextStatus = statusFromPipeline(response.status);
        set((state) => {
          const runtime = state.conversationStates[conversationId];
          if (!runtime
            || state.connectionGeneration !== requestConnectionGeneration
            || !isCurrentMetadataRequest(conversationId, metadataGeneration)) return state;
          const nextConversation = { ...runtime.conversation, status: nextStatus };
          return {
            conversations: state.conversations.map((item) => item.conversationId === conversationId
              ? nextConversation
              : item),
            conversationStates: {
              ...state.conversationStates,
              [conversationId]: {
                ...runtime,
                conversation: nextConversation,
                status: nextStatus,
                statusConfirmed: true,
                knownRunId: response.runId,
                remoteLastSequence: response.lastSequence,
                connectionError: statusIsRunning(nextStatus) ? runtime.connectionError : undefined,
                messagesLoaded: statusIsRunning(nextStatus) ? runtime.messagesLoaded : false,
              },
            },
          };
        });

        const latest = get();
        if (!statusIsRunning(nextStatus)) {
          if (latest.mode !== "collapsed" && latest.selectedConversationId === conversationId) {
            void latest.loadMessagesIfNeeded(conversationId);
          }
          scheduleStatusPolling();
          return;
        }
        if (latest.mode === "collapsed"
          || latest.selectedConversationId !== conversationId
          || latest.connection
          || !response.runId) {
          scheduleStatusPolling();
          return;
        }

        const afterSequence = alignCursorWithRun(
          conversationId,
          response.runId,
        );
        connect(conversationId, "reconnect", undefined, response.runId, afterSequence);
      })
      .catch(() => {
        // Confirmation is metadata-only. Leave the run state untouched and
        // retry without letting the polling request mutate content state.
        if (generation === ensureGeneration) scheduleEnsureRetry();
        scheduleStatusPolling();
      })
      .finally(() => {
        if (pendingEnsure?.generation === generation) pendingEnsure = null;
      });
  };

  const ensureConnection = () => {
    const state = get();
    const conversationId = state.selectedConversationId;
    if (state.mode === "collapsed" || !conversationId) {
      scheduleStatusPolling();
      return;
    }
    const runtime = state.conversationStates[conversationId];
    if (!runtime || !statusIsRunning(runtime.status)) {
      scheduleStatusPolling();
      return;
    }
    if (state.connection?.conversationId === conversationId) return;

    void state.loadMessagesIfNeeded(conversationId);
    if (!runtime.statusConfirmed || !runtime.knownRunId) {
      confirmStatusAndReconnect(conversationId);
      return;
    }

    const afterSequence = alignCursorWithRun(
      conversationId,
      runtime.knownRunId,
    );
    connect(conversationId, "reconnect", undefined, runtime.knownRunId, afterSequence);
  };

  return {
    invalidateConnection,
    clearPolling,
    reset: () => {
      lifecycleGeneration += 1;
      invalidateConnection();
      clearPolling();
      pollInFlightGeneration = null;
      pollFailureCount = 0;
      eventKeysByRun.clear();
      metadataGenerations.clear();
    },
    scheduleStatusPolling,
    ensureConnection,
    startConnection: (conversationId, request) => connect(conversationId, "start", request),
  };
}
