"use client";

import { create } from "zustand";
import {
  listConversations,
  listMessages,
  type AgentConversation,
  type AgentMessage,
} from "@/lib/api/ai-assistant";
import { cancelPipeline, type AiChatReq } from "@/lib/api/ai-pipeline";
import {
  cancelCallingTimelineTools,
  restoreOptimisticallyCancelledTimelineTools,
} from "@/lib/store/pipeline-timeline";
import {
  clampDockWidth,
  clampLauncherPosition,
  clampRect,
  getDefaultLauncherPosition,
  getDefaultNormalRect,
  type AssistantPoint,
  type AssistantRect,
} from "@/components/dashboard/assistant/geometry";
import {
  makeRuntime,
  mergeMessages,
  normalizeTitle,
  pendingPipelineForNextRun,
  statusIsRunning,
  timelineForMessages,
  uniqueConversations,
} from "./assistant-runtime";
import {
  clearAssistantPersistTimer,
  commitAssistantPersist,
  defaultPersistedState,
  scheduleAssistantPersist,
} from "./assistant-persistence";
import { createAssistantConnectionCoordinator } from "./assistant-connection-coordinator";
import {
  ASSISTANT_CATEGORY,
  NEW_ASSISTANT_DRAFT_KEY,
  type AssistantConversationRuntime,
  type AssistantStoreState,
} from "./assistant-types";

export {
  ASSISTANT_CATEGORY,
  NEW_ASSISTANT_DRAFT_KEY,
} from "./assistant-types";
export type {
  AssistantConnection,
  AssistantConnectionMode,
  AssistantConversationRuntime,
  AssistantMode,
  AssistantStoreState,
} from "./assistant-types";

const PAGE_SIZE = 20;
const defaultRect = getDefaultNormalRect();
const defaultLauncher = getDefaultLauncherPosition();

export const useAssistantStore = create<AssistantStoreState>()((set, get) => {
  const connectionCoordinator = createAssistantConnectionCoordinator(set, get);

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

  const persist = () => scheduleAssistantPersist(get);

  return {
    hydratedUserId: null,
    initialized: false,
    mode: "collapsed",
    lastOpenMode: "floating",
    restoreMode: "floating",
    launcherPosition: defaultLauncher,
    normalRect: defaultRect,
    dockWidth: 520,
    selectedConversationId: null,
    selectedModelId: null,
    conversations: [],
    conversationStates: {},
    newDraft: "",
    drawerOpen: false,
    conversationsLoading: false,
    hasMoreConversations: false,
    conversationPage: 0,
    connection: null,
    connectionGeneration: 0,

    initializeForUser: (userId: number) => {
      if (!Number.isSafeInteger(userId) || userId <= 0) return;
      if (get().initialized && get().hydratedUserId === userId) return;

      if (get().hydratedUserId) commitAssistantPersist(get);
      connectionCoordinator.reset();
      clearAssistantPersistTimer();
      const persisted = defaultPersistedState(userId);
      const drafts = persisted.drafts ?? {};
      const restoredMode = persisted.mode ?? "collapsed";
      const initialMode = typeof window !== "undefined"
        && window.innerWidth < 650
        && restoredMode !== "collapsed"
        ? "maximized"
        : restoredMode;
      set({
        hydratedUserId: userId,
        initialized: true,
        mode: initialMode,
        lastOpenMode: persisted.lastOpenMode ?? "floating",
        restoreMode: persisted.restoreMode ?? "floating",
        launcherPosition: persisted.launcherPosition ?? defaultLauncher,
        normalRect: persisted.normalRect ?? defaultRect,
        dockWidth: persisted.dockWidth ?? 520,
        selectedConversationId: persisted.selectedConversationId ?? null,
        selectedModelId: persisted.selectedModelId ?? null,
        conversations: [],
        conversationStates: {},
        newDraft: drafts[NEW_ASSISTANT_DRAFT_KEY] ?? "",
        drawerOpen: false,
        conversationsLoading: true,
        conversationsError: undefined,
        hasMoreConversations: false,
        conversationPage: 0,
      });

      void listConversations({ pageNo: 1, pageSize: PAGE_SIZE, category: ASSISTANT_CATEGORY })
        .then((result) => {
          if (get().hydratedUserId !== userId) return;
          const filtered = result.list.filter((conversation) =>
            !conversation.category || conversation.category === ASSISTANT_CATEGORY,
          );
          set((state) => {
            const conversationStates: Record<string, AssistantConversationRuntime> = {};
            for (const conversation of filtered) {
              conversationStates[conversation.conversationId] = makeRuntime(
                conversation,
                drafts,
                persisted.runIds ?? {},
                persisted.lastSequences ?? {},
              );
            }
            const selected = state.selectedConversationId && conversationStates[state.selectedConversationId]
              ? state.selectedConversationId
              : null;
            return {
              conversations: filtered,
              conversationStates,
              selectedConversationId: selected,
              conversationsLoading: false,
              conversationPage: 1,
              hasMoreConversations: filtered.length < result.total,
              conversationsError: undefined,
            };
          });
          connectionCoordinator.scheduleStatusPolling();
          const selectedId = get().selectedConversationId;
          if (get().mode !== "collapsed" && selectedId) {
            void get().loadMessagesIfNeeded(selectedId);
            connectionCoordinator.ensureConnection();
          }
        })
        .catch((error: unknown) => {
          if (get().hydratedUserId !== userId) return;
          set({
            conversationsLoading: false,
            conversationsError: error instanceof Error ? error.message : "加载助手会话失败",
          });
        });
    },

    resetForUser: () => {
      if (get().hydratedUserId) commitAssistantPersist(get);
      connectionCoordinator.reset();
      clearAssistantPersistTimer();
      set({
        hydratedUserId: null,
        initialized: false,
        mode: "collapsed",
        lastOpenMode: "floating",
        restoreMode: "floating",
        launcherPosition: defaultLauncher,
        normalRect: defaultRect,
        dockWidth: 520,
        conversations: [],
        conversationStates: {},
        selectedConversationId: null,
        selectedModelId: null,
        newDraft: "",
        drawerOpen: false,
        conversationsLoading: false,
        conversationsError: undefined,
        hasMoreConversations: false,
        conversationPage: 0,
        connection: null,
      });
    },

    loadMoreConversations: () => {
      const state = get();
      if (state.conversationsLoading || !state.hasMoreConversations || !state.hydratedUserId) return;
      const page = state.conversationPage + 1;
      const userId = state.hydratedUserId;
      set({ conversationsLoading: true });
      void listConversations({ pageNo: page, pageSize: PAGE_SIZE, category: ASSISTANT_CATEGORY })
        .then((result) => {
          if (get().hydratedUserId !== userId) return;
          const filtered = result.list.filter((conversation) =>
            !conversation.category || conversation.category === ASSISTANT_CATEGORY,
          );
          set((current) => {
            const conversations = uniqueConversations(current.conversations, filtered);
            const persisted = defaultPersistedState(userId);
            const conversationStates = { ...current.conversationStates };
            for (const conversation of filtered) {
              const existing = conversationStates[conversation.conversationId];
              conversationStates[conversation.conversationId] = existing
                ? {
                    ...existing,
                    conversation: { ...conversation, status: existing.status },
                  }
                : makeRuntime(
                    conversation,
                    persisted.drafts ?? {},
                    persisted.runIds ?? {},
                    persisted.lastSequences ?? {},
                  );
            }
            return {
              conversations,
              conversationStates,
              conversationsLoading: false,
              conversationPage: page,
              hasMoreConversations: conversations.length < result.total,
            };
          });
          connectionCoordinator.scheduleStatusPolling();
        })
        .catch((error: unknown) => {
          if (get().hydratedUserId !== userId) return;
          set({
            conversationsLoading: false,
            conversationsError: error instanceof Error ? error.message : "加载更多会话失败",
          });
        });
    },

    selectConversation: (conversationId: string | null) => {
      const state = get();
      if (state.selectedConversationId !== conversationId) {
        connectionCoordinator.invalidateConnection();
      }
      set((current) => {
        if (!conversationId) return { selectedConversationId: null, drawerOpen: false };
        const runtime = current.conversationStates[conversationId];
        if (!runtime) return { selectedConversationId: conversationId, drawerOpen: false };
        return {
          selectedConversationId: conversationId,
          drawerOpen: false,
          conversationStates: {
            ...current.conversationStates,
            [conversationId]: { ...runtime, unread: false },
          },
        };
      });
      persist();
      connectionCoordinator.scheduleStatusPolling();
      if (conversationId && get().mode !== "collapsed") {
        void get().loadMessagesIfNeeded(conversationId);
        connectionCoordinator.ensureConnection();
      }
    },

    startNewConversation: () => {
      if (get().connection || get().selectedConversationId) connectionCoordinator.invalidateConnection();
      set({ selectedConversationId: null, drawerOpen: false });
      persist();
    },

    setDraft: (conversationId, draft) => {
      if (!conversationId) set({ newDraft: draft });
      else updateRuntime(conversationId, (runtime) => ({ ...runtime, draft }));
      persist();
    },

    setSelectedModelId: (modelId) => {
      set({ selectedModelId: modelId });
      persist();
    },

    sendMessage: async (message, modelId, projectId) => {
      const content = message.trim();
      if (!content) return;
      const state = get();
      if (state.connection) throw new Error("当前会话仍在生成中");

      const selectedId = state.selectedConversationId;
      let conversationId = selectedId;
      const title = normalizeTitle(content);
      if (!conversationId) {
        conversationId = typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `assistant-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const conversation: AgentConversation = {
          id: -Date.now(),
          conversationId,
          userId: state.hydratedUserId ?? 0,
          projectId: projectId ?? null,
          category: ASSISTANT_CATEGORY,
          title,
          messageCount: 0,
          status: "completed",
        };
        const runtime = {
          ...makeRuntime(conversation, {}, {}, {}),
          // The optimistic conversation has no server history yet. Treat its
          // empty local transcript as loaded so connection recovery cannot
          // race the create request with a history lookup that must 404.
          messagesLoaded: true,
        };
        set((current) => ({
          conversations: [conversation, ...current.conversations],
          conversationStates: { ...current.conversationStates, [conversationId!]: runtime },
          selectedConversationId: conversationId,
          newDraft: "",
        }));
      }

      let runtime = get().conversationStates[conversationId];
      if (!runtime) throw new Error("会话尚未准备好");
      if (selectedId && statusIsRunning(runtime.status)) throw new Error("当前会话仍在生成中");
      const shouldSetTitle = !selectedId || runtime.conversation.title === "新对话";

      // A completed live run may have just projected its final message. Give
      // the history endpoint a chance to materialize it before starting the
      // next turn, while preserving the in-memory timeline as a fallback.
      if (runtime.pipeline.timeline.length > 0 && !runtime.messagesLoaded) {
        await get().loadMessagesIfNeeded(conversationId);
        runtime = get().conversationStates[conversationId] ?? runtime;
      }

      const optimisticMessage: AgentMessage = {
        id: -Date.now(),
        conversationId,
        role: "user",
        content,
        messageOrder: Math.max(0, ...runtime.messages.map((item) => item.messageOrder ?? 0)) + 1,
      };
      const pendingPipeline = {
        ...pendingPipelineForNextRun(conversationId),
        // Keep an already visible answer until the persisted projection is
        // available; new events append to this same reducer state.
        timeline: runtime.messagesLoaded ? [] : runtime.pipeline.timeline,
      };
      const conversationTitle = runtime.conversation.title === "新对话" ? title : runtime.conversation.title;
      const conversationProjectId = selectedId
        ? runtime.conversation.projectId
        : runtime.conversation.projectId ?? projectId ?? null;
      updateRuntime(conversationId, (current) => ({
        ...current,
        messages: [...current.messages, optimisticMessage],
        pipeline: pendingPipeline,
        status: "running",
        statusConfirmed: true,
        knownRunId: undefined,
        remoteLastSequence: 0,
        messagesError: undefined,
        connectionError: undefined,
        conversation: {
          ...current.conversation,
          status: "running",
          title: conversationTitle,
          projectId: conversationProjectId,
        },
      }));
      set((current) => ({
        conversations: current.conversations.map((item) => item.conversationId === conversationId
          ? { ...item, status: "running", title: conversationTitle, projectId: conversationProjectId }
          : item),
      }));
      persist();

      const request: AiChatReq = {
        message: content,
        conversationId,
        modelId: modelId ?? undefined,
        category: ASSISTANT_CATEGORY,
        title: shouldSetTitle ? title : undefined,
        projectId: conversationProjectId ?? undefined,
      };
      connectionCoordinator.startConnection(conversationId, request);
      connectionCoordinator.scheduleStatusPolling();
    },

    stopGeneration: async () => {
      const state = get();
      const conversationId = state.selectedConversationId;
      const runtime = conversationId ? state.conversationStates[conversationId] : undefined;
      const connectionRunId = state.connection?.conversationId === conversationId
        ? state.connection.runId
        : undefined;
      const runId = connectionRunId || runtime?.pipeline.runId || runtime?.knownRunId;
      if (!conversationId || !runtime || runtime.status === "CANCEL_REQUESTED") return;
      const previousStatus = runtime.status;
      const previousConversationStatus = runtime.conversation.status;
      const previousPipelineStatus = runtime.pipeline.status;
      const previousTimeline = runtime.pipeline.timeline;
      const previousListStatus = state.conversations.find(
        (item) => item.conversationId === conversationId,
      )?.status;
      updateRuntime(conversationId, (current) => ({
        ...current,
        status: "CANCEL_REQUESTED",
        connectionError: undefined,
        pipeline: {
          ...current.pipeline,
          status: "cancelling",
          timeline: cancelCallingTimelineTools(current.pipeline.timeline),
        },
        conversation: { ...current.conversation, status: "CANCEL_REQUESTED" },
      }));
      set((current) => ({
        conversations: current.conversations.map((item) => item.conversationId === conversationId
          ? { ...item, status: "CANCEL_REQUESTED" }
          : item),
      }));
      try {
        await cancelPipeline(runId ? { runId } : { conversationId });
        connectionCoordinator.scheduleStatusPolling();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        updateRuntime(conversationId, (current) => current.status === "CANCEL_REQUESTED"
          ? {
              ...current,
              status: previousStatus,
              connectionError: `取消请求失败：${message}`,
              pipeline: {
                ...current.pipeline,
                status: previousPipelineStatus,
                timeline: restoreOptimisticallyCancelledTimelineTools(
                  current.pipeline.timeline,
                  previousTimeline,
                ),
              },
              conversation: {
                ...current.conversation,
                status: previousConversationStatus,
              },
            }
          : current);
        set((current) => ({
          conversations: current.conversations.map((item) =>
            item.conversationId === conversationId && item.status === "CANCEL_REQUESTED"
              ? { ...item, status: previousListStatus ?? previousStatus }
              : item),
        }));
        throw error;
      }
    },

    markConversationRead: (conversationId) => updateRuntime(
      conversationId,
      (runtime) => ({ ...runtime, unread: false }),
    ),

    deleteConversation: async (conversationId, id) => {
      const runtime = get().conversationStates[conversationId];
      if (runtime && statusIsRunning(runtime.status)) throw new Error("运行中的会话不能删除");
      const { deleteConversation } = await import("@/lib/api/ai-assistant");
      let persistedId = id;
      if (id < 0) {
        const result = await listConversations({
          pageNo: 1,
          pageSize: PAGE_SIZE,
          category: ASSISTANT_CATEGORY,
        });
        const persisted = result.list.find((item) => item.conversationId === conversationId);
        if (!persisted) throw new Error("会话尚未同步，请稍后重试");
        persistedId = persisted.id;
      }
      await deleteConversation(persistedId);
      if (get().selectedConversationId === conversationId) connectionCoordinator.invalidateConnection();
      set((state) => {
        const conversationStates = { ...state.conversationStates };
        delete conversationStates[conversationId];
        return {
          conversations: state.conversations.filter((item) => item.conversationId !== conversationId),
          conversationStates,
          selectedConversationId: state.selectedConversationId === conversationId ? null : state.selectedConversationId,
        };
      });
      persist();
    },

    setMode: (mode, canDock = true) => {
      const current = get().mode;
      const nextMode = mode === "docked" && !canDock ? "floating" : mode;
      if (current === nextMode) return;
      const commitImmediately = nextMode === "collapsed";
      if (nextMode === "collapsed") {
        connectionCoordinator.invalidateConnection();
        set({ mode: "collapsed", drawerOpen: false });
        connectionCoordinator.scheduleStatusPolling();
      } else {
        set((state) => ({
          mode: nextMode,
          lastOpenMode: nextMode === "maximized" ? state.lastOpenMode : nextMode,
          restoreMode: nextMode === "maximized"
            ? (current === "docked" ? "docked" : "floating")
            : state.restoreMode,
          drawerOpen: false,
        }));
        if (current === "collapsed") {
          const selectedId = get().selectedConversationId;
          if (selectedId) {
            void get().loadMessagesIfNeeded(selectedId);
            connectionCoordinator.ensureConnection();
          }
        }
      }
      if (commitImmediately) commitAssistantPersist(get);
      else persist();
    },

    openAssistant: (canDock = true) => {
      const state = get();
      const mobileViewport = typeof window !== "undefined" && window.innerWidth < 650;
      const desired = mobileViewport
        ? "maximized"
        : state.lastOpenMode === "maximized"
          ? state.restoreMode
          : state.lastOpenMode;
      state.setMode(desired === "docked" && !canDock ? "floating" : desired, canDock);
    },

    closeAssistant: () => get().setMode("collapsed"),
    setDrawerOpen: (open) => set({ drawerOpen: open }),

    updateLauncherPosition: (position: AssistantPoint, commit = false) => {
      set({ launcherPosition: clampLauncherPosition(position) });
      if (commit) commitAssistantPersist(get);
    },

    updateNormalRect: (rect: AssistantRect, commit = false) => {
      set({ normalRect: clampRect(rect) });
      if (commit) commitAssistantPersist(get);
    },

    updateDockWidth: (width, availableWidth, commit = false) => {
      const next = clampDockWidth(width, availableWidth);
      if (!next) return;
      set({ dockWidth: next });
      if (commit) commitAssistantPersist(get);
    },

    clampViewportGeometry: (availableWidth) => {
      const viewport = typeof window === "undefined"
        ? undefined
        : { width: window.innerWidth, height: window.innerHeight };
      let changed = false;
      set((state) => {
        const launcherPosition = clampLauncherPosition(state.launcherPosition, viewport);
        const normalRect = clampRect(state.normalRect, viewport);
        const dockWidth = availableWidth !== undefined
          ? clampDockWidth(state.dockWidth, availableWidth) || state.dockWidth
          : state.dockWidth;
        changed = launcherPosition.x !== state.launcherPosition.x
          || launcherPosition.y !== state.launcherPosition.y
          || normalRect.x !== state.normalRect.x
          || normalRect.y !== state.normalRect.y
          || normalRect.width !== state.normalRect.width
          || normalRect.height !== state.normalRect.height
          || dockWidth !== state.dockWidth;
        return changed ? { launcherPosition, normalRect, dockWidth } : state;
      });
      if (changed) persist();
    },

    loadMessagesIfNeeded: async (conversationId) => {
      if (!conversationId) return;
      const userId = get().hydratedUserId;
      const runtime = get().conversationStates[conversationId];
      if (!runtime || runtime.messagesLoaded || runtime.messagesLoading) return;
      updateRuntime(conversationId, (current) => ({
        ...current,
        messagesLoading: true,
        messagesError: undefined,
      }));
      try {
        const incoming = await listMessages(conversationId);
        if (get().hydratedUserId !== userId) return;
        set((state) => {
          const current = state.conversationStates[conversationId];
          if (!current) return state;
          const messages = mergeMessages(current.messages, incoming);
          const shouldBuildTimeline = !state.connection && !statusIsRunning(current.status);
          return {
            conversationStates: {
              ...state.conversationStates,
              [conversationId]: {
                ...current,
                messages,
                messagesLoaded: true,
                messagesLoading: false,
                messagesError: undefined,
                pipeline: shouldBuildTimeline
                  ? {
                      ...current.pipeline,
                      timeline: timelineForMessages(messages),
                      conversationId,
                    }
                  : current.pipeline,
              },
            },
          };
        });
      } catch (error: unknown) {
        updateRuntime(conversationId, (current) => ({
          ...current,
          messagesLoading: false,
          messagesError: error instanceof Error ? error.message : "加载消息失败",
        }));
      }
    },

    ensureContentConnection: connectionCoordinator.ensureConnection,
  };
});
