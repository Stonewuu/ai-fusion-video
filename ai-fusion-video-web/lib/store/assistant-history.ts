import type { StoreApi } from "zustand";
import { listMessages } from "@/lib/api/ai-assistant";
import type { AssistantConversationRuntime, AssistantStoreState } from "./assistant-types";
import { mergeMessages, statusIsRunning, timelineForMessages } from "./assistant-runtime";

/** 历史读取独立于实时连接；迟到的 HTTP 响应不能覆盖更新的 SSE 轮次。 */
export function createAssistantHistoryLoader(
  set: StoreApi<AssistantStoreState>["setState"],
  get: StoreApi<AssistantStoreState>["getState"],
  getGeneration: () => number,
  updateRuntime: (
    conversationId: string,
    update: (runtime: AssistantConversationRuntime) => AssistantConversationRuntime,
  ) => void,
) {
  return async (conversationId: string) => {
    if (!conversationId) return;
    const userId = get().hydratedUserId;
    const generation = getGeneration();
    const runtime = get().conversationStates[conversationId];
    if (!runtime || runtime.messagesLoaded || runtime.messagesLoading) return;
    updateRuntime(conversationId, (current) => ({
      ...current,
      messagesLoading: true,
      messagesError: undefined,
    }));
    try {
      const incoming = await listMessages(conversationId);
      if (generation !== getGeneration() || get().hydratedUserId !== userId) return;
      let needsFreshHistory = false;
      set((state) => {
        const current = state.conversationStates[conversationId];
        if (!current) return state;
        const messages = mergeMessages(current.messages, incoming);
        // 请求期间结束或切到下一轮时，保留更晚到达的实时内容，再读取该轮完整历史。
        needsFreshHistory = runtime.knownRunId !== current.knownRunId
          || (statusIsRunning(runtime.status) && !statusIsRunning(current.status));
        const shouldBuildTimeline = !needsFreshHistory && !state.connection && !statusIsRunning(current.status);
        return {
          conversationStates: {
            ...state.conversationStates,
            [conversationId]: {
              ...current,
              messages,
              messagesLoaded: !needsFreshHistory,
              messagesLoading: false,
              messagesError: undefined,
              pipeline: shouldBuildTimeline
                ? { ...current.pipeline, timeline: timelineForMessages(messages), conversationId }
                : current.pipeline,
            },
          },
        };
      });
      if (needsFreshHistory && get().selectedConversationId === conversationId && get().mode !== "collapsed") {
        await get().loadMessagesIfNeeded(conversationId);
      }
    } catch (error: unknown) {
      if (generation !== getGeneration() || get().hydratedUserId !== userId) return;
      updateRuntime(conversationId, (current) => ({
        ...current,
        messagesLoading: false,
        messagesError: error instanceof Error ? error.message : "加载消息失败",
      }));
    }
  };
}
