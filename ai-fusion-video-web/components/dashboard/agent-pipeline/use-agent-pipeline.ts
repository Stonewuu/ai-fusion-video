"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelPipeline,
  pipelineStream,
  type AiChatStreamEvent,
} from "@/lib/api/ai-pipeline";
import {
  createInitialPipelineState,
  createPendingPipelineState,
  reducePipelineEvent,
} from "./state";
import type { AgentPipelineProps, AgentPipelineState } from "./types";

export function useAgentPipeline({
  request,
  autoStart = false,
  onComplete,
  onError,
}: AgentPipelineProps) {
  const [state, setState] = useState<AgentPipelineState>(
    createInitialPipelineState
  );
  const abortRef = useRef<AbortController | null>(null);
  const startedRef = useRef(false);

  const handleEvent = useCallback((event: AiChatStreamEvent) => {
    setState((prev) => reducePipelineEvent(prev, event));
  }, []);

  const startStream = useCallback(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    setState(createPendingPipelineState());

    const controller = pipelineStream(request, {
      onEvent: handleEvent,
      onError: (err) => {
        setState((prev) => ({
          ...prev,
          error: `Pipeline 连接中断：${err.message}`,
        }));
        onError?.(err.message);
      },
      // Terminal status is reduced exclusively from the journal event.
      onComplete: () => undefined,
    });

    abortRef.current = controller;
  }, [request, handleEvent, onError]);

  useEffect(() => {
    if (autoStart && !startedRef.current) {
      queueMicrotask(() => {
        if (!startedRef.current) {
          startStream();
        }
      });
    }
  }, [autoStart, startStream]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    []
  );

  useEffect(() => {
    if (state.status === "done") {
      onComplete?.(state.conversationId);
    }
  }, [state.status, state.conversationId, onComplete]);

  const cancelStream = useCallback(async () => {
    if (!state.runId) {
      const message = "Pipeline 尚未返回 runId，无法提交取消请求";
      setState((prev) => ({ ...prev, error: message }));
      onError?.(message);
      return;
    }
    try {
      await cancelPipeline({ runId: state.runId });
      // Keep consuming until the durable CANCELLED terminal event arrives.
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setState((prev) => ({ ...prev, error: message }));
      onError?.(message);
    }
  }, [state.runId, onError]);

  const isActive =
    state.status === "reasoning" || state.status === "running";

  return {
    state,
    isActive,
    startStream,
    cancelStream,
  };
}
