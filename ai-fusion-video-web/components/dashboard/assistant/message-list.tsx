"use client";

import {
  Fragment,
  memo,
  useMemo,
} from "react";
import { ArrowDown, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import {
  MessageTimeline,
  type TimelineToolConfirmation,
} from "@/components/dashboard/notification-panel/timeline";
import { messagesToTimeline } from "@/components/dashboard/notification-panel/history";
import type { AgentMessage } from "@/lib/api/ai-assistant";
import type { TimelineItem } from "@/lib/store/pipeline-store";
import { useAssistantStore, type AssistantConversationRuntime } from "@/lib/store/assistant-store";
import { cn } from "@/lib/utils";
import { AssistantBrandIcon } from "./assistant-brand-icon";
import { useAssistantMessageScroll } from "./use-assistant-message-scroll";
import { UserMessageBubble } from "./user-message-bubble";

interface AssistantMessageListProps {
  conversationId: string;
  inputHeight: number;
}

interface MessageSegment {
  key: string;
  user?: AgentMessage;
  assistant: AgentMessage[];
}

interface RenderableMessageSegment extends MessageSegment {
  timeline: TimelineItem[];
}

const EMPTY_TIMELINE: TimelineItem[] = [];
function isRunning(runtime: AssistantConversationRuntime) {
  return [
    "running",
    "pending",
    "RUNNING",
    "WAITING_CONFIRMATION",
    "WAITING_EXTERNAL",
    "CANCEL_REQUESTED",
  ].includes(runtime.status);
}

function buildSegments(messages: AgentMessage[], activeRunId?: string): MessageSegment[] {
  const segments: MessageSegment[] = [];
  let current: MessageSegment = { key: "prelude", assistant: [] };

  const pushCurrent = () => {
    if (current.user || current.assistant.length > 0) segments.push(current);
  };

  for (const message of messages) {
    if (message.role === "user") {
      pushCurrent();
      current = {
        key: `segment-${message.id}-${message.messageOrder}`,
        user: message,
        assistant: [],
      };
      continue;
    }
    // The live reducer is the authoritative view of the selected active run.
    // Do not render its projected rows a second time from the history API.
    if (activeRunId && message.runId === activeRunId) continue;
    current.assistant.push(message);
  }
  pushCurrent();
  return segments;
}

const AssistantTimelineContent = memo(function AssistantTimelineContent({
  timeline,
  reasoningText,
  reasoningStartTime,
  reasoningDurationMs,
  streaming = false,
  toolConfirmation,
}: {
  timeline: TimelineItem[];
  reasoningText?: string;
  reasoningStartTime?: number;
  reasoningDurationMs?: number;
  streaming?: boolean;
  toolConfirmation?: TimelineToolConfirmation;
}) {
  if (timeline.length === 0) return null;
  return (
    <div className="min-w-0 w-full">
      <MessageTimeline
        reasoningText={reasoningText}
        reasoningStartTime={reasoningStartTime}
        reasoningDurationMs={reasoningDurationMs}
        timeline={timeline}
        initialScrollToEnd={false}
        streaming={streaming}
        error={undefined}
        toolConfirmation={toolConfirmation}
      />
    </div>
  );
});

export function AssistantMessageList({
  conversationId,
  inputHeight,
}: AssistantMessageListProps) {
  const runtime = useAssistantStore((state) => state.conversationStates[conversationId]);
  const loadMessages = useAssistantStore((state) => state.loadMessagesIfNeeded);
  const ensureConnection = useAssistantStore((state) => state.ensureContentConnection);
  const respondToToolConfirmation = useAssistantStore(
    (state) => state.respondToToolConfirmation,
  );

  const running = !!runtime && isRunning(runtime);
  const contentReady = !!runtime && (runtime.messagesLoaded || !!runtime.messagesError);
  const contentVersion = runtime
    ? `${runtime.messagesLoaded}:${runtime.messages.length}:${runtime.pipeline.lastSequence}`
    : "pending";
  const showPipelineTimeline = !!runtime && (running || !runtime.messagesLoaded);
  const activeRunId = showPipelineTimeline
    ? runtime?.pipeline.runId ?? runtime?.knownRunId
    : undefined;
  const messages = runtime?.messages;
  const segments = useMemo<RenderableMessageSegment[]>(
    () => buildSegments(messages ?? [], activeRunId).map((segment) => ({
      ...segment,
      timeline: messagesToTimeline(segment.assistant),
    })),
    [activeRunId, messages],
  );
  const liveTimeline = runtime && showPipelineTimeline
    ? runtime.pipeline.timeline
    : EMPTY_TIMELINE;
  const error = runtime?.messagesError || runtime?.pipeline.error || runtime?.connectionError;
  const pendingConfirmation = runtime?.pipeline.pendingConfirmation;

  const {
    viewportRef,
    contentRef,
    viewportReady,
    showBackToBottom,
    onViewportScroll,
    onWheel,
    onKeyDown,
    onTouchStart,
    onTouchMove,
    onScrollbarPointerDown,
    scrollToBottom,
  } = useAssistantMessageScroll({
    contentReady,
    contentVersion,
    running,
    inputHeight,
  });

  if (!runtime) return null;

  const hasContent = segments.length > 0
    || liveTimeline.length > 0
    || !!pendingConfirmation
    || !!error;

  return (
    <div data-assistant-message-list className="relative min-h-0 flex-1">
      <OverlayScrollArea
        data-assistant-message-content
        data-ready={viewportReady ? "true" : "false"}
        className={cn(
          "h-full transition-opacity duration-[400ms] ease-out motion-reduce:transition-none",
          viewportReady ? "opacity-100" : "opacity-0",
        )}
        viewportRef={viewportRef}
        onViewportWheel={onWheel}
        onViewportKeyDown={onKeyDown}
        onViewportTouchStart={onTouchStart}
        onViewportTouchMove={onTouchMove}
        onScrollbarPointerDown={onScrollbarPointerDown}
        viewportClassName="assistant-message-viewport"
        viewportStyle={{
          paddingBottom: inputHeight + 28,
          scrollPaddingBottom: inputHeight + 28,
        }}
        onViewportScroll={onViewportScroll}
      >
        <div
          ref={contentRef}
          className="mx-auto flex min-h-full w-full max-w-3xl flex-col gap-4 px-4 py-5"
        >
          {runtime.messagesLoading && runtime.messages.length === 0 ? (
            <div className="flex items-center justify-center gap-2 py-12 text-xs text-muted-foreground">
              <Loader2 className="size-4 animate-spin motion-reduce:animate-none" /> 加载消息
            </div>
          ) : null}

          {!runtime.messagesLoading && !hasContent ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 py-20 text-center text-muted-foreground">
              <span className="flex size-11 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                <AssistantBrandIcon className="size-7" />
              </span>
              <p className="text-sm">告诉融光助手你想完成什么</p>
              <p className="max-w-xs text-xs text-muted-foreground/70">可以询问剧本、分镜或视频创作问题。</p>
            </div>
          ) : null}

          {segments.map((segment) => (
            <Fragment key={segment.key}>
              {segment.user ? <UserMessageBubble message={segment.user} /> : null}
              <AssistantTimelineContent
                timeline={segment.timeline}
              />
            </Fragment>
          ))}

          <AssistantTimelineContent
            timeline={liveTimeline}
            reasoningText={running ? runtime.pipeline.reasoningText : undefined}
            reasoningStartTime={running ? runtime.pipeline.reasoningStartTime : undefined}
            reasoningDurationMs={running ? runtime.pipeline.reasoningDurationMs : undefined}
            streaming={running}
            toolConfirmation={pendingConfirmation ? {
              toolCallIds: pendingConfirmation.toolCalls.map(
                (toolCall) => toolCall.toolCallId,
              ),
              parentToolCallId: pendingConfirmation.parentToolCallId,
              decisions: pendingConfirmation.decisions,
              submitting: pendingConfirmation.submitting,
              showActions: runtime.pipeline.status !== "cancelling",
              onDecision: (toolCallId, approved) =>
                void respondToToolConfirmation(toolCallId, approved),
            } : undefined}
          />

          {running && liveTimeline.length === 0 && !pendingConfirmation ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none" /> 正在思考…
            </div>
          ) : null}

          {error ? (
            <div className="flex items-start gap-2 rounded-xl border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              <span className="min-w-0 flex-1 break-words">{error}</span>
              <Button
                type="button"
                variant="destructive-ghost"
                size="xs"
                data-assistant-interactive="true"
                onClick={() => {
                  void loadMessages(conversationId);
                  ensureConnection();
                }}
              >
                <RefreshCw /> 重试
              </Button>
            </div>
          ) : null}
        </div>
      </OverlayScrollArea>

      <div
        aria-hidden={viewportReady}
        className={cn(
          "pointer-events-none absolute inset-0 z-20 flex items-center justify-center gap-2 text-xs text-muted-foreground transition-opacity duration-150 ease-in motion-reduce:transition-none",
          viewportReady ? "opacity-0" : "opacity-100",
        )}
      >
        <Loader2 className="size-4 animate-spin motion-reduce:animate-none" /> 加载消息
      </div>

      {showBackToBottom ? (
        <Button
          type="button"
          variant="secondary"
          size="sm"
          className="absolute left-1/2 z-10 -translate-x-1/2"
          style={{ bottom: inputHeight + 16 }}
          data-assistant-interactive="true"
          data-assistant-back-to-bottom
          onClick={scrollToBottom}
        >
          <ArrowDown /> 回到底部
        </Button>
      ) : null}
    </div>
  );
}
