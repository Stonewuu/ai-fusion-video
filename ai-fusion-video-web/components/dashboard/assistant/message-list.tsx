"use client";

import {
  Fragment,
  useMemo,
  type RefObject,
} from "react";
import { ArrowDown, Bot, Loader2, RefreshCw, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { MessageTimeline } from "@/components/dashboard/notification-panel/timeline";
import { messagesToTimeline } from "@/components/dashboard/notification-panel/history";
import type { AgentMessage } from "@/lib/api/ai-assistant";
import type { TimelineItem } from "@/lib/store/pipeline-store";
import { useAssistantStore, type AssistantConversationRuntime } from "@/lib/store/assistant-store";
import { useAssistantMessageScroll } from "./use-assistant-message-scroll";

interface AssistantMessageListProps {
  conversationId: string;
  inputHeight: number;
}

interface MessageSegment {
  key: string;
  user?: AgentMessage;
  assistant: AgentMessage[];
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

function UserBubble({ message }: { message: AgentMessage }) {
  return (
    <div className="flex justify-end gap-2">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2.5 text-sm leading-relaxed text-primary-foreground shadow-sm">
        <p className="whitespace-pre-wrap break-words">{message.content}</p>
      </div>
      <span
        className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground"
        aria-hidden="true"
      >
        <User className="size-3.5" />
      </span>
    </div>
  );
}

function AssistantTimelineBubble({
  timeline,
  scrollRef,
  runtime,
  streaming = false,
}: {
  timeline: TimelineItem[];
  scrollRef: RefObject<HTMLDivElement | null>;
  runtime: AssistantConversationRuntime;
  streaming?: boolean;
}) {
  if (timeline.length === 0) return null;
  return (
    <div className="flex items-start gap-2">
      <span
        className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary"
        aria-hidden="true"
      >
        <Bot className="size-3.5" />
      </span>
      <div className="min-w-0 flex-1 rounded-2xl rounded-tl-md border border-border/20 bg-card/50 px-3.5 py-3">
        <MessageTimeline
          reasoningText={streaming ? runtime.pipeline.reasoningText : undefined}
          reasoningStartTime={streaming ? runtime.pipeline.reasoningStartTime : undefined}
          reasoningDurationMs={streaming ? runtime.pipeline.reasoningDurationMs : undefined}
          timeline={timeline}
          scrollRef={scrollRef}
          initialScrollToEnd={false}
          streaming={streaming}
          error={undefined}
        />
      </div>
    </div>
  );
}

export function AssistantMessageList({
  conversationId,
  inputHeight,
}: AssistantMessageListProps) {
  const runtime = useAssistantStore((state) => state.conversationStates[conversationId]);
  const loadMessages = useAssistantStore((state) => state.loadMessagesIfNeeded);
  const ensureConnection = useAssistantStore((state) => state.ensureContentConnection);

  const running = !!runtime && isRunning(runtime);
  const contentReady = !!runtime && (runtime.messagesLoaded || !!runtime.messagesError);
  const showPipelineTimeline = !!runtime && (running || !runtime.messagesLoaded);
  const activeRunId = showPipelineTimeline
    ? runtime?.pipeline.runId ?? runtime?.knownRunId
    : undefined;
  const messages = runtime?.messages;
  const segments = useMemo(
    () => buildSegments(messages ?? [], activeRunId),
    [activeRunId, messages],
  );
  const liveTimeline = runtime && showPipelineTimeline
    ? runtime.pipeline.timeline
    : EMPTY_TIMELINE;
  const error = runtime?.messagesError || runtime?.pipeline.error || runtime?.connectionError;

  const {
    viewportRef,
    contentRef,
    viewportReady,
    showBackToBottom,
    onViewportScroll,
    onWheel,
    onTouchStart,
    onTouchMove,
    scrollToBottom,
  } = useAssistantMessageScroll({
    contentReady,
    running,
    inputHeight,
  });

  if (!runtime) return null;

  const hasContent = segments.length > 0 || liveTimeline.length > 0 || !!error;

  return (
    <div className="relative min-h-0 flex-1">
      <OverlayScrollArea
        className="h-full"
        viewportRef={viewportRef}
        onViewportWheel={onWheel}
        onViewportTouchStart={onTouchStart}
        onViewportTouchMove={onTouchMove}
        viewportClassName="assistant-message-viewport"
        viewportStyle={{
          paddingBottom: inputHeight + 28,
          scrollPaddingBottom: inputHeight + 28,
          visibility: viewportReady ? "visible" : "hidden",
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
                <Bot className="size-5" />
              </span>
              <p className="text-sm">告诉融光助手你想完成什么</p>
              <p className="max-w-xs text-xs text-muted-foreground/70">可以询问剧本、分镜或视频创作问题。</p>
            </div>
          ) : null}

          {segments.map((segment) => {
            const timeline = messagesToTimeline(segment.assistant);
            return (
              <Fragment key={segment.key}>
                {segment.user ? <UserBubble message={segment.user} /> : null}
                <AssistantTimelineBubble
                  timeline={timeline}
                  scrollRef={viewportRef}
                  runtime={runtime}
                />
              </Fragment>
            );
          })}

          <AssistantTimelineBubble
            timeline={liveTimeline}
            scrollRef={viewportRef}
            runtime={runtime}
            streaming={running}
          />

          {running && liveTimeline.length === 0 ? (
            <div className="flex items-center gap-2 pl-8 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none" /> 正在思考…
            </div>
          ) : null}

          {error ? (
            <div className="ml-8 flex items-start gap-2 rounded-xl border border-destructive/20 bg-destructive/5 px-3 py-2 text-xs text-destructive">
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

      {showBackToBottom ? (
        <Button
          type="button"
          variant="secondary"
          size="sm"
          className="absolute left-1/2 z-10 -translate-x-1/2"
          style={{ bottom: inputHeight + 16 }}
          data-assistant-interactive="true"
          onClick={scrollToBottom}
        >
          <ArrowDown /> 回到底部
        </Button>
      ) : null}
    </div>
  );
}
