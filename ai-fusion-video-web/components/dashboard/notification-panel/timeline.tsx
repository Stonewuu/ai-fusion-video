"use client";

import {
  memo,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type RefObject,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  AlertTriangle,
  Ban,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  Loader2,
  XCircle,
} from "lucide-react";
import {
  ToolCallCard,
} from "@/components/dashboard/shared/tool-call-card";
import { StreamMarkdown } from "@/components/dashboard/stream-markdown";
import { StreamThink } from "@/components/dashboard/stream-think";
import { cn } from "@/lib/utils";
import type {
  SubTimelineItem,
  TimelineItem,
} from "@/lib/store/pipeline-store";
import {
  getToolDisplayName,
  isSubAgentTool,
} from "./constants";
import { useSmartScroll } from "./hooks";
import {
  parseTaskContent,
  type TaskMediaLinkInfo,
} from "./utils";

// Preserve the full timeline while letting the browser skip layout and paint
// for rows far outside the scroll viewport.
const TIMELINE_ROW_STYLE: CSSProperties = {
  contentVisibility: "auto",
  containIntrinsicSize: "auto 96px",
};

const REASONING_TIMER_INTERVAL_MS = 100;

function useReasoningElapsedMs(
  startedAtMs: number | undefined,
  durationMs: number | undefined,
  active: boolean,
): number | undefined {
  const [now, setNow] = useState(() => Date.now());
  const running = active && durationMs === undefined && startedAtMs !== undefined;

  useEffect(() => {
    if (!running) return;

    const updateNow = () => setNow(Date.now());
    updateNow();
    const timer = window.setInterval(updateNow, REASONING_TIMER_INTERVAL_MS);
    window.addEventListener("focus", updateNow);
    document.addEventListener("visibilitychange", updateNow);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", updateNow);
      document.removeEventListener("visibilitychange", updateNow);
    };
  }, [running, startedAtMs]);

  if (durationMs !== undefined) return durationMs;
  if (!running || startedAtMs === undefined) return undefined;
  return Math.max(0, now - startedAtMs);
}

function ReasoningThink({
  content,
  startedAtMs,
  durationMs,
  streaming,
  compact,
  maxHeight,
}: {
  content: string;
  startedAtMs?: number;
  durationMs?: number;
  streaming: boolean;
  compact?: boolean;
  maxHeight?: number;
}) {
  const reasoningStreaming = streaming && durationMs === undefined;
  const displayedDurationMs = useReasoningElapsedMs(
    startedAtMs,
    durationMs,
    reasoningStreaming,
  );
  const title = displayedDurationMs !== undefined
    ? `思考 (${(displayedDurationMs / 1000).toFixed(1)}s)`
    : reasoningStreaming
      ? "思考中"
      : "思考";

  return (
    <StreamThink
      title={title}
      content={content}
      compact={compact}
      maxHeight={maxHeight}
      streaming={reasoningStreaming}
    />
  );
}

function TaskMediaLinks({ mediaLinks }: { mediaLinks: TaskMediaLinkInfo[] }) {
  if (mediaLinks.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      {mediaLinks.map((mediaLink, index) => (
        <div
          key={`${mediaLink.resolvedUrl}-${index}`}
          className="rounded-xl border border-border/30 bg-muted/20 px-3 py-3"
        >
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-muted-foreground">
                {mediaLink.label}
              </p>
              <a
                href={mediaLink.resolvedUrl}
                target="_blank"
                rel="noreferrer"
                className="mt-1 block break-all text-xs leading-relaxed text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
                title={mediaLink.resolvedUrl}
              >
                {mediaLink.resolvedUrl}
              </a>
            </div>
            <a
              href={mediaLink.resolvedUrl}
              target="_blank"
              rel="noreferrer"
              download
              className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg border border-primary/25 bg-primary/10 px-3 py-2 text-xs font-medium text-primary transition-colors hover:bg-primary/15"
            >
              <Download className="h-3.5 w-3.5" />
              下载视频
            </a>
          </div>
        </div>
      ))}
    </div>
  );
}

export interface TimelineToolConfirmation {
  toolCallIds: string[];
  parentToolCallId?: string;
  decisions: Readonly<Record<string, boolean>>;
  submitting: boolean;
  showActions: boolean;
  onDecision: (toolCallId: string, approved: boolean) => void;
}

function confirmationDecision(
  confirmation: TimelineToolConfirmation,
  toolCallId: string,
): "approve" | "deny" | undefined {
  if (!Object.prototype.hasOwnProperty.call(confirmation.decisions, toolCallId)) {
    return undefined;
  }
  return confirmation.decisions[toolCallId] ? "approve" : "deny";
}

const SubTimelineEntry = memo(function SubTimelineEntry({
  child,
  streaming,
}: {
  child: SubTimelineItem;
  streaming: boolean;
}) {
  let content;

  if (child.type === "reasoning") {
    content = (
      <ReasoningThink
        content={child.text}
        startedAtMs={child.startedAtMs}
        durationMs={child.durationMs}
        compact
        maxHeight={120}
        streaming={streaming}
      />
    );
  } else if (child.type === "tool") {
    content = (
      <ToolCallCard
        items={[{
          toolCallId: child.id,
          toolName: child.name,
          argumentsText: child.arguments,
          status: child.status,
          result: child.result,
        }]}
      />
    );
  } else {
    content = (
      <div className="text-xs leading-relaxed text-muted-foreground/80">
        <StreamMarkdown
          content={child.text}
          compact
          tone="muted"
          streaming={streaming}
        />
      </div>
    );
  }

  return <div style={TIMELINE_ROW_STYLE}>{content}</div>;
});

const SubAgentCard = memo(function SubAgentCard({
  item,
  toolConfirmation,
}: {
  item: Extract<TimelineItem, { type: "tool" }>;
  toolConfirmation?: TimelineToolConfirmation;
}) {
  const children = item.children ?? [];
  const isRunning = item.status === "calling";
  const [completedExpanded, setCompletedExpanded] = useState(false);
  const expanded = isRunning || completedExpanded;
  const lastContentChild = [...children]
    .reverse()
    .find(
      (
        child
      ): child is Extract<SubTimelineItem, { type: "content" }> =>
        child.type === "content"
    );
  const renderedResult =
    !isRunning && item.result
      ? lastContentChild?.text.trim() === item.result.trim()
        ? null
        : item.result
      : null;
  const hasResult = !!renderedResult;
  const hasContent = children.length > 0 || hasResult;
  const innerScrollRef = useSmartScroll([children], isRunning);
  const childConfirmationItems = toolConfirmation?.parentToolCallId === item.id
    ? toolConfirmation.toolCallIds.map((toolCallId) => {
        const matches = children.filter(
          (child): child is Extract<SubTimelineItem, { type: "tool" }> =>
            child.type === "tool" && child.id === toolCallId,
        );
        if (matches.length !== 1 || matches[0].status !== "awaiting_approval") {
          throw new Error(`Child confirmation tool is not awaiting approval: ${toolCallId}`);
        }
        return matches[0];
      })
    : [];
  const childConfirmationIds = new Set(
    childConfirmationItems.map((child) => child.id),
  );

  const toolCount = children.filter((child) => child.type === "tool").length;
  const doneToolCount = children.filter(
    (child) => child.type === "tool" && (
      child.status === "done"
      || child.status === "error"
      || child.status === "cancelled"
      || child.status === "rejected"
    )
  ).length;
  const activeToolCount = children.filter(
    (child) => child.type === "tool" && (
      child.status === "calling"
      || child.status === "awaiting_approval"
      || child.status === "approved"
    )
  ).length;
  const toolProgressLabel =
    toolCount > 0
      ? isRunning
        ? activeToolCount > 0
          ? `已完成 ${doneToolCount}/${toolCount}`
          : `已执行 ${toolCount} 步`
        : `${toolCount} 步`
      : null;

  return (
    <div
      className={cn(
        "rounded-xl text-sm border overflow-hidden",
        isRunning
          ? "border-purple-500/20 bg-purple-500/5"
          : item.status === "done"
            ? "border-green-500/20 bg-green-500/5"
            : item.status === "error"
              ? "border-destructive/20 bg-destructive/5"
              : "border-border/30 bg-muted/30"
      )}
    >
      <div
        className={cn(
          "flex items-center gap-2.5 px-4 py-2.5 transition-colors",
          !isRunning &&
            "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5"
        )}
        onClick={() => {
          if (!isRunning) {
            setCompletedExpanded((current) => !current);
          }
        }}
      >
        {isRunning ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-purple-400 shrink-0" />
        ) : item.status === "done" ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
        ) : item.status === "error" ? (
          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
        ) : (
          <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        <span className="font-medium text-xs">{getToolDisplayName(item.name)}</span>
        {toolProgressLabel && (
          <span className="text-[10px] text-muted-foreground/60 ml-1">
            {toolProgressLabel}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2 shrink-0">
          {isRunning && (
            <span className="text-xs text-purple-400/80 text-right">运行中…</span>
          )}
          <span>
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/50" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
            )}
          </span>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {expanded && hasContent && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div
              ref={innerScrollRef}
              className="border-t border-purple-500/10 px-4 py-3 space-y-2 max-h-[400px] overflow-y-auto"
            >
              {children.map((child, index) => {
                if (child.type === "tool" && childConfirmationIds.has(child.id)) {
                  if (!toolConfirmation) {
                    throw new Error("Child confirmation has no controls");
                  }
                  return (
                    <ToolCallCard
                      key={`sub-tool-confirmation-${child.id}`}
                      items={[{
                        toolCallId: child.id,
                        toolName: child.name,
                        argumentsText: child.arguments,
                        status: child.status,
                        result: child.result,
                      }]}
                      approval={{
                        decision: confirmationDecision(toolConfirmation, child.id),
                        submitting: toolConfirmation.submitting,
                        showActions: toolConfirmation.showActions,
                        onApprove: () => toolConfirmation.onDecision(child.id, true),
                        onReject: () => toolConfirmation.onDecision(child.id, false),
                      }}
                    />
                  );
                }
                return (
                  <SubTimelineEntry
                    key={
                      child.type === "tool"
                        ? `sub-tool-${child.id}`
                        : `sub-${child.type}-${index}`
                    }
                    child={child}
                    streaming={isRunning && index === children.length - 1}
                  />
                );
              })}

              {hasResult && (
                <div className="text-xs leading-relaxed text-foreground/70">
                  <StreamMarkdown content={renderedResult} compact />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

const TimelineEntry = memo(function TimelineEntry({
  item,
  previousItem,
  streaming,
  toolConfirmation,
  optimizeOffscreen = true,
}: {
  item: TimelineItem;
  previousItem: TimelineItem | null;
  streaming: boolean;
  toolConfirmation?: TimelineToolConfirmation;
  optimizeOffscreen?: boolean;
}) {
  let content;

  if (item.type === "reasoning") {
    content = (
      <ReasoningThink
        content={item.text}
        startedAtMs={item.startedAtMs}
        durationMs={item.durationMs}
        streaming={streaming}
      />
    );
  } else if (item.type === "tool") {
    if (isSubAgentTool(item.name) || (item.children && item.children.length > 0)) {
      content = <SubAgentCard item={item} toolConfirmation={toolConfirmation} />;
    } else {
      content = (
        <ToolCallCard
          items={[{
            toolCallId: item.id,
            toolName: item.name,
            argumentsText: item.arguments,
            status: item.status,
            result: item.result,
          }]}
        />
      );
    }
  } else {
    if (
      previousItem?.type === "tool" &&
      (isSubAgentTool(previousItem.name) ||
        (previousItem.children && previousItem.children.length > 0)) &&
      previousItem.result &&
      item.text.trim() === previousItem.result.trim()
    ) {
      return null;
    }

    const { markdownContent, mediaLinks } = parseTaskContent(item.text);
    content = (
      <div className="space-y-3 text-sm leading-relaxed">
        {markdownContent ? (
          <StreamMarkdown content={markdownContent} streaming={streaming} />
        ) : null}
        <TaskMediaLinks mediaLinks={mediaLinks} />
      </div>
    );
  }

  return <div style={optimizeOffscreen ? TIMELINE_ROW_STYLE : undefined}>{content}</div>;
});

export interface MessageTimelineProps {
  reasoningText?: string;
  reasoningStartTime?: number;
  reasoningDurationMs?: number;
  timeline: TimelineItem[];
  scrollRef: RefObject<HTMLDivElement | null>;
  initialScrollToEnd?: boolean;
  streaming?: boolean;
  error?: string;
  toolConfirmation?: TimelineToolConfirmation;
}

type MessageTimelineRow =
  | {
      key: string;
      type: "fallback-reasoning";
      content: string;
      startedAtMs?: number;
      durationMs?: number;
      streaming: boolean;
    }
  | {
      key: string;
      type: "tool-confirmation";
      item: Extract<TimelineItem, { type: "tool" }>;
    }
  | {
      key: string;
      type: "entry";
      item: TimelineItem;
      previousItem: TimelineItem | null;
      streaming: boolean;
    }
  | { key: string; type: "error"; message: string };

function MessageTimelineRowContent({
  row,
  optimizeOffscreen,
  toolConfirmation,
}: {
  row: MessageTimelineRow;
  optimizeOffscreen: boolean;
  toolConfirmation?: TimelineToolConfirmation;
}) {
  if (row.type === "fallback-reasoning") {
    return (
      <ReasoningThink
        content={row.content}
        startedAtMs={row.startedAtMs}
        durationMs={row.durationMs}
        streaming={row.streaming}
      />
    );
  }

  if (row.type === "error") {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-sm text-destructive">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <span className="leading-relaxed">{row.message}</span>
      </div>
    );
  }

  if (row.type === "tool-confirmation") {
    if (!toolConfirmation || toolConfirmation.parentToolCallId) {
      throw new Error("Root tool confirmation row has no matching confirmation batch");
    }
    return (
      <ToolCallCard
        items={[{
          toolCallId: row.item.id,
          toolName: row.item.name,
          argumentsText: row.item.arguments,
          status: row.item.status,
          result: row.item.result,
        }]}
        approval={{
          decision: confirmationDecision(toolConfirmation, row.item.id),
          submitting: toolConfirmation.submitting,
          showActions: toolConfirmation.showActions,
          onApprove: () => toolConfirmation.onDecision(row.item.id, true),
          onReject: () => toolConfirmation.onDecision(row.item.id, false),
        }}
      />
    );
  }

  return (
    <TimelineEntry
      item={row.item}
      previousItem={row.previousItem}
      streaming={row.streaming}
      toolConfirmation={toolConfirmation}
      optimizeOffscreen={optimizeOffscreen}
    />
  );
}

function VirtualizedMessageTimeline({
  rows,
  scrollRef,
  toolConfirmation,
}: {
  rows: MessageTimelineRow[];
  scrollRef: RefObject<HTMLDivElement | null>;
  toolConfirmation?: TimelineToolConfirmation;
}) {
  // TanStack Virtual owns a mutable measurement instance; this component is
  // intentionally excluded from React Compiler memoization.
  // eslint-disable-next-line react-hooks/incompatible-library
  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (index) => {
      const row = rows[index];
      if (row?.type === "fallback-reasoning") return 192;
      if (row?.type === "error") return 48;
      if (row?.type === "entry" && row.item.type === "reasoning") return 192;
      return 72;
    },
    getItemKey: (index) => rows[index]?.key ?? index,
    gap: 12,
    overscan: 6,
  });
  const initialScrollCompletedRef = useRef(false);

  useEffect(() => {
    if (initialScrollCompletedRef.current || rows.length === 0) return;

    let secondFrameId: number | undefined;
    const firstFrameId = requestAnimationFrame(() => {
      initialScrollCompletedRef.current = true;
      rowVirtualizer.scrollToIndex(rows.length - 1, { align: "end" });
      // Run once more after the last variable-height row has been measured.
      secondFrameId = requestAnimationFrame(() => {
        rowVirtualizer.scrollToIndex(rows.length - 1, { align: "end" });
      });
    });

    return () => {
      cancelAnimationFrame(firstFrameId);
      if (secondFrameId !== undefined) cancelAnimationFrame(secondFrameId);
    };
  }, [rowVirtualizer, rows.length]);

  return (
    <div
      className="relative w-full"
      style={{ height: rowVirtualizer.getTotalSize() }}
    >
      {rowVirtualizer.getVirtualItems().map((virtualRow) => {
        const row = rows[virtualRow.index];
        if (!row) return null;

        return (
          <div
            key={virtualRow.key}
            ref={rowVirtualizer.measureElement}
            data-index={virtualRow.index}
            className="absolute left-0 top-0 w-full"
            style={{ transform: `translateY(${virtualRow.start}px)` }}
          >
            <MessageTimelineRowContent
              row={row}
              optimizeOffscreen
              toolConfirmation={toolConfirmation}
            />
          </div>
        );
      })}
    </div>
  );
}

export function MessageTimeline({
  reasoningText,
  reasoningStartTime,
  reasoningDurationMs,
  timeline,
  scrollRef,
  initialScrollToEnd = false,
  streaming,
  error,
  toolConfirmation,
}: MessageTimelineProps) {
  const hasTimelineReasoning = timeline.some((item) => item.type === "reasoning");
  const fallbackReasoningStreaming =
    !!streaming && reasoningDurationMs === undefined;

  const rows = useMemo<MessageTimelineRow[]>(() => {
    const nextRows: MessageTimelineRow[] = [];

    if (!hasTimelineReasoning && reasoningText) {
      nextRows.push({
        key: "fallback-reasoning",
        type: "fallback-reasoning",
        content: reasoningText,
        startedAtMs: reasoningStartTime,
        durationMs: reasoningDurationMs,
        streaming: fallbackReasoningStreaming,
      });
    }

    let rootConfirmationItems: Array<Extract<TimelineItem, { type: "tool" }>> = [];
    if (toolConfirmation && !toolConfirmation.parentToolCallId) {
      rootConfirmationItems = toolConfirmation.toolCallIds.map((toolCallId) => {
        const matches = timeline.filter(
          (item): item is Extract<TimelineItem, { type: "tool" }> =>
            item.type === "tool" && item.id === toolCallId,
        );
        if (matches.length !== 1 || matches[0].status !== "awaiting_approval") {
          throw new Error(`Confirmation batch tool is not awaiting approval: ${toolCallId}`);
        }
        return matches[0];
      });
    }
    const rootConfirmationIds = new Set(rootConfirmationItems.map((item) => item.id));

    timeline.forEach((item, index) => {
      if (item.type === "tool" && rootConfirmationIds.has(item.id)) {
        nextRows.push({
          key: `tool-confirmation-${item.id}`,
          type: "tool-confirmation",
          item,
        });
        return;
      }
      nextRows.push({
        key: item.type === "tool" ? `tool-${item.id}` : `${item.type}-${index}`,
        type: "entry",
        item,
        previousItem: index > 0 ? timeline[index - 1] : null,
        streaming: !!streaming && index === timeline.length - 1,
      });
    });

    if (error) {
      nextRows.push({ key: "error", type: "error", message: error });
    }

    return nextRows;
  }, [
    error,
    hasTimelineReasoning,
    fallbackReasoningStreaming,
    reasoningDurationMs,
    reasoningStartTime,
    reasoningText,
    streaming,
    timeline,
    toolConfirmation,
  ]);

  // Assistant conversations render several independent timelines inside one
  // shared viewport. Flow layout avoids applying viewport-relative virtual
  // offsets and attaching several virtualizers to the same scroll element.
  if (!initialScrollToEnd) {
    return (
      <div className="space-y-3">
        {rows.map((row) => (
          <MessageTimelineRowContent
            key={row.key}
            row={row}
            optimizeOffscreen={false}
            toolConfirmation={toolConfirmation}
          />
        ))}
      </div>
    );
  }

  return (
    <VirtualizedMessageTimeline
      rows={rows}
      scrollRef={scrollRef}
      toolConfirmation={toolConfirmation}
    />
  );
}
