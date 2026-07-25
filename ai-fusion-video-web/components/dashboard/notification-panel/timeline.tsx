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
  Wrench,
  XCircle,
} from "lucide-react";
import { StreamMarkdown } from "@/components/dashboard/stream-markdown";
import { StreamThink } from "@/components/dashboard/stream-think";
import { cn } from "@/lib/utils";
import type {
  SubTimelineItem,
  TimelineItem,
  ToolTimelineStatus,
} from "@/lib/store/pipeline-store";
import {
  getToolDisplayName,
  isSubAgentTool,
} from "./constants";
import { useSmartScroll } from "./hooks";
import { ToolResultDisplay } from "./results";
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

const ExpandableToolCard = memo(function ExpandableToolCard({
  toolName,
  toolStatus,
  result,
}: {
  toolName: string;
  toolStatus: Exclude<ToolTimelineStatus, "calling">;
  result?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasResult = !!result;

  return (
    <div
      className={cn(
        "rounded-xl text-sm border overflow-hidden",
        toolStatus === "done"
          ? "border-green-500/20 bg-green-500/5"
          : toolStatus === "error"
            ? "border-destructive/20 bg-destructive/5"
            : "border-border/30 bg-muted/30"
      )}
    >
      <div
        className={cn(
          "flex items-center gap-3 px-4 py-2.5",
          hasResult &&
            "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
        )}
        onClick={() => hasResult && setExpanded(!expanded)}
      >
        {toolStatus === "done" ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
        ) : toolStatus === "error" ? (
          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
        ) : (
          <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        {isSubAgentTool(toolName) ? (
          <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        ) : (
          <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="font-medium text-xs">{getToolDisplayName(toolName)}</span>
        <span className="flex items-center gap-1 text-xs text-muted-foreground ml-auto">
          {toolStatus === "done" ? "已完成" : toolStatus === "error" ? "失败" : "已取消"}
          {hasResult &&
            (expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/50" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
            ))}
        </span>
      </div>

      {hasResult && (
        <AnimatePresence initial={false}>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div
                className={cn(
                  "border-t px-4 py-3",
                  toolStatus === "error"
                    ? "border-destructive/10"
                    : toolStatus === "done"
                      ? "border-green-500/10"
                      : "border-border/20"
                )}
              >
                <ToolResultDisplay toolName={toolName} content={result} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      )}
    </div>
  );
});

const CallingToolCard = memo(function CallingToolCard({
  toolName,
}: {
  toolName: string;
}) {
  return (
    <div className="rounded-xl text-sm border overflow-hidden border-blue-500/20 bg-blue-500/5">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />
        {isSubAgentTool(toolName) ? (
          <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        ) : (
          <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="font-medium text-xs">{getToolDisplayName(toolName)}</span>
        <span className="text-xs text-blue-400/80 ml-auto">调用中…</span>
      </div>
    </div>
  );
});

const SubTimelineEntry = memo(function SubTimelineEntry({
  child,
  streaming,
}: {
  child: SubTimelineItem;
  streaming: boolean;
}) {
  let content;

  if (child.type === "reasoning") {
    const reasoningStreaming = streaming && child.durationMs === undefined;
    const title = child.durationMs !== undefined
      ? `思考 (${(child.durationMs / 1000).toFixed(1)}s)`
      : reasoningStreaming
        ? "思考中"
        : "思考";
    content = (
      <StreamThink
        title={title}
        content={child.text}
        compact
        maxHeight={120}
        streaming={reasoningStreaming}
      />
    );
  } else if (child.type === "tool") {
    content =
      child.status === "calling" ? (
        <CallingToolCard toolName={child.name} />
      ) : (
        <ExpandableToolCard
          toolName={child.name}
          toolStatus={child.status}
          result={child.result}
        />
      );
  } else {
    content = (
      <div className="text-xs leading-relaxed text-muted-foreground/80">
        <StreamMarkdown content={child.text} compact tone="muted" />
      </div>
    );
  }

  return <div style={TIMELINE_ROW_STYLE}>{content}</div>;
});

const SubAgentCard = memo(function SubAgentCard({
  item,
}: {
  item: Extract<TimelineItem, { type: "tool" }>;
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

  const toolCount = children.filter((child) => child.type === "tool").length;
  const doneToolCount = children.filter(
    (child) => child.type === "tool" && child.status !== "calling"
  ).length;
  const activeToolCount = children.filter(
    (child) => child.type === "tool" && child.status === "calling"
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
              {children.map((child, index) => (
                <SubTimelineEntry
                  key={
                    child.type === "tool"
                      ? `sub-tool-${child.id}`
                      : `sub-${child.type}-${index}`
                  }
                  child={child}
                  streaming={isRunning && index === children.length - 1}
                />
              ))}

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
  optimizeOffscreen = true,
}: {
  item: TimelineItem;
  previousItem: TimelineItem | null;
  streaming: boolean;
  optimizeOffscreen?: boolean;
}) {
  let content;

  if (item.type === "reasoning") {
    const reasoningStreaming = streaming && item.durationMs === undefined;
    const title = item.durationMs !== undefined
      ? `思考 (${(item.durationMs / 1000).toFixed(1)}s)`
      : reasoningStreaming
        ? "思考中"
        : "思考";
    content = (
      <StreamThink
        title={title}
        content={item.text}
        streaming={reasoningStreaming}
      />
    );
  } else if (item.type === "tool") {
    if (isSubAgentTool(item.name) || (item.children && item.children.length > 0)) {
      content = <SubAgentCard item={item} />;
    } else if (item.status === "calling") {
      content = <CallingToolCard toolName={item.name} />;
    } else {
      content = (
        <ExpandableToolCard
          toolName={item.name}
          toolStatus={item.status}
          result={item.result}
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
        {markdownContent ? <StreamMarkdown content={markdownContent} /> : null}
        <TaskMediaLinks mediaLinks={mediaLinks} />
      </div>
    );
  }

  return <div style={optimizeOffscreen ? TIMELINE_ROW_STYLE : undefined}>{content}</div>;
});

export interface MessageTimelineProps {
  reasoningText?: string;
  reasoningDurationMs?: number;
  timeline: TimelineItem[];
  scrollRef: RefObject<HTMLDivElement | null>;
  initialScrollToEnd?: boolean;
  streaming?: boolean;
  error?: string;
}

type MessageTimelineRow =
  | {
      key: string;
      type: "fallback-reasoning";
      title: string;
      content: string;
      streaming: boolean;
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
}: {
  row: MessageTimelineRow;
  optimizeOffscreen: boolean;
}) {
  if (row.type === "fallback-reasoning") {
    return (
      <StreamThink
        title={row.title}
        content={row.content}
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

  return (
    <TimelineEntry
      item={row.item}
      previousItem={row.previousItem}
      streaming={row.streaming}
      optimizeOffscreen={optimizeOffscreen}
    />
  );
}

function VirtualizedMessageTimeline({
  rows,
  scrollRef,
}: {
  rows: MessageTimelineRow[];
  scrollRef: RefObject<HTMLDivElement | null>;
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
            <MessageTimelineRowContent row={row} optimizeOffscreen />
          </div>
        );
      })}
    </div>
  );
}

export function MessageTimeline({
  reasoningText,
  reasoningDurationMs,
  timeline,
  scrollRef,
  initialScrollToEnd = false,
  streaming,
  error,
}: MessageTimelineProps) {
  const hasTimelineReasoning = timeline.some((item) => item.type === "reasoning");
  const fallbackReasoningStreaming =
    !!streaming && reasoningDurationMs === undefined;
  const reasoningTitle = reasoningDurationMs !== undefined
    ? `思考 (${(reasoningDurationMs / 1000).toFixed(1)}s)`
    : fallbackReasoningStreaming
      ? "思考中"
      : "思考";

  const rows = useMemo<MessageTimelineRow[]>(() => {
    const nextRows: MessageTimelineRow[] = [];

    if (!hasTimelineReasoning && reasoningText) {
      nextRows.push({
        key: "fallback-reasoning",
        type: "fallback-reasoning",
        title: reasoningTitle,
        content: reasoningText,
        streaming: fallbackReasoningStreaming,
      });
    }

    timeline.forEach((item, index) => {
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
    reasoningText,
    reasoningTitle,
    streaming,
    timeline,
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
          />
        ))}
      </div>
    );
  }

  return <VirtualizedMessageTimeline rows={rows} scrollRef={scrollRef} />;
}
