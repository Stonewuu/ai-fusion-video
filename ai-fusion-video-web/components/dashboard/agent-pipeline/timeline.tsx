"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Ban,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Loader2,
  Wrench,
  XCircle,
} from "lucide-react";
import { StreamMarkdown } from "@/components/dashboard/stream-markdown";
import { StreamThink } from "@/components/dashboard/stream-think";
import { cn } from "@/lib/utils";
import {
  getToolDisplayName,
  isSubAgentTool,
} from "../shared/ai-task-display";
import { ToolResultDisplay } from "./results";
import type { SubTimelineItem, TimelineItem } from "./types";

function SubTimelineToolItem({
  child,
}: {
  child: Extract<SubTimelineItem, { type: "tool" }>;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border text-xs px-3 py-2 flex items-center gap-2",
        child.status === "calling" && "border-blue-500/20 bg-blue-500/5",
        child.status === "done" && "border-green-500/20 bg-green-500/5",
        child.status === "error" && "border-destructive/20 bg-destructive/5",
        child.status === "cancelled" && "border-border/30 bg-muted/30"
      )}
    >
      {child.status === "calling" ? (
        <Loader2 className="h-3 w-3 animate-spin text-blue-400 shrink-0" />
      ) : child.status === "done" ? (
        <CheckCircle2 className="h-3 w-3 text-green-400 shrink-0" />
      ) : child.status === "error" ? (
        <XCircle className="h-3 w-3 text-destructive shrink-0" />
      ) : (
        <Ban className="h-3 w-3 text-muted-foreground shrink-0" />
      )}
      {isSubAgentTool(child.name) ? (
        <Bot className="h-3 w-3 text-purple-400 shrink-0" />
      ) : (
        <Wrench className="h-3 w-3 text-muted-foreground shrink-0" />
      )}
      <span className="font-medium">{getToolDisplayName(child.name)}</span>
      <span className="ml-auto text-muted-foreground/60">
        {child.status === "calling"
          ? "执行中..."
          : child.status === "done"
            ? "已完成"
            : child.status === "error"
              ? "失败"
              : "已取消"}
      </span>
    </div>
  );
}

function ToolTimelineItem({
  item,
  isExpanded,
  onToggle,
}: {
  item: Extract<TimelineItem, { type: "tool" }>;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const hasResult =
    item.status !== "calling" && item.result;
  const hasChildren = !!item.children?.length;
  const canExpand = hasResult || hasChildren;

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      className={cn(
        "rounded-xl text-sm border overflow-hidden",
        item.status === "calling" && "border-blue-500/20 bg-blue-500/5",
        item.status === "done" && "border-green-500/20 bg-green-500/5",
        item.status === "error" && "border-destructive/20 bg-destructive/5",
        item.status === "cancelled" && "border-border/30 bg-muted/30"
      )}
    >
      <div
        className={cn(
          "flex items-center gap-3 px-4 py-2.5",
          canExpand && "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
        )}
        onClick={() => canExpand && onToggle()}
      >
        {item.status === "calling" ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />
        ) : item.status === "done" ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
        ) : item.status === "error" ? (
          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
        ) : (
          <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        {item.agentName || isSubAgentTool(item.name) ? (
          <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        ) : (
          <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="font-medium text-xs">{getToolDisplayName(item.name)}</span>
        {item.status === "calling" && (
          <span className="text-xs text-muted-foreground ml-auto">执行中...</span>
        )}
        {item.status === "done" && (
          <span className="flex items-center gap-1.5 text-xs text-green-400/80 ml-auto">
            ✓ 完成
            {canExpand &&
              (isExpanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              ))}
          </span>
        )}
        {item.status === "error" && (
          <span className="flex items-center gap-1.5 text-xs text-destructive ml-auto">
            ✗ 失败
            {canExpand &&
              (isExpanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              ))}
          </span>
        )}
        {item.status === "cancelled" && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground ml-auto">
            已取消
            {canExpand &&
              (isExpanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              ))}
          </span>
        )}
      </div>

      <AnimatePresence>
        {isExpanded && (hasResult || hasChildren) && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div
              className={cn(
                "border-t px-4 py-3 space-y-2",
                item.status === "error"
                  ? "border-destructive/10"
                  : item.status === "done"
                    ? "border-green-500/10"
                    : "border-border/20"
              )}
            >
              {item.children && item.children.length > 0 && (
                <div className="space-y-2 pl-2 border-l-2 border-purple-500/20">
                  {item.children.map((child, childIndex) => {
                    const childCount = item.children?.length ?? 0;
                    if (child.type === "reasoning") {
                      const childIsStreaming =
                        item.status === "calling" &&
                        childIndex === childCount - 1 &&
                        child.durationMs === undefined;
                      const childTitle = child.durationMs !== undefined
                        ? `子Agent 思考 (${(child.durationMs / 1000).toFixed(1)}s)`
                        : childIsStreaming
                          ? "子Agent 思考中"
                          : "子Agent 思考";
                      return (
                        <div key={`sub-reasoning-${childIndex}`} className="text-xs">
                          <StreamThink
                            title={childTitle}
                            content={child.text}
                            compact
                            maxHeight={120}
                            streaming={childIsStreaming}
                          />
                        </div>
                      );
                    }

                    if (child.type === "tool") {
                      return (
                        <SubTimelineToolItem
                          key={`sub-tool-${child.id}`}
                          child={child}
                        />
                      );
                    }

                    return (
                      <div
                        key={`sub-content-${childIndex}`}
                        className="rounded-lg border border-border/20 bg-card/20 p-3 text-xs leading-relaxed"
                      >
                        <StreamMarkdown content={child.text} compact />
                      </div>
                    );
                  })}
                </div>
              )}
              {hasResult && (
                <ToolResultDisplay toolName={item.name} result={item.result!} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function AgentPipelineTimeline({
  timeline,
  isActive,
}: {
  timeline: TimelineItem[];
  isActive: boolean;
}) {
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [timeline]);

  if (timeline.length === 0) {
    return null;
  }

  return (
    <div ref={scrollRef} className="space-y-2 max-h-[60vh] overflow-y-auto">
      {timeline.map((item, index) => {
        if (item.type === "reasoning") {
          const reasoningStreaming =
            isActive &&
            index === timeline.length - 1 &&
            item.durationMs === undefined;
          const title = item.durationMs !== undefined
            ? `思考 (${(item.durationMs / 1000).toFixed(1)}s)`
            : reasoningStreaming
              ? "思考中"
              : "思考";
          return (
            <motion.div
              key={`reasoning-${index}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <StreamThink
                title={title}
                content={item.text}
                streaming={reasoningStreaming}
              />
            </motion.div>
          );
        }

        if (item.type === "tool") {
          const isExpanded = expandedTools.has(item.id);
          return (
            <ToolTimelineItem
              key={`tool-${item.id}`}
              item={item}
              isExpanded={isExpanded}
              onToggle={() => {
                setExpandedTools((prev) => {
                  const next = new Set(prev);
                  if (next.has(item.id)) {
                    next.delete(item.id);
                  } else {
                    next.add(item.id);
                  }
                  return next;
                });
              }}
            />
          );
        }

        const prevItem = index > 0 ? timeline[index - 1] : null;
        if (
          prevItem?.type === "tool" &&
          prevItem.children &&
          prevItem.children.length > 0 &&
          prevItem.result &&
          item.text.trim() === prevItem.result.trim()
        ) {
          return null;
        }

        return (
          <motion.div
            key={`content-${index}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={cn(
              "rounded-xl border border-border/30 bg-card/30 p-4",
              "text-sm leading-relaxed"
            )}
          >
            <StreamMarkdown
              content={item.text}
              streaming={isActive && index === timeline.length - 1}
            />
          </motion.div>
        );
      })}
    </div>
  );
}
