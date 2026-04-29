"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Wrench,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Bot,
  Ban,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { XMarkdown } from "@ant-design/x-markdown";
import { cn } from "@/lib/utils";
import { ToolResultDisplay } from "@/components/dashboard/agent-pipeline-results";
import { StreamThink } from "@/components/dashboard/stream-think";
import {
  pipelineStream,
  cancelPipeline,
  type AiChatReq,
  type AiChatStreamEvent,
} from "@/lib/api/ai-pipeline";

// ========== 类型 ==========

/** 子 Agent 时间线元素 */
type SubTimelineItem =
  | { type: "tool"; id: string; name: string; arguments: string; status: "calling" | "done" | "error"; result?: string }
  | { type: "content"; text: string }
  | { type: "reasoning"; text: string; durationMs?: number };

/** 时间线中的每个元素 */
type TimelineItem =
  | { type: "tool"; id: string; name: string; arguments: string; status: "calling" | "done" | "error"; result?: string; agentName?: string; children?: SubTimelineItem[] }
  | { type: "reasoning"; text: string; durationMs?: number }
  | { type: "content"; text: string };

interface PipelineState {
  status: "idle" | "reasoning" | "running" | "done" | "error" | "cancelled";
  reasoningText: string;
  reasoningDurationMs?: number;
  /** 按时间顺序排列的事件流 */
  timeline: TimelineItem[];
  conversationId?: string;
  error?: string;
}

interface AgentPipelineProps {
  /** 发送给后端的请求配置 */
  request: AiChatReq;
  /** 是否自动开始 */
  autoStart?: boolean;
  /** 完成回调 */
  onComplete?: (conversationId?: string) => void;
  /** 错误回调 */
  onError?: (error: string) => void;
}

function appendReasoningToSubTimeline(
  children: SubTimelineItem[],
  reasoningContent: string
): SubTimelineItem[] {
  const last = children[children.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...children.slice(0, -1),
      { ...last, text: last.text + reasoningContent },
    ];
  }
  return [...children, { type: "reasoning", text: reasoningContent }];
}

function updateLastSubTimelineReasoningDuration(
  children: SubTimelineItem[],
  durationMs: number
): SubTimelineItem[] {
  for (let index = children.length - 1; index >= 0; index--) {
    const item = children[index];
    if (item.type === "reasoning") {
      return children.map((child, childIndex) =>
        childIndex === index && child.type === "reasoning"
          ? { ...child, durationMs }
          : child
      );
    }
  }
  return children;
}

function appendReasoningToTimeline(
  timeline: TimelineItem[],
  reasoningContent: string
): TimelineItem[] {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "reasoning") {
    return [
      ...timeline.slice(0, -1),
      { ...last, text: last.text + reasoningContent },
    ];
  }
  return [...timeline, { type: "reasoning", text: reasoningContent }];
}

function updateLastTimelineReasoningDuration(
  timeline: TimelineItem[],
  durationMs: number
): TimelineItem[] {
  for (let index = timeline.length - 1; index >= 0; index--) {
    const item = timeline[index];
    if (item.type === "reasoning") {
      return timeline.map((timelineItem, timelineIndex) =>
        timelineIndex === index && timelineItem.type === "reasoning"
          ? { ...timelineItem, durationMs }
          : timelineItem
      );
    }
  }
  return timeline;
}

// ========== 工具名中文映射 ==========

const toolDisplayNames: Record<string, string> = {
  // ── 项目 ──
  list_my_projects: "查询我的项目",
  get_project: "查询项目信息",

  // ── 资产 ──
  list_project_assets: "查询项目资产",
  batch_create_assets: "批量创建资产",
  create_asset: "创建资产",
  get_asset: "查询资产详情",
  update_asset: "更新资产",
  query_asset_items: "查询子资产列表",
  batch_create_asset_items: "批量创建子资产",
  add_asset_item: "添加子资产图片",
  update_asset_image: "更新子资产图片",
  query_asset_metadata: "查询资产属性定义",

  // ── 剧本 ──
  get_project_script: "查询项目剧本",
  get_script: "查询剧本详情",
  get_script_structure: "查询剧本结构",
  update_script: "更新剧本",
  update_script_info: "更新剧本信息",
  save_script_episode: "保存分集",
  save_script_scene_items: "保存场次",
  get_script_episode: "查询分集详情",
  get_script_scene: "查询场次详情",
  manage_script_scenes: "管理剧本场次",
  update_script_scene: "更新剧本场次",

  // ── 分镜 ──
  list_project_storyboards: "查询项目分镜列表",
  get_storyboard: "查询分镜详情",
  insert_storyboard_item: "插入分镜条目",
  save_storyboard_episode: "保存分镜分集",
  save_storyboard_scene_shots: "保存分镜场次镜头",

  // ── 生图 ──
  get_generation_model_capabilities: "查询生成模型能力",
  generate_image: "AI 生成图片",

  // ── 子 Agent ──
  episode_scene_writer: "分集场次解析（子Agent）",
  episode_script_creator: "分集剧本创作（子Agent）",
  episode_storyboard_writer: "分集分镜编写（子Agent）",
  storyboard_asset_preprocessor: "子资产预处理（子Agent）",
  generate_asset_image: "生成资产图片（子Agent）",
};

function getToolDisplayName(name: string) {
  return toolDisplayNames[name] || name;
}

// ========== 主组件 ==========

export function AgentPipeline({
  request,
  autoStart = false,
  onComplete,
  onError,
}: AgentPipelineProps) {
  const [state, setState] = useState<PipelineState>({
    status: "idle",
    reasoningText: "",
    timeline: [],
  });
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const startedRef = useRef(false);

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [state.timeline]);

  /** 在 timeline 中找到指定 tool ID 的节点并更新其 children */
  const appendToToolChildren = useCallback(
    (
      timeline: TimelineItem[],
      parentToolCallId: string,
      updater: (children: SubTimelineItem[]) => SubTimelineItem[]
    ): TimelineItem[] =>
      timeline.map((item) =>
        item.type === "tool" && item.id === parentToolCallId
          ? { ...item, children: updater(item.children ?? []) }
          : item
      ),
    []
  );

  const handleEvent = useCallback((event: AiChatStreamEvent) => {
    setState((prev) => {
      const next = { ...prev, timeline: [...prev.timeline] };

      if (event.conversationId) {
        next.conversationId = event.conversationId;
      }

      const isSubAgent = !!event.parentToolCallId;

      switch (event.outputType) {
        case "REASONING":
          if (event.reasoningContent) {
            if (isSubAgent) {
              next.timeline = appendToToolChildren(
                next.timeline,
                event.parentToolCallId!,
                (children) =>
                  appendReasoningToSubTimeline(
                    children,
                    event.reasoningContent!
                  )
              );
            } else {
              next.status = "reasoning";
              next.timeline = appendReasoningToTimeline(
                next.timeline,
                event.reasoningContent
              );
            }
          }
          break;

        case "CONTENT": {
          next.status = "running";
          if (event.reasoningDurationMs && !isSubAgent) {
            next.reasoningDurationMs = event.reasoningDurationMs;
            next.timeline = updateLastTimelineReasoningDuration(
              next.timeline,
              event.reasoningDurationMs
            );
          }
          if (event.content) {
            if (isSubAgent) {
              next.timeline = appendToToolChildren(
                next.timeline,
                event.parentToolCallId!,
                (children) => {
                  let updated = [...children];
                  if (event.reasoningDurationMs) {
                    updated = updateLastSubTimelineReasoningDuration(
                      updated,
                      event.reasoningDurationMs
                    );
                  }
                  const last = updated[updated.length - 1];
                  if (last && last.type === "content") {
                    return [
                      ...updated.slice(0, -1),
                      { ...last, text: last.text + event.content },
                    ];
                  }
                  return [...updated, { type: "content" as const, text: event.content! }];
                }
              );
            } else {
              const last = next.timeline[next.timeline.length - 1];
              if (last && last.type === "content") {
                next.timeline[next.timeline.length - 1] = {
                  ...last,
                  text: last.text + event.content,
                };
              } else {
                next.timeline.push({ type: "content", text: event.content });
              }
            }
          }
          break;
        }

        case "TOOL_CALL":
          next.status = "running";
          if (event.toolCalls) {
            for (const tc of event.toolCalls) {
              if (isSubAgent) {
                next.timeline = appendToToolChildren(
                  next.timeline,
                  event.parentToolCallId!,
                  (children) => {
                    if (children.some((c) => c.type === "tool" && c.id === tc.id)) return children;
                    return [
                      ...children,
                      {
                        type: "tool" as const,
                        id: tc.id,
                        name: tc.name,
                        arguments: tc.arguments,
                        status: "calling" as const,
                      },
                    ];
                  }
                );
              } else {
                const exists = next.timeline.some(
                  (item) => item.type === "tool" && item.id === tc.id
                );
                if (!exists) {
                  next.timeline.push({
                    type: "tool",
                    id: tc.id,
                    name: tc.name,
                    arguments: tc.arguments,
                    status: "calling",
                    agentName: event.agentName,
                  });
                }
              }
            }
          }
          break;

        case "TOOL_FINISHED":
          if (event.toolCallId) {
            const toolItemStatus = event.toolStatus === "error" ? "error" as const : "done" as const;
            if (isSubAgent) {
              next.timeline = appendToToolChildren(
                next.timeline,
                event.parentToolCallId!,
                (children) =>
                  children.map((c) =>
                    c.type === "tool" && c.id === event.toolCallId
                      ? { ...c, status: toolItemStatus, result: event.toolResult }
                      : c
                  )
              );
            } else {
              next.timeline = next.timeline.map((item) =>
                item.type === "tool" && item.id === event.toolCallId
                  ? { ...item, status: toolItemStatus, result: event.toolResult }
                  : item
              );
            }
          }
          break;

        case "DONE":
          next.status = "done";
          if (event.content) {
            const last = next.timeline[next.timeline.length - 1];
            if (last && last.type === "content") {
              next.timeline[next.timeline.length - 1] = {
                ...last,
                text: last.text + event.content,
              };
            } else {
              next.timeline.push({ type: "content", text: event.content });
            }
          }
          break;

        case "ERROR":
          if (isSubAgent) {
            next.timeline = appendToToolChildren(
              next.timeline,
              event.parentToolCallId!,
              (children) => [
                ...children,
                {
                  type: "content" as const,
                  text: `❌ ${event.agentName || "子Agent"} 出错: ${event.error || "未知错误"}`,
                },
              ]
            );
          } else {
            next.status = "error";
            next.error = event.error || "未知错误";
          }
          break;

        case "CANCELLED":
          next.status = "cancelled";
          break;
      }

      return next;
    });
  }, [appendToToolChildren]);

  const startStream = useCallback(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    setState({
      status: "reasoning",
      reasoningText: "",
      timeline: [],
    });

    const controller = pipelineStream(request, {
      onEvent: handleEvent,
      onError: (err) => {
        setState((prev) => ({
          ...prev,
          status: "error",
          error: err.message,
        }));
        onError?.(err.message);
      },
      onComplete: () => {
        setState((prev) => {
          if (prev.status === "running" || prev.status === "reasoning") {
            return { ...prev, status: "done" };
          }
          return prev;
        });
      },
    });

    abortRef.current = controller;
  }, [request, handleEvent, onError]);

  // 自动启动
  useEffect(() => {
    if (autoStart && !startedRef.current) {
      queueMicrotask(() => {
        if (!startedRef.current) {
          startStream();
        }
      });
    }
  }, [autoStart, startStream]);

  // 完成回调
  useEffect(() => {
    if (state.status === "done") {
      onComplete?.(state.conversationId);
    }
  }, [state.status, state.conversationId, onComplete]);

  const handleCancel = async () => {
    abortRef.current?.abort();
    if (state.conversationId) {
      try {
        await cancelPipeline(state.conversationId);
      } catch {
        // 忽略取消错误
      }
    }
    setState((prev) => ({ ...prev, status: "cancelled" }));
  };

  const isActive =
    state.status === "reasoning" || state.status === "running";

  return (
    <div className="space-y-4">
      {/* 状态栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isActive && (
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
          )}
          {state.status === "done" && (
            <CheckCircle2 className="h-4 w-4 text-green-400" />
          )}
          {state.status === "error" && (
            <XCircle className="h-4 w-4 text-destructive" />
          )}
          {state.status === "cancelled" && (
            <Ban className="h-4 w-4 text-muted-foreground" />
          )}
          <span className="text-sm font-medium">
            {state.status === "idle" && "准备就绪"}
            {state.status === "reasoning" && "AI 正在思考..."}
            {state.status === "running" && "正在解析..."}
            {state.status === "done" && "解析完成"}
            {state.status === "error" && "解析出错"}
            {state.status === "cancelled" && "已取消"}
          </span>
        </div>
        {isActive && (
          <button
            onClick={handleCancel}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-medium",
              "border border-border/40 hover:bg-destructive/10 hover:text-destructive",
              "transition-colors"
            )}
          >
            取消
          </button>
        )}
        {state.status === "idle" && (
          <button
            onClick={startStream}
            className={cn(
              "px-4 py-1.5 rounded-lg text-xs font-medium",
              "bg-primary text-primary-foreground",
              "hover:opacity-90 transition-opacity"
            )}
          >
            开始解析
          </button>
        )}
      </div>

      {/* 时间线：工具调用和内容按到达顺序交替渲染 */}
      {state.timeline.length > 0 && (
        <div ref={scrollRef} className="space-y-2 max-h-[60vh] overflow-y-auto">
          {state.timeline.map((item, idx) => {
            if (item.type === "reasoning") {
              const title = item.durationMs
                ? `思考 (${(item.durationMs / 1000).toFixed(1)}s)`
                : isActive && idx === state.timeline.length - 1
                  ? "思考中"
                  : "思考";
              return (
                <motion.div
                  key={`reasoning-${idx}`}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <StreamThink
                    title={title}
                    content={item.text}
                    streaming={isActive && idx === state.timeline.length - 1}
                  />
                </motion.div>
              );
            }

            if (item.type === "tool") {
              const isExpanded = expandedTools.has(item.id);
              const hasResult = (item.status === "done" || item.status === "error") && item.result;
              const hasChildren = item.children && item.children.length > 0;
              const canExpand = hasResult || hasChildren;
              const toggleExpand = () => {
                if (!canExpand) return;
                setExpandedTools((prev) => {
                  const next = new Set(prev);
                  if (next.has(item.id)) {
                    next.delete(item.id);
                  } else {
                    next.add(item.id);
                  }
                  return next;
                });
              };

              return (
                <motion.div
                  key={`tool-${item.id}`}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className={cn(
                    "rounded-xl text-sm border overflow-hidden",
                    item.status === "calling" &&
                      "border-blue-500/20 bg-blue-500/5",
                    item.status === "done" &&
                      "border-green-500/20 bg-green-500/5",
                    item.status === "error" &&
                      "border-destructive/20 bg-destructive/5"
                  )}
                >
                  {/* 工具调用标题行 */}
                  <div
                    className={cn(
                      "flex items-center gap-3 px-4 py-2.5",
                      canExpand && "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                    )}
                    onClick={toggleExpand}
                  >
                    {item.status === "calling" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />
                    ) : item.status === "done" ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                    )}
                    {item.agentName || item.name === "episode_scene_writer" ? (
                      <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
                    ) : (
                      <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    )}
                    <span className="font-medium text-xs">
                      {getToolDisplayName(item.name)}
                    </span>
                    {item.status === "calling" && (
                      <span className="text-xs text-muted-foreground ml-auto">
                        执行中...
                      </span>
                    )}
                    {item.status === "done" && (
                      <span className="flex items-center gap-1.5 text-xs text-green-400/80 ml-auto">
                        ✓ 完成
                        {canExpand && (
                          isExpanded
                            ? <ChevronDown className="h-3 w-3" />
                            : <ChevronRight className="h-3 w-3" />
                        )}
                      </span>
                    )}
                    {item.status === "error" && (
                      <span className="flex items-center gap-1.5 text-xs text-destructive ml-auto">
                        ✗ 失败
                        {canExpand && (
                          isExpanded
                            ? <ChevronDown className="h-3 w-3" />
                            : <ChevronRight className="h-3 w-3" />
                        )}
                      </span>
                    )}
                  </div>

                  {/* 工具调用结果展示区域 + 子 Agent 嵌套内容 */}
                  <AnimatePresence>
                    {isExpanded && (hasResult || (item.children && item.children.length > 0)) && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className={cn(
                          "border-t px-4 py-3 space-y-2",
                          item.status === "error" ? "border-destructive/10" : "border-green-500/10"
                        )}>
                          {/* 子 Agent 嵌套时间线 */}
                          {item.children && item.children.length > 0 && (
                            <div className="space-y-2 pl-2 border-l-2 border-purple-500/20">
                              {item.children.map((child, ci) => {
                                const childCount = item.children?.length ?? 0;
                                if (child.type === "reasoning") {
                                  const childIsStreaming =
                                    item.status === "calling" &&
                                    ci === childCount - 1;
                                  const childTitle = child.durationMs
                                    ? `子Agent 思考 (${(child.durationMs / 1000).toFixed(1)}s)`
                                    : childIsStreaming
                                      ? "子Agent 思考中"
                                      : "子Agent 思考";
                                  return (
                                    <div key={`sub-reasoning-${ci}`} className="text-xs">
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
                                    <div
                                      key={`sub-tool-${child.id}`}
                                      className={cn(
                                        "rounded-lg border text-xs px-3 py-2 flex items-center gap-2",
                                        child.status === "calling" && "border-blue-500/20 bg-blue-500/5",
                                        child.status === "done" && "border-green-500/20 bg-green-500/5",
                                        child.status === "error" && "border-destructive/20 bg-destructive/5"
                                      )}
                                    >
                                      {child.status === "calling" ? (
                                        <Loader2 className="h-3 w-3 animate-spin text-blue-400 shrink-0" />
                                      ) : child.status === "done" ? (
                                        <CheckCircle2 className="h-3 w-3 text-green-400 shrink-0" />
                                      ) : (
                                        <XCircle className="h-3 w-3 text-destructive shrink-0" />
                                      )}
                                      <Wrench className="h-3 w-3 text-muted-foreground shrink-0" />
                                      <span className="font-medium">{getToolDisplayName(child.name)}</span>
                                      <span className="ml-auto text-muted-foreground/60">
                                        {child.status === "calling" ? "执行中..." : child.status === "done" ? "✓" : "✗"}
                                      </span>
                                    </div>
                                  );
                                }
                                // content
                                return (
                                  <div
                                    key={`sub-content-${ci}`}
                                    className="rounded-lg border border-border/20 bg-card/20 p-3 text-xs leading-relaxed"
                                  >
                                    <XMarkdown content={child.text} />
                                  </div>
                                );
                              })}
                            </div>
                          )}
                          {/* 工具执行结果 */}
                          {hasResult && (
                            <ToolResultDisplay
                              toolName={item.name}
                              result={item.result!}
                            />
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            }

            // content：如果紧跟在子 Agent 工具之后且内容与工具 result 相同，说明是重复输出，跳过
            {
              const prevItem = idx > 0 ? state.timeline[idx - 1] : null;
              if (
                prevItem?.type === "tool" &&
                prevItem.children && prevItem.children.length > 0 &&
                prevItem.result &&
                item.text.trim() === prevItem.result.trim()
              ) {
                return null;
              }
            }
            return (
              <motion.div
                key={`content-${idx}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={cn(
                  "rounded-xl border border-border/30 bg-card/30 p-4",
                  "text-sm leading-relaxed"
                )}
              >
                <XMarkdown
                  content={item.text}
                  streaming={
                    isActive && idx === state.timeline.length - 1
                      ? { hasNextChunk: true, tail: true, enableAnimation: true }
                      : undefined
                  }
                />
              </motion.div>
            );
          })}
        </div>
      )}

      {/* 错误信息 */}
      {state.error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive"
        >
          {state.error}
        </motion.div>
      )}
    </div>
  );
}
