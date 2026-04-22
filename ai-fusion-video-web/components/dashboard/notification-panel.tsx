"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { createPortal } from "react-dom";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Ban,
  X,
  Trash2,
  ChevronDown,
  ChevronRight,
  Wrench,
  Bot,
  Maximize2,
  Minimize2,
  Clock,
  MessageSquare,
  BookOpen,
  ListTree,
  Film,
  Clapperboard,
  MapPin,
  AlertTriangle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Think } from "@ant-design/x";
import { XMarkdown } from "@ant-design/x-markdown";
import { cn } from "@/lib/utils";
import { StreamMarkdown } from "@/components/dashboard/stream-markdown";
import { resolveMediaUrl } from "@/lib/api/client";
import { GenerationModelCapabilitiesResult } from "@/components/dashboard/generation-model-capabilities-result";
import {
  ImageGenerateResult,
  VideoGenerateResult,
} from "@/components/dashboard/generation-media-result";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import type { PipelineTask, TimelineItem, SubTimelineItem } from "@/lib/store/pipeline-store";
import {
  listConversations,
  listMessages,
  type AgentConversation,
  type AgentMessage,
} from "@/lib/api/ai-assistant";
import { PIPELINE_AGENT_TYPES } from "@/lib/api/ai-pipeline";

// ========== 工具名中文映射 ==========

const toolDisplayNames: Record<string, string> = {
  // ── 项目 ──
  get_project: "查询项目信息",
  list_my_projects: "查询我的项目",

  // ── 资产 ──
  list_project_assets: "查询项目资产列表",
  create_asset: "创建资产",
  get_asset: "查询资产详情",
  update_asset: "更新资产",
  batch_create_assets: "批量创建资产",
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
  save_script_episode: "保存剧本分集",
  save_script_scene_items: "保存剧本场次",
  get_script_episode: "查询剧本分集详情",
  get_script_scene: "查询剧本场次详情",
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

  // ── 生视频 ──
  generate_video: "AI 生成视频",
  update_storyboard_item_video: "更新分镜视频",
  get_storyboard_scene_items: "查询场次镜头列表",

  // ── 子 Agent ──
  episode_scene_writer: "分集场次解析（子Agent）",
  episode_script_creator: "分集剧本创作（子Agent）",
  episode_storyboard_writer: "分集分镜编写（子Agent）",
  storyboard_asset_preprocessor: "子资产预处理（子Agent）",
  generate_asset_image: "生成资产图片（子Agent）",
  generate_storyboard_video: "生成分镜视频（子Agent）",
};

function getToolDisplayName(name: string) {
  return toolDisplayNames[name] || name;
}

// ========== 时间格式化 ==========

function formatElapsed(durationMs: number): string {
  const secs = Math.floor(durationMs / 1000);
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m${secs % 60}s`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

/** 实时计时 hook：running 时每秒 tick，结束后停止 */
function useElapsed(task: PipelineTask): string {
  const [now, setNow] = useState(() => task.finishedAt ?? Date.now());

  useEffect(() => {
    if (task.status !== "running") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [task.status]);

  const endTime = task.finishedAt ?? now;
  return formatElapsed(endTime - task.createdAt);
}

/** 将 useElapsed hook 包装为可在 JSX 中使用的组件 */
function ElapsedText({ task }: { task: PipelineTask }) {
  const text = useElapsed(task);
  return <>{text}</>;
}

/** 纯函数：计算已结束任务的固定耗时字符串 */
function getElapsedStr(task: PipelineTask): string {
  const endTime = task.finishedAt ?? Date.now();
  return formatElapsed(endTime - task.createdAt);
}

function formatDatetime(dateStr?: string): string {
  if (!dateStr) return "";
  try {
    return formatTimestamp(new Date(dateStr).getTime());
  } catch {
    return dateStr;
  }
}

/** 将时间戳格式化为友好字符串（今天 HH:MM / 昨天 HH:MM / X月X日 HH:MM） */
function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const time = d.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });

  if (d.toDateString() === now.toDateString()) {
    return `今天 ${time}`;
  }

  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) {
    return `昨天 ${time}`;
  }

  if (d.getFullYear() === now.getFullYear()) {
    return `${d.getMonth() + 1}月${d.getDate()}日 ${time}`;
  }

  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 ${time}`;
}

// ========== 判断工具是否为子Agent ==========

const SUB_AGENT_TOOL_NAMES = [
  "episode_scene_writer",
  "episode_script_creator",
  "episode_storyboard_writer",
  "storyboard_asset_preprocessor",
  "generate_asset_image",
  "generate_storyboard_video",
];

function isSubAgentTool(name: string) {
  return SUB_AGENT_TOOL_NAMES.includes(name);
}

// ========== 工具调用行（紧凑版，用于左侧面板） ==========

function ToolItem({
  item,
  large,
}: {
  item: Extract<TimelineItem, { type: "tool" }>;
  large: boolean;
}) {
  const iconSize = large ? "h-3.5 w-3.5" : "h-2.5 w-2.5";
  const textSize = large ? "text-xs" : "text-[10px]";

  return (
    <div
      className={cn(
        "flex items-center gap-1.5 rounded-lg",
        large ? "px-2.5 py-1.5" : "px-2 py-1",
        item.status === "calling" && "bg-blue-500/5",
        item.status === "done" && "bg-green-500/5",
        item.status === "error" && "bg-destructive/5"
      )}
    >
      {item.status === "calling" ? (
        <Loader2
          className={cn("animate-spin text-blue-400 shrink-0", iconSize)}
        />
      ) : item.status === "done" ? (
        <CheckCircle2
          className={cn("text-green-400 shrink-0", iconSize)}
        />
      ) : (
        <XCircle className={cn("text-destructive shrink-0", iconSize)} />
      )}
      {isSubAgentTool(item.name) ? (
        <Bot className={cn("text-purple-400 shrink-0", iconSize)} />
      ) : (
        <Wrench
          className={cn("text-muted-foreground/60 shrink-0", iconSize)}
        />
      )}
      <span className={cn("font-medium truncate", textSize)}>
        {getToolDisplayName(item.name)}
      </span>
      <span
        className={cn("ml-auto shrink-0", textSize, {
          "text-muted-foreground/60": item.status === "calling",
          "text-green-400/80": item.status === "done",
          "text-destructive/80": item.status === "error",
        })}
      >
        {item.status === "calling" ? "…" : item.status === "done" ? "✓" : "✗"}
      </span>
    </div>
  );
}

// ========== 可展开的工具调用卡片（用于右侧详情面板） ==========

function ExpandableToolCard({
  toolName,
  toolStatus,
  result,
}: {
  toolName: string;
  toolStatus: "done" | "error";
  result?: string;
}) {
  const [expanded, setExpanded] = useState(true); // 默认展开
  const hasResult = !!result;

  return (
    <div
      className={cn(
        "rounded-xl text-sm border overflow-hidden",
        toolStatus === "done"
          ? "border-green-500/20 bg-green-500/5"
          : "border-destructive/20 bg-destructive/5"
      )}
    >
      {/* 标题行 */}
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
        ) : (
          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
        )}
        {isSubAgentTool(toolName) ? (
          <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        ) : (
          <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="font-medium text-xs">
          {getToolDisplayName(toolName)}
        </span>
        <span className="flex items-center gap-1 ml-auto">
          {hasResult &&
            (expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/50" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
            ))}
        </span>
      </div>

      {/* 结果展示 - 默认展开 */}
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
                    : "border-green-500/10"
                )}
              >
                <ToolResultDisplay toolName={toolName} content={result!} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      )}
    </div>
  );
}

// ========== 正在调用中的工具卡片 ==========

function CallingToolCard({ toolName }: { toolName: string }) {
  return (
    <div
      className={cn(
        "rounded-xl text-sm border overflow-hidden",
        "border-blue-500/20 bg-blue-500/5"
      )}
    >
      <div className="flex items-center gap-3 px-4 py-2.5">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />
        {isSubAgentTool(toolName) ? (
          <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        ) : (
          <Wrench className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="font-medium text-xs">
          {getToolDisplayName(toolName)}
        </span>
        <span className="text-xs text-blue-400/80 ml-auto">调用中…</span>
      </div>
    </div>
  );
}

// ========== 智能自动滚动 Hook ==========

/**
 * 自动滚到底部，用户手动上滚可打断，滚回底部后恢复自动跟随。
 * @param deps  内容变化的依赖项（每次变化触发滚底检测）
 * @param active 是否处于活跃状态（如 running），非活跃不自动滚
 */
function useSmartScroll(deps: unknown[], active: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  const userScrolledUp = useRef(false);

  // 判断是否「接近底部」（容差 40px）
  const isNearBottom = useCallback(() => {
    const el = ref.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }, []);

  // 监听用户滚动：上滚 → 打断；滚回底部 → 恢复
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onScroll = () => {
      userScrolledUp.current = !isNearBottom();
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [isNearBottom]);

  // 内容变化时：如果没有被打断且处于活跃状态 → 滚到底部
  useEffect(() => {
    if (!active) return;
    if (!userScrolledUp.current && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

// ========== 子 Agent 嵌套卡片 ==========

function SubAgentCard({
  item,
}: {
  item: Extract<TimelineItem, { type: "tool" }>;
}) {
  const [expanded, setExpanded] = useState(true);
  const children = item.children ?? [];
  const isRunning = item.status === "calling";
  const hasChildren = children.length > 0;
  const hasResult = !isRunning && !!item.result;
  const hasContent = hasChildren || hasResult;

  // 子 Agent 内部智能自动滚动
  const innerScrollRef = useSmartScroll([children], isRunning);

  // 统计子 Agent 内部工具调用数
  const toolCount = children.filter((c) => c.type === "tool").length;
  const doneToolCount = children.filter(
    (c) => c.type === "tool" && (c.status === "done" || c.status === "error")
  ).length;

  return (
    <div
      className={cn(
        "rounded-xl text-sm border overflow-hidden",
        isRunning
          ? "border-purple-500/20 bg-purple-500/5"
          : item.status === "done"
            ? "border-green-500/20 bg-green-500/5"
            : "border-destructive/20 bg-destructive/5"
      )}
    >
      {/* 标题行 */}
      <div
        className={cn(
          "flex items-center gap-2.5 px-4 py-2.5",
          "cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
        )}
        onClick={() => setExpanded(!expanded)}
      >
        {isRunning ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-purple-400 shrink-0" />
        ) : item.status === "done" ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
        ) : (
          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
        )}
        <Bot className="h-3.5 w-3.5 text-purple-400 shrink-0" />
        <span className="font-medium text-xs">
          {getToolDisplayName(item.name)}
        </span>
        {toolCount > 0 && (
          <span className="text-[10px] text-muted-foreground/60 ml-1">
            {isRunning ? `${doneToolCount}/${toolCount}` : `${toolCount} 步`}
          </span>
        )}
        {isRunning && (
          <span className="text-xs text-purple-400/80 ml-auto">运行中…</span>
        )}
        <span className="ml-auto shrink-0">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/50" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/50" />
          )}
        </span>
      </div>

      {/* 子 Agent 内部时间线 + 最终结果 */}
      <AnimatePresence initial={false}>
        {expanded && hasContent && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div ref={innerScrollRef} className="border-t border-purple-500/10 px-4 py-3 space-y-2 max-h-[400px] overflow-y-auto">
              {children.map((child, cidx) => {
                if (child.type === "reasoning") {
                  const childIsStreaming =
                    isRunning && cidx === children.length - 1;
                  const childTitle = child.durationMs
                    ? `思考 (${(child.durationMs / 1000).toFixed(1)}s)`
                    : childIsStreaming
                      ? "思考中"
                      : "思考";
                  return (
                    <Think
                      key={`sub-reasoning-${cidx}`}
                      style={{ maxHeight: 120, overflowY: "auto" }}
                      title={childTitle}
                    >
                      <StreamMarkdown
                        content={child.text}
                        compact
                        streaming={childIsStreaming}
                      />
                    </Think>
                  );
                }
                if (child.type === "tool") {
                  if (child.status === "calling") {
                    return (
                      <CallingToolCard
                        key={`sub-tool-${child.id}`}
                        toolName={child.name}
                      />
                    );
                  }
                  return (
                    <ExpandableToolCard
                      key={`sub-tool-${child.id}`}
                      toolName={child.name}
                      toolStatus={child.status}
                      result={child.result}
                    />
                  );
                }
                if (child.type === "content") {
                  return (
                    <div
                      key={`sub-content-${cidx}`}
                      className="text-xs leading-relaxed text-muted-foreground/80"
                    >
                      <XMarkdown content={child.text} />
                    </div>
                  );
                }
                return null;
              })}

              {/* 子 Agent 最终执行结果（纯文本总结） */}
              {hasResult && (
                <div className="text-xs text-foreground/70 leading-relaxed">
                  {item.result}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}


interface MessageTimelineProps {
  /** 思考内容 */
  reasoningText?: string;
  /** 思考耗时 */
  reasoningDurationMs?: number;
  /** 时间线条目（使用 store 的 TimelineItem，含 children） */
  timeline: TimelineItem[];
  /** 是否正在流式输出 */
  streaming?: boolean;
  /** 错误信息 */
  error?: string;
}

function MessageTimeline({ reasoningText, reasoningDurationMs, timeline, streaming, error }: MessageTimelineProps) {
  const hasTimelineReasoning = timeline.some((item) => item.type === "reasoning");
  const reasoningTitle = reasoningDurationMs
    ? `思考 (${(reasoningDurationMs / 1000).toFixed(1)}s)`
    : streaming
      ? "思考中"
      : "思考";

  return (
    <>
      {/* 历史兼容：旧数据里 reasoning 仍是单独字段时，作为头部兜底展示 */}
      {!hasTimelineReasoning && reasoningText && (
        <Think
          style={{ maxHeight: 192, overflowY: "auto" }}
          title={reasoningTitle}
        >
          <StreamMarkdown content={reasoningText} streaming={!!streaming} />
        </Think>
      )}

      {/* 时间线条目 */}
      {timeline.map((item, idx) => {
        if (item.type === "reasoning") {
          const itemTitle = item.durationMs
            ? `思考 (${(item.durationMs / 1000).toFixed(1)}s)`
            : streaming && idx === timeline.length - 1
              ? "思考中"
              : "思考";
          return (
            <Think
              key={`reasoning-${idx}`}
              style={{ maxHeight: 192, overflowY: "auto" }}
              title={itemTitle}
            >
              <StreamMarkdown
                content={item.text}
                streaming={streaming && idx === timeline.length - 1}
              />
            </Think>
          );
        }

        if (item.type === "tool") {
          // 子 Agent 工具 → 渲染嵌套卡片
          if (isSubAgentTool(item.name) || (item.children && item.children.length > 0)) {
            return <SubAgentCard key={`sub-agent-${item.id}`} item={item} />;
          }
          // 普通工具 - 正在调用中
          if (item.status === "calling") {
            return <CallingToolCard key={`tool-${item.id}`} toolName={item.name} />;
          }
          // 普通工具 - 已完成/出错
          return (
            <ExpandableToolCard
              key={`tool-${item.id}`}
              toolName={item.name}
              toolStatus={item.status}
              result={item.result}
            />
          );
        }
        // 文本内容：如果紧跟在子 Agent 工具之后且内容与工具 result 相同，说明是重复输出，跳过
        {
          const prevItem = idx > 0 ? timeline[idx - 1] : null;
          if (
            prevItem?.type === "tool" &&
            (isSubAgentTool(prevItem.name) || (prevItem.children && prevItem.children.length > 0)) &&
            prevItem.result &&
            item.text.trim() === prevItem.result.trim()
          ) {
            return null;
          }
        }
        return (
          <div key={`content-${idx}`} className="text-sm leading-relaxed">
            <XMarkdown
              content={item.text}
            />
          </div>
        );
      })}

      {/* 错误 */}
      {error && (
        <p className="text-xs text-destructive">❌ {error}</p>
      )}
    </>
  );
}

// ========== 实时 Pipeline 详情面板（右侧） ==========

function PipelineDetailPanel({ task }: { task: PipelineTask }) {
  const [idleTimelineLength, setIdleTimelineLength] = useState<number | null>(null);
  const timelineLength = task.state.timeline.length;
  const isIdle = task.status === "running" && idleTimelineLength === timelineLength;

  // 外层智能自动滚动（用户上滚打断，滚回底部恢复）
  const timelineRef = useSmartScroll(
    [task.state.timeline, isIdle],
    task.status === "running"
  );

  // 空闲检测：超过 2 秒无新 timeline 更新时显示提示
  useEffect(() => {
    if (task.status !== "running") return;
    const timer = setTimeout(() => {
      setIdleTimelineLength(timelineLength);
    }, 2000);
    return () => clearTimeout(timer);
  }, [timelineLength, task.status]);

  const statusText = {
    running: "运行中",
    done: "已完成",
    error: "出错",
    cancelled: "已取消",
  };

  const statusColor = {
    running: "text-blue-400",
    done: "text-green-400",
    error: "text-destructive",
    cancelled: "text-muted-foreground",
  };

  return (
    <div className="flex flex-col h-full">
      {/* 标题 */}
      <div className="px-4 py-3 border-b border-border/20 shrink-0 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="text-sm font-semibold truncate">{task.label}</h4>
          <p
            className={cn(
              "text-xs mt-0.5",
              statusColor[task.status]
            )}
          >
            {statusText[task.status]} · <ElapsedText task={task} />
            <span className="text-muted-foreground/50 ml-1">
              启动于 {formatTimestamp(task.createdAt)}
            </span>
          </p>
        </div>
        {task.status === "running" && (
          <button
            onClick={() => usePipelineStore.getState().cancelPipeline(task.id)}
            className={cn(
              "shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium",
              "border border-destructive/20 text-destructive/70",
              "hover:bg-destructive/10 hover:text-destructive hover:border-destructive/40",
              "transition-colors"
            )}
            title="停止工作流"
          >
            <Ban className="h-3 w-3" />
            停止
          </button>
        )}
      </div>

      {/* 内容 */}
      <div ref={timelineRef} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
        <MessageTimeline
          reasoningText={task.state.reasoningText}
          reasoningDurationMs={task.state.reasoningDurationMs}
          timeline={task.state.timeline}
          streaming={task.status === "running"}
          error={task.state.error}
        />

        {/* 空闲提示 */}
        <AnimatePresence>
          {isIdle && task.status === "running" && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.25 }}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-primary/5 border border-primary/10"
            >
              <Loader2 className="h-3 w-3 animate-spin text-primary/60" />
              <span className="text-xs text-primary/70 font-medium">AI 全力处理中…</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ========== Agent 类型中文名称映射 ==========

const agentTypeNames: Record<string, string> = {
  script_full_parse: "剧本全量解析",
  story_to_script: "故事转剧本",
  script_to_storyboard: "剧本转分镜",
  script_episode_parse: "分集上传解析",
  episode_scene_writer: "分集场次编写",
  episode_script_creator: "分集剧本创作",
  episode_storyboard_writer: "分集分镜编写",
  storyboard_asset_preprocessor: "子资产预处理",
  asset_image_gen: "资产图片生成",
  asset_image_executor: "资产图片执行",
  storyboard_video_gen: "分镜视频生成",
  storyboard_video_executor: "分镜视频执行",
  script_assistant: "剧本助手",
  ai_media: "默认助手",
  concept_visualizer: "概念可视化",
};

function getAgentTypeName(type: string): string {
  return agentTypeNames[type] || type;
}

// ========== 资产类型中文映射 ==========

const assetTypeNames: Record<string, string> = {
  character: "角色",
  scene: "场景",
  prop: "道具",
  vehicle: "载具",
  building: "建筑",
  costume: "服装",
  effect: "特效",
};

const storyboardShotTypeNames: Record<string, string> = {
  远景: "远景",
  全景: "全景",
  中景: "中景",
  近景: "近景",
  特写: "特写",
};

// ========== 工具结果格式化显示（带 toolName 分派） ==========

/** 将值友好化展示 */
function formatResultValue(val: unknown): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "boolean") return val ? "是" : "否";
  if (typeof val === "number") return String(val);
  if (typeof val === "string") {
    return val.length > 150 ? val.slice(0, 150) + "…" : val;
  }
  if (Array.isArray(val)) return `[${val.length} 项]`;
  if (typeof val === "object") return `{${Object.keys(val).length} 个字段}`;
  return String(val);
}

function ToolResultDisplay({ toolName, content }: { toolName: string; content: string }) {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    // 纯文本原文显示
    const isLong = content.length > 300;
    return (
      <div className="text-xs text-foreground/70 whitespace-pre-wrap leading-relaxed">
        {isLong ? content.slice(0, 300) + "…" : content}
      </div>
    );
  }

  // 错误状态统一展示
  if (
    typeof parsed === "object" &&
    parsed !== null &&
    (parsed as Record<string, unknown>).status === "error"
  ) {
    const msg = (parsed as Record<string, unknown>).message;
    return (
      <p className="text-xs text-destructive inline-flex items-center gap-1">
        <XCircle className="h-3.5 w-3.5 shrink-0" /> {typeof msg === "string" ? msg : "操作失败"}
      </p>
    );
  }

  // 根据工具类型做针对性展示
  switch (toolName) {
    case "get_generation_model_capabilities":
      return <GenerationModelCapabilitiesResult data={parsed} />;
    case "generate_image":
      return <ImageGenerateResult data={parsed} />;
    case "generate_video":
      return <VideoGenerateResult data={parsed} />;
    case "list_project_assets":
      return <AssetListResult data={parsed} />;
    case "query_asset_metadata":
      return <MetadataResult data={parsed} />;
    case "batch_create_assets":
      return <BatchCreateResult data={parsed} />;
    case "get_project_script":
      return <ScriptInfoResult data={parsed} />;
    case "get_script_structure":
      return <ScriptStructureResult data={parsed} />;
    case "get_script_episode":
      return <EpisodeDetailResult data={parsed} />;
    case "get_script_scene":
      return <SceneDetailResult data={parsed} />;
    case "query_asset_items":
      return <AssetItemsResult data={parsed} />;
    case "save_script_episode":
      return <SaveEpisodeResult data={parsed} />;
    case "save_script_scene_items":
    case "update_script_info":
    case "update_script_scene":
    case "manage_script_scenes":
    case "update_script":
    case "update_asset":
    case "update_asset_image":
      return <MutationResult data={parsed} toolName={toolName} />;
    case "create_asset":
    case "add_asset_item":
    case "batch_create_asset_items":
      return <CreateResult data={parsed} toolName={toolName} />;
    case "get_project":
      return <ProjectInfoResult data={parsed} />;
    case "insert_storyboard_item":
    case "save_storyboard_episode":
    case "save_storyboard_scene_shots":
    case "get_storyboard":
    case "list_project_storyboards":
      return <StoryboardResult data={parsed} toolName={toolName} />;
    case "get_script":
    case "get_asset":
    case "list_my_projects":
    default:
      return <GenericResult data={parsed} />;
  }
}

/** 资产列表结果 */
function AssetListResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const assets = (obj.assets as Array<Record<string, unknown>>) || [];
  const total = (obj.total as number) ?? assets.length;
  const typeStr = obj.type as string;

  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">
        共 <span className="font-medium text-foreground">{total}</span> 项
        {typeStr && typeStr !== "all" && (
          <span className="ml-1 px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px]">
            {assetTypeNames[typeStr] || typeStr}
          </span>
        )}
      </p>
      {assets.length > 0 && (
        <ul className="space-y-1">
          {assets.slice(0, 10).map((asset, i) => (
            <li
              key={asset.id ? String(asset.id) : i}
              className="flex items-center gap-2 text-xs text-muted-foreground/90"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400/40 shrink-0" />
              <span className="font-medium text-foreground">
                {String(asset.name || "未命名")}
              </span>
              {!!asset.type && (
                <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                  {assetTypeNames[String(asset.type)] || String(asset.type)}
                </span>
              )}
              {asset.itemCount !== undefined && (
                <span className="text-[10px]">
                  {String(asset.itemCount)} 个子项
                </span>
              )}
            </li>
          ))}
          {assets.length > 10 && (
            <li className="text-[10px] text-muted-foreground/60 pl-3">
              …还有 {assets.length - 10} 项
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

/** 元数据结果 */
function MetadataResult({ data }: { data: unknown }) {
  if (typeof data !== "object" || data === null) {
    return <GenericResult data={data} />;
  }

  const obj = data as Record<string, unknown>;
  const assetType = obj.assetType as string | undefined;
  const properties =
    (obj.fields as Array<Record<string, unknown>>) ||
    (obj.properties as Array<Record<string, unknown>>) ||
    (obj.attributes as Array<Record<string, unknown>>);

  if (!properties || !Array.isArray(properties)) {
    return <GenericResult data={data} />;
  }

  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">
        {assetType && (
          <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 text-[10px] mr-2">
            {assetTypeNames[assetType] || assetType}
          </span>
        )}
        共 <span className="font-medium text-foreground">{properties.length}</span> 个属性
      </p>
      <ul className="space-y-0.5">
        {properties.slice(0, 15).map((prop, i) => (
          <li
            key={i}
            className="flex items-center gap-2 text-xs text-muted-foreground/90"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400/40 shrink-0" />
            <span className="font-medium text-foreground">
              {String(prop.fieldLabel || prop.fieldKey || prop.name || prop.key || `属性${i + 1}`)}
            </span>
            {!!(prop.fieldType || prop.type) && (
              <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                {String(prop.fieldType || prop.type)}
              </span>
            )}
            {prop.required === true && (
              <span className="text-[10px] text-orange-400">必填</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** 批量创建结果 */
function BatchCreateResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const created = obj.created as Array<unknown> | undefined;
  const existing = obj.existing as Array<unknown> | undefined;
  const message = obj.message as string | undefined;
  const createdCount = (obj.createdCount as number) ?? created?.length;
  const existingCount = (obj.existingCount as number) ?? existing?.length;

  return (
    <div className="space-y-2">
      {message && (
        <p className="text-xs text-muted-foreground">{message}</p>
      )}
      {/* 统计标签 */}
      <div className="flex flex-wrap gap-1.5">
        {createdCount !== undefined && createdCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/10 text-green-500 text-[10px] font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            新建 {createdCount} 项
          </span>
        )}
        {existingCount !== undefined && existingCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 text-[10px] font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            复用 {existingCount} 项
          </span>
        )}
      </div>
      {/* 创建的资产列表 */}
      {created && Array.isArray(created) && created.length > 0 && (
        <ul className="space-y-0.5">
          {created.slice(0, 8).map((item, i) => {
            const it = item as Record<string, unknown>;
            return (
              <li
                key={i}
                className="flex items-center gap-2 text-xs text-muted-foreground/90"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-green-400/60 shrink-0" />
                <span className="font-medium text-foreground">
                  {String(it.name || it.id || `#${i + 1}`)}
                </span>
                {!!it.type && (
                  <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                    {assetTypeNames[String(it.type)] || String(it.type)}
                  </span>
                )}
              </li>
            );
          })}
          {created.length > 8 && (
            <li className="text-[10px] text-muted-foreground/60 pl-3">
              …还有 {created.length - 8} 项
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

/** 写入/更新操作结果 */
function MutationResult({
  data,
  toolName,
}: {
  data: unknown;
  toolName: string;
}) {
  const obj = data as Record<string, unknown>;
  const status = obj.status as string | undefined;
  const message = obj.message as string | undefined;
  const id = obj.id ?? obj.episodeId ?? obj.scriptId;

  const toolLabels: Record<string, string> = {
    save_scene_items: "场次",
    update_script_info: "剧本信息",
    update_script_scene: "场次",
    manage_script_scenes: "场次",
    update_script: "剧本",
    update_asset: "资产",
    update_asset_image: "资产图片",
  };
  const label = toolLabels[toolName] || "数据";

  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
        {status === "error" ? <AlertTriangle className="h-3.5 w-3.5 text-yellow-500 shrink-0" /> : <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />}{" "}
        {message || (status === "error" ? `${label}操作失败` : `${label}已更新`)}
      </p>
      {id !== undefined && (
        <p className="text-[10px] text-muted-foreground/60">
          {label} ID: {String(id)}
        </p>
      )}
      {obj.sceneCount !== undefined && (
        <p className="text-[10px] text-muted-foreground/60">
          场次数: {String(obj.sceneCount)}
        </p>
      )}
    </div>
  );
}

/** 剧本信息结果 — get_project_script */
function ScriptInfoResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const title = obj.title as string | undefined;
  const totalEpisodes = obj.totalEpisodes as number | undefined;
  const genre = obj.genre as string | undefined;
  const parsingStatus = obj.parsingStatus as number | undefined;
  const episodes = obj.episodes as Array<Record<string, unknown>> | undefined;

  const statusMap: Record<number, string> = { 0: "未解析", 1: "解析中", 2: "已完成", 3: "解析失败" };

  return (
    <div className="space-y-2">
      {/* 剧本标题卡 */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground flex items-center gap-1.5"><BookOpen className="h-3.5 w-3.5 text-indigo-400 shrink-0" /> {title || "未命名剧本"}</span>
        {genre && (
          <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 text-[10px]">
            {genre}
          </span>
        )}
        {parsingStatus !== undefined && (
          <span className={cn(
            "px-1.5 py-0.5 rounded text-[10px]",
            parsingStatus === 2 ? "bg-green-500/10 text-green-500" : "bg-yellow-500/10 text-yellow-500"
          )}>
            {statusMap[parsingStatus] || `状态${parsingStatus}`}
          </span>
        )}
      </div>
      {/* 统计信息 */}
      <div className="flex gap-3 text-[10px] text-muted-foreground">
        <span>总集数: <span className="font-medium text-foreground">{totalEpisodes ?? 0}</span></span>
        {obj.scriptId != null && <span>剧本ID: {String(obj.scriptId)}</span>}
      </div>
      {/* 分集列表 */}
      {episodes && Array.isArray(episodes) && episodes.length > 0 && (
        <div className="space-y-0.5">
          <p className="text-[10px] text-muted-foreground/70">分集概览:</p>
          <ul className="space-y-0.5">
            {episodes.slice(0, 10).map((ep, i) => (
              <li key={ep.episodeId ? String(ep.episodeId) : i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/40 shrink-0" />
                <span className="font-medium text-foreground">
                  第{String(ep.episodeNumber || i + 1)}集
                </span>
                {ep.title != null && <span className="text-muted-foreground/70 truncate">{String(ep.title)}</span>}
                {ep.totalScenes !== undefined && (
                  <span className="text-[10px] text-muted-foreground/50">{String(ep.totalScenes)}场</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** 剧本结构结果 — get_script_structure */
function ScriptStructureResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const title = obj.title as string | undefined;
  const totalEpisodes = obj.totalEpisodes as number | undefined;
  const episodes = obj.episodes as Array<Record<string, unknown>> | undefined;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground flex items-center gap-1.5"><ListTree className="h-3.5 w-3.5 text-blue-400 shrink-0" /> {title || "剧本结构"}</span>
        <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px]">
          {totalEpisodes ?? 0} 集
        </span>
      </div>
      {episodes && Array.isArray(episodes) && episodes.length > 0 && (
        <div className="space-y-1.5">
          {episodes.slice(0, 8).map((ep, i) => {
            const scenes = ep.scenes as Array<Record<string, unknown>> | undefined;
            return (
              <div key={ep.episodeId ? String(ep.episodeId) : i} className="space-y-0.5">
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/60 shrink-0" />
                  <span className="font-medium text-foreground">
                    第{String(ep.episodeNumber || i + 1)}集 {ep.title ? `— ${String(ep.title)}` : ""}
                  </span>
                  {ep.totalScenes !== undefined && (
                    <span className="text-[10px] text-muted-foreground/50">{String(ep.totalScenes)}场</span>
                  )}
                </div>
                {scenes && scenes.length > 0 && (
                  <ul className="ml-4 space-y-0">
                    {scenes.slice(0, 6).map((sc, j) => (
                      <li key={j} className="text-[10px] text-muted-foreground/70 truncate">
                        · {String(sc.sceneHeading || `场次${j + 1}`)}
                      </li>
                    ))}
                    {scenes.length > 6 && (
                      <li className="text-[10px] text-muted-foreground/50">…还有 {scenes.length - 6} 个场次</li>
                    )}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** 分集详情结果 — get_script_episode */
function EpisodeDetailResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const title = obj.title as string | undefined;
  const episodeNumber = obj.episodeNumber as number | undefined;
  const synopsis = obj.synopsis as string | undefined;
  const totalScenes = obj.totalScenes as number | undefined;
  const scenes = obj.scenes as Array<Record<string, unknown>> | undefined;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground flex items-center gap-1.5">
          <Film className="h-3.5 w-3.5 text-orange-400 shrink-0" />
          第{episodeNumber ?? "?"}集 {title || ""}
        </span>
        {totalScenes !== undefined && (
          <span className="px-1.5 py-0.5 rounded bg-orange-500/10 text-orange-400 text-[10px]">
            {totalScenes} 个场次
          </span>
        )}
        {obj.episode_version !== undefined && (
          <span className="text-[10px] text-muted-foreground/50">v{String(obj.episode_version)}</span>
        )}
      </div>
      {synopsis && (
        <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
          {synopsis.length > 150 ? synopsis.slice(0, 150) + "…" : synopsis}
        </p>
      )}
      {scenes && Array.isArray(scenes) && scenes.length > 0 && (
        <ul className="space-y-0.5">
          {scenes.slice(0, 10).map((sc, i) => (
            <li key={sc.sceneItemId ? String(sc.sceneItemId) : i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
              <span className="w-1.5 h-1.5 rounded-full bg-orange-400/40 shrink-0" />
              <span className="text-foreground truncate">
                {String(sc.sceneHeading || `场次${sc.sceneNumber || i + 1}`)}
              </span>
              {sc.characters != null && (
                <span className="text-[10px] text-muted-foreground/50 truncate">
                  {Array.isArray(sc.characters)
                    ? (sc.characters as string[]).slice(0, 3).join("、")
                    : String(sc.characters).slice(0, 30)}
                </span>
              )}
            </li>
          ))}
          {scenes.length > 10 && (
            <li className="text-[10px] text-muted-foreground/50 pl-3">…还有 {scenes.length - 10} 个场次</li>
          )}
        </ul>
      )}
    </div>
  );
}

/** 场次详情结果 — get_script_scene */
function SceneDetailResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const heading = obj.sceneHeading as string | undefined;
  const location = obj.location as string | undefined;
  const timeOfDay = obj.timeOfDay as string | undefined;
  const intExt = obj.intExt as string | undefined;
  const description = obj.sceneDescription as string | undefined;
  const characters = obj.characters as unknown;
  const dialogues = obj.dialogues as unknown;

  const charList = Array.isArray(characters) ? characters as string[] : [];
  const dialogueCount = Array.isArray(dialogues) ? (dialogues as unknown[]).length : 0;

  return (
    <div className="space-y-1.5">
      <span className="text-xs font-medium text-foreground flex items-center gap-1.5"><Clapperboard className="h-3.5 w-3.5 text-amber-400 shrink-0" /> {heading || "场次详情"}</span>
      {/* 场次属性标签 */}
      <div className="flex flex-wrap gap-1">
        {location && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-500 text-[10px]"><MapPin className="h-2.5 w-2.5" /> {location}</span>
        )}
        {timeOfDay && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 text-[10px]"><Clock className="h-2.5 w-2.5" /> {timeOfDay}</span>
        )}
        {intExt && (
          <span className="px-1.5 py-0.5 rounded bg-slate-500/10 text-slate-400 text-[10px]">{intExt}</span>
        )}
        {dialogueCount > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 text-[10px]"><MessageSquare className="h-2.5 w-2.5" /> {dialogueCount} 条对白</span>
        )}
      </div>
      {/* 出场角色 */}
      {charList.length > 0 && (
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-[10px] text-muted-foreground/70">出场角色:</span>
          {charList.slice(0, 6).map((c, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px] text-foreground/80">{String(c)}</span>
          ))}
          {charList.length > 6 && <span className="text-[10px] text-muted-foreground/50">+{charList.length - 6}</span>}
        </div>
      )}
      {/* 场景描述 */}
      {description && (
        <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
          {description.length > 200 ? description.slice(0, 200) + "…" : description}
        </p>
      )}
    </div>
  );
}

/** 子资产列表结果 — query_asset_items */
function AssetItemsResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const assets = obj.assets as Array<Record<string, unknown>> | undefined;

  if (assets && Array.isArray(assets)) {
    const totalAssets = (obj.totalAssets as number) ?? assets.length;
    const totalItems = assets.reduce((sum, asset) => {
      const assetItems = asset.items as Array<Record<string, unknown>> | undefined;
      const assetTotalItems = typeof asset.totalItems === "number"
        ? asset.totalItems
        : assetItems?.length ?? 0;
      return sum + assetTotalItems;
    }, 0);

    return (
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-medium text-foreground">子资产列表</span>
          <span className="text-[10px] text-muted-foreground/50">
            共 {totalAssets} 个资产
          </span>
          <span className="text-[10px] text-muted-foreground/50">
            共 {totalItems} 个子资产
          </span>
        </div>

        <ul className="space-y-0.5">
          {assets.slice(0, 8).map((asset, i) => {
            const status = asset.status as string | undefined;
            const message = asset.message as string | undefined;
            const assetType = asset.assetType == null ? undefined : String(asset.assetType);
            const assetItems = asset.items as Array<Record<string, unknown>> | undefined;
            const assetTotalItems = typeof asset.totalItems === "number"
              ? asset.totalItems
              : assetItems?.length ?? 0;
            const previewNames = assetItems?.slice(0, 3)
              .map(item => String(item.name || item.id || "未命名子资产"))
              .join(" / ");

            return (
              <li
                key={asset.assetId ? String(asset.assetId) : i}
                className="flex items-center gap-2 text-xs text-muted-foreground/90"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400/40 shrink-0" />
                <span className="font-medium text-foreground">
                  {String(asset.assetName || `资产${i + 1}`)}
                </span>
                {assetType ? (
                  <span className="px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 text-[10px]">
                    {assetTypeNames[assetType] || assetType}
                  </span>
                ) : null}
                {status === "error" ? (
                  <span className="text-[10px] text-destructive">
                    {message || "查询失败"}
                  </span>
                ) : (
                  <>
                    <span className="text-[10px] text-muted-foreground/50">
                      {assetTotalItems} 个子资产
                    </span>
                    {previewNames && (
                      <span className="text-[10px] text-muted-foreground/60 truncate">
                        {previewNames}
                      </span>
                    )}
                  </>
                )}
              </li>
            );
          })}
          {assets.length > 8 && (
            <li className="text-[10px] text-muted-foreground/50 pl-3">
              …还有 {assets.length - 8} 个资产
            </li>
          )}
        </ul>
      </div>
    );
  }

  const assetName = obj.assetName as string | undefined;
  const assetType = obj.assetType as string | undefined;
  const totalItems = obj.totalItems as number | undefined;
  const items = obj.items as Array<Record<string, unknown>> | undefined;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground">{assetName || "子资产列表"}</span>
        {assetType && (
          <span className="px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 text-[10px]">
            {assetTypeNames[assetType] || assetType}
          </span>
        )}
        <span className="text-[10px] text-muted-foreground/50">
          共 {totalItems ?? 0} 个子资产
        </span>
      </div>
      {items && Array.isArray(items) && items.length > 0 && (
        <ul className="space-y-0.5">
          {items.slice(0, 8).map((item, i) => (
            <li key={item.id ? String(item.id) : i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400/40 shrink-0" />
              <span className="font-medium text-foreground">{String(item.name || `变体${i + 1}`)}</span>
              {item.itemType != null && (
                <span className="text-[10px] text-muted-foreground/50">{String(item.itemType)}</span>
              )}
            </li>
          ))}
          {items.length > 8 && (
            <li className="text-[10px] text-muted-foreground/50 pl-3">…还有 {items.length - 8} 个</li>
          )}
        </ul>
      )}
    </div>
  );
}

/** 项目信息结果 — get_project / query_project */
function ProjectInfoResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const name = obj.name as string | undefined;
  const description = obj.description as string | undefined;

  return (
    <div className="space-y-1.5">
      <span className="text-xs font-medium text-foreground">
        {name || "未命名项目"}
      </span>
      {description && (
        <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
          {description.length > 200 ? description.slice(0, 200) + "…" : description}
        </p>
      )}
    </div>
  );
}

/** 保存分集结果 — save_episode */
function SaveEpisodeResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const message = obj.message as string | undefined;
  const episodeNumber = obj.episodeNumber as number | undefined;
  const title = obj.title as string | undefined;
  const episodeId = obj.episodeId;
  const version = obj.episode_version;

  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
        <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" /> {message || `第${episodeNumber ?? "?"}集「${title || ""}」保存成功`}
      </p>
      <div className="flex gap-3 text-[10px] text-muted-foreground/60">
        {episodeId !== undefined && <span>分集ID: {String(episodeId)}</span>}
        {version !== undefined && <span>版本: v{String(version)}</span>}
      </div>
    </div>
  );
}

/** 创建类操作结果 */
function CreateResult({
  data,
  toolName,
}: {
  data: unknown;
  toolName: string;
}) {
  const obj = data as Record<string, unknown>;
  const status = obj.status as string | undefined;
  const message = obj.message as string | undefined;
  const id = obj.id ?? obj.assetId ?? obj.itemId;
  const name = obj.name as string | undefined;

  const toolLabels: Record<string, string> = {
    create_asset: "资产",
    add_asset_item: "子资产",
    batch_create_asset_items: "批量子资产",
  };
  const label = toolLabels[toolName] || "资源";

  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
        {status === "error" ? <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" /> : <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />}{" "}
        {message || (status === "error" ? `${label}创建失败` : `${label}创建成功`)}
      </p>
      {name && (
        <p className="text-[10px] text-foreground/80">名称: {name}</p>
      )}
      {id !== undefined && (
        <p className="text-[10px] text-muted-foreground/60">
          {label}ID: {String(id)}
        </p>
      )}
    </div>
  );
}

/** 通用结果展示 */
/** 分镜相关结果 — 查询/保存/插入 */
function StoryboardResult({ data, toolName }: { data: unknown; toolName: string }) {
  const obj = data as Record<string, unknown>;

  // 保存类操作（有 status + message）
  const isSave = toolName.startsWith("save_") || toolName === "insert_storyboard_item";
  if (isSave) {
    const status = obj.status as string | undefined;
    const message = obj.message as string | undefined;
    const sceneId = obj.sceneId as number | undefined;
    const sceneNumber = obj.sceneNumber as string | undefined;
    const shotCount = obj.shotCount as number | undefined;
    const episodeId = obj.episodeId as number | undefined;
    const sceneCount = obj.sceneCount as number | undefined;

    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
          {status === "error"
            ? <AlertTriangle className="h-3.5 w-3.5 text-yellow-500 shrink-0" />
            : <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />}
          {message || (status === "error" ? "保存失败" : "保存成功")}
        </p>
        {(sceneNumber || sceneId !== undefined) && (
          <p className="text-[10px] text-muted-foreground/60">
            {sceneNumber ? `场次 ${sceneNumber}` : `场次 ID ${sceneId}`}
            {shotCount !== undefined && ` · ${shotCount} 个镜头`}
          </p>
        )}
        {episodeId !== undefined && (
          <p className="text-[10px] text-muted-foreground/60">
            集 ID: {episodeId}{sceneCount !== undefined && ` · ${sceneCount} 个场次`}
          </p>
        )}
      </div>
    );
  }

  // 列表查询（list_project_storyboards / list_storyboards）
  const storyboards = obj.storyboards as Array<Record<string, unknown>> | undefined;
  if (storyboards) {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground">
          共 <span className="font-medium text-foreground">{storyboards.length}</span> 个分镜脚本
        </p>
        {storyboards.slice(0, 3).map((sb, i) => (
          <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400/60 shrink-0" />
            <span className="truncate">
              {(sb.title as string) || `分镜 #${sb.id || i + 1}`}
              {sb.totalItems !== undefined && ` · ${sb.totalItems} 项`}
            </span>
          </div>
        ))}
      </div>
    );
  }

  // 单个分镜详情（get_storyboard / query_storyboard）
  const title = obj.title as string | undefined;
  const description = obj.description as string | undefined;
  const totalItems = obj.totalItems as number | undefined;
  const items = obj.items as Array<Record<string, unknown>> | undefined;
  const storyboardId = obj.storyboardId ?? obj.id;
  const shotItems = Array.isArray(items) ? items : [];

  return (
    <div className="space-y-1.5">
      <div className="space-y-0.5">
        {title && (
          <p className="text-xs font-medium text-foreground">{title}</p>
        )}
        <p className="text-[10px] text-muted-foreground/60">
          {storyboardId !== undefined && `ID: ${storyboardId}`}
          {totalItems !== undefined && ` · 共 ${totalItems} 个镜头`}
          {shotItems.length > 0 && ` · 预览 ${Math.min(shotItems.length, 3)} 项`}
        </p>
        {description && (
          <p className="text-[10px] leading-4 text-muted-foreground/70 line-clamp-1">
            {description}
          </p>
        )}
      </div>

      {shotItems.length > 0 ? (
        <div className="space-y-1.5">
          {shotItems.slice(0, 3).map((item, i) => {
            const shotNumber = item.shotNumber ?? item.autoShotNumber ?? i + 1;
            const shotType = item.shotType as string | undefined;
            const cameraMovement = item.cameraMovement as string | undefined;
            const content =
              (item.content as string | undefined) ||
              (item.sceneExpectation as string | undefined) ||
              "（无画面描述）";
            const duration = typeof item.duration === "number" ? item.duration : undefined;
            const imageUrl = resolveMediaUrl(
              (item.generatedImageUrl as string | null | undefined) ||
              (item.imageUrl as string | null | undefined)
            );
            const videoUrl = resolveMediaUrl(
              (item.generatedVideoUrl as string | null | undefined) ||
              (item.videoUrl as string | null | undefined)
            );
            const hasImage = !!imageUrl;
            const hasVideo = !!videoUrl;

            return (
              <div
                key={String(item.id ?? i)}
                className="rounded-lg border border-border/25 bg-background/50 px-2 py-1.5"
              >
                <div className="flex gap-2">
                  <div className="relative h-10 w-16 shrink-0 overflow-hidden rounded-md border border-border/20 bg-muted/20">
                    {hasVideo ? (
                      <>
                        <video
                          src={videoUrl || ""}
                          className="h-full w-full object-cover"
                          muted
                          playsInline
                          preload="metadata"
                        />
                        <div className="absolute inset-0 flex items-center justify-center bg-black/15">
                          <Clapperboard className="h-3.5 w-3.5 text-white/90" />
                        </div>
                      </>
                    ) : hasImage ? (
                      <>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl || ""}
                          alt={`镜头 ${String(shotNumber)}`}
                          className="h-full w-full object-cover"
                        />
                      </>
                    ) : (
                      <div className="flex h-full w-full items-center justify-center">
                        <Film className="h-4 w-4 text-muted-foreground/35" />
                      </div>
                    )}
                  </div>

                  <div className="min-w-0 flex-1 space-y-0.5">
                    <div className="flex flex-wrap items-center gap-1">
                      <span className="text-xs font-semibold text-foreground">
                        #{String(shotNumber)}
                      </span>
                      {shotType && (
                        <span className="rounded bg-muted/50 px-1.5 py-0.5 text-[10px] text-muted-foreground">
                          {storyboardShotTypeNames[shotType] || shotType}
                        </span>
                      )}
                      {cameraMovement && (
                        <span className="rounded bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-400">
                          {cameraMovement}
                        </span>
                      )}
                      {duration !== undefined && (
                        <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400">
                          {duration}s
                        </span>
                      )}
                    </div>

                    <p className="text-[10px] leading-4 text-muted-foreground/85 line-clamp-2">
                      {content}
                    </p>

                    <div className="flex flex-wrap items-center gap-1">
                      {hasImage && (
                        <span className="text-[10px] text-cyan-400/80">有画面</span>
                      )}
                      {hasVideo && (
                        <span className="text-[10px] text-emerald-400/80">已有视频</span>
                      )}
                      {!hasImage && !hasVideo && (
                        <span className="text-[10px] text-muted-foreground/60">暂无媒体</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {shotItems.length > 3 && (
            <p className="pl-1 text-[10px] text-muted-foreground/60">
              …还有 {shotItems.length - 3} 个镜头未展开
            </p>
          )}
        </div>
      ) : (
        <p className="text-[11px] text-muted-foreground/70">暂无镜头数据</p>
      )}
    </div>
  );
}

/** 通用字段中文映射 */
const fieldLabels: Record<string, string> = {
  status: "状态", message: "信息", total: "总数", id: "ID", name: "名称",
  type: "类型", count: "数量", title: "标题", projectId: "项目ID",
  storyboardId: "分镜ID", sceneId: "场次ID", episodeId: "集ID",
  sceneNumber: "场次号", shotCount: "镜头数", sceneCount: "场次数",
  totalItems: "总项数", totalEpisodes: "总集数", genre: "类型",
  description: "描述", content: "内容", url: "链接",
  createdAt: "创建时间", updatedAt: "更新时间",
  storyboards: "分镜列表", items: "子项", episodes: "集列表",
  assets: "资产列表", scripts: "剧本列表", projects: "项目列表",
};

function getFieldLabel(key: string): string {
  return fieldLabels[key] || key;
}

function GenericResult({ data }: { data: unknown }) {
  if (typeof data !== "object" || data === null) {
    return (
      <p className="text-xs text-muted-foreground">
        {formatResultValue(data)}
      </p>
    );
  }

  // 有 status + message 的对象，优先用简洁的状态展示
  const obj = data as Record<string, unknown>;
  if (typeof obj.status === "string" && typeof obj.message === "string") {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
          {obj.status === "error" || obj.status === "failed"
            ? <AlertTriangle className="h-3.5 w-3.5 text-yellow-500 shrink-0" />
            : <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />}
          {obj.message as string}
        </p>
        {Object.entries(obj)
          .filter(([k]) => k !== "status" && k !== "message")
          .slice(0, 4)
          .map(([key, val]) => (
            <p key={key} className="text-[10px] text-muted-foreground/60">
              {getFieldLabel(key)}: {formatResultValue(val)}
            </p>
          ))}
      </div>
    );
  }

  // 数组直接展示
  if (Array.isArray(data)) {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground">
          返回 <span className="font-medium text-foreground">{data.length}</span> 条记录
        </p>
        {data.slice(0, 5).map((item, i) => (
          <div
            key={i}
            className="flex items-center gap-2 text-xs text-muted-foreground/90"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 shrink-0" />
            <span>
              {typeof item === "object" && item !== null
                ? String(
                    (item as Record<string, unknown>).name ||
                      (item as Record<string, unknown>).title ||
                      (item as Record<string, unknown>).id ||
                      JSON.stringify(item).slice(0, 80)
                  )
                : formatResultValue(item)}
            </span>
          </div>
        ))}
        {data.length > 5 && (
          <p className="text-[10px] text-muted-foreground/60 pl-3">
            …还有 {data.length - 5} 条
          </p>
        )}
      </div>
    );
  }

  // 对象：展示关键字段
  const entries = Object.entries(obj);
  const priorityKeys = ["status", "message", "total", "id", "name", "title", "type", "count"];
  const sortedEntries = entries.sort((a, b) => {
    const ai = priorityKeys.indexOf(a[0]);
    const bi = priorityKeys.indexOf(b[0]);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return 0;
  });

  const displayEntries = sortedEntries.filter(([, v]) => {
    if (typeof v === "string" && v.length > 300) return false;
    return true;
  });

  return (
    <div className="space-y-0.5">
      {displayEntries.slice(0, 8).map(([key, val]) => (
        <div key={key} className="flex items-baseline gap-2 text-xs">
          <span className="text-muted-foreground/70 shrink-0">{getFieldLabel(key)}:</span>
          <span className="text-muted-foreground">{formatResultValue(val)}</span>
        </div>
      ))}
      {displayEntries.length > 8 && (
        <p className="text-[10px] text-muted-foreground/60">
          …还有 {displayEntries.length - 8} 个字段
        </p>
      )}
    </div>
  );
}

// ========== 将历史消息转换为统一时间线格式 ==========

function pushReasoningToTimeline(
  timeline: TimelineItem[],
  text: string,
  durationMs?: number
) {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "reasoning") {
    last.text += text;
    if (durationMs !== undefined) {
      last.durationMs = durationMs;
    }
    return;
  }
  timeline.push({
    type: "reasoning",
    text,
    ...(durationMs !== undefined ? { durationMs } : {}),
  });
}

function pushContentToTimeline(timeline: TimelineItem[], text: string) {
  const last = timeline[timeline.length - 1];
  if (last && last.type === "content") {
    last.text += text;
    return;
  }
  timeline.push({ type: "content", text });
}

function updateLastTimelineReasoningDuration(
  timeline: TimelineItem[],
  durationMs: number
) {
  for (let index = timeline.length - 1; index >= 0; index--) {
    const item = timeline[index];
    if (item.type === "reasoning") {
      item.durationMs = durationMs;
      return;
    }
  }
}

function pushReasoningToSubTimeline(
  children: SubTimelineItem[],
  text: string,
  durationMs?: number
) {
  const last = children[children.length - 1];
  if (last && last.type === "reasoning") {
    last.text += text;
    if (durationMs !== undefined) {
      last.durationMs = durationMs;
    }
    return;
  }
  children.push({
    type: "reasoning",
    text,
    ...(durationMs !== undefined ? { durationMs } : {}),
  });
}

function pushContentToSubTimeline(children: SubTimelineItem[], text: string) {
  const last = children[children.length - 1];
  if (last && last.type === "content") {
    last.text += text;
    return;
  }
  children.push({ type: "content", text });
}

function updateLastSubTimelineReasoningDuration(
  children: SubTimelineItem[],
  durationMs: number
) {
  for (let index = children.length - 1; index >= 0; index--) {
    const item = children[index];
    if (item.type === "reasoning") {
      item.durationMs = durationMs;
      return;
    }
  }
}

function messagesToTimeline(messages: AgentMessage[]): MessageTimelineProps["timeline"] {
  const timeline: MessageTimelineProps["timeline"] = [];
  // toolCallId → 已建 timeline 中对应 tool item 的索引，用于 TOOL_FINISHED 更新和子 Agent 归属
  const toolIndexMap = new Map<string, number>();

  const appendToParentChildren = (
    parentToolCallId: string,
    updater: (children: SubTimelineItem[]) => void
  ) => {
    const parentIdx = toolIndexMap.get(parentToolCallId);
    if (parentIdx === undefined) return false;

    const parentItem = timeline[parentIdx];
    if (parentItem.type !== "tool") return false;

    if (!parentItem.children) {
      parentItem.children = [];
    }

    updater(parentItem.children);
    return true;
  };

  for (const msg of messages) {
    if (msg.role === "tool") {
      const toolCallId = msg.toolCallId || `hist-tool-${msg.id}`;

      if (msg.parentToolCallId) {
        // 子 Agent 内部的工具调用 → 归入父工具的 children
        appendToParentChildren(msg.parentToolCallId, (children) => {
          if (msg.toolStatus === "running") {
            children.push({
              type: "tool",
              id: toolCallId,
              name: msg.toolName || "tool",
              arguments: msg.content || "",
              status: "calling",
            });
            return;
          }

          const existingChild = children.find(
            (child) => child.type === "tool" && child.id === toolCallId
          );
          if (existingChild && existingChild.type === "tool") {
            existingChild.status = msg.toolStatus === "error" ? "error" : "done";
            existingChild.result = msg.content;
            return;
          }

          children.push({
            type: "tool",
            id: toolCallId,
            name: msg.toolName || "tool",
            arguments: "",
            status: msg.toolStatus === "error" ? "error" : "done",
            result: msg.content,
          });
        });
      } else {
        // 主 Agent 的工具调用
        if (msg.toolStatus === "running") {
          // 工具调用发起
          const idx = timeline.length;
          timeline.push({
            type: "tool",
            id: toolCallId,
            name: msg.toolName || "tool",
            arguments: msg.content || "",
            status: "calling",
          });
          toolIndexMap.set(toolCallId, idx);
        } else {
          // 工具调用结果：查找已有的 calling 记录并更新
          const existingIdx = toolIndexMap.get(toolCallId);
          if (existingIdx !== undefined) {
            const existingItem = timeline[existingIdx];
            if (existingItem.type === "tool") {
              existingItem.status = msg.toolStatus === "error" ? "error" : "done";
              existingItem.result = msg.content;
            }
          } else {
            // 没有对应的 calling 记录（兼容旧数据），直接添加完成记录
            const idx = timeline.length;
            timeline.push({
              type: "tool",
              id: toolCallId,
              name: msg.toolName || "tool",
              arguments: "",
              status: msg.toolStatus === "error" ? "error" : "done",
              result: msg.content,
            });
            toolIndexMap.set(toolCallId, idx);
          }
        }
      }
    } else {
      if (msg.parentToolCallId) {
        appendToParentChildren(msg.parentToolCallId, (children) => {
          if (msg.reasoningContent) {
            pushReasoningToSubTimeline(
              children,
              msg.reasoningContent,
              msg.reasoningDurationMs
            );
          } else if (msg.reasoningDurationMs !== undefined) {
            updateLastSubTimelineReasoningDuration(
              children,
              msg.reasoningDurationMs
            );
          }

          if (msg.content) {
            if (msg.reasoningDurationMs !== undefined) {
              updateLastSubTimelineReasoningDuration(
                children,
                msg.reasoningDurationMs
              );
            }
            pushContentToSubTimeline(children, msg.content);
          }
        });
      } else {
        if (msg.reasoningContent) {
          pushReasoningToTimeline(
            timeline,
            msg.reasoningContent,
            msg.reasoningDurationMs
          );
        } else if (msg.reasoningDurationMs !== undefined) {
          updateLastTimelineReasoningDuration(timeline, msg.reasoningDurationMs);
        }

        if (msg.content) {
          if (msg.reasoningDurationMs !== undefined) {
            updateLastTimelineReasoningDuration(timeline, msg.reasoningDurationMs);
          }
          pushContentToTimeline(timeline, msg.content);
        }
      }
    }
  }
  return timeline;
}

// ========== 历史消息详情面板（右侧） ==========

function HistoryDetailPanel({
  conversation,
}: {
  conversation: AgentConversation;
}) {
  const [messageState, setMessageState] = useState<{
    conversationId: string;
    messages: AgentMessage[];
  } | null>(null);
  const loading = messageState?.conversationId !== conversation.conversationId;
  const messages =
    messageState?.conversationId === conversation.conversationId
      ? messageState.messages
      : [];

  useEffect(() => {
    let cancelled = false;
    listMessages(conversation.conversationId)
      .then((msgs) => {
        if (!cancelled) {
          setMessageState({
            conversationId: conversation.conversationId,
            messages: msgs,
          });
        }
      })
      .catch((err) => {
        console.error(err);
        if (!cancelled) {
          setMessageState({
            conversationId: conversation.conversationId,
            messages: [],
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [conversation.conversationId]);

  // Pipeline 类型：过滤掉用户消息（根据 category 或 agentType 判断）
  const isPipeline =
    conversation.category === "pipeline" ||
    (conversation.agentType != null &&
      (PIPELINE_AGENT_TYPES as readonly string[]).includes(conversation.agentType));
  const displayMessages = isPipeline
    ? messages.filter((msg) => msg.role !== "user")
    : messages;

  // 提取第一条 assistant 消息的 reasoning 信息
  const firstAssistant = displayMessages.find((m) => m.role === "assistant" && m.reasoningContent);

  // 转换为统一时间线格式
  const timeline = messagesToTimeline(displayMessages);

  return (
    <div className="flex flex-col h-full">
      {/* 标题 */}
      <div className="px-4 py-3 border-b border-border/20 shrink-0">
        <h4 className="text-sm font-semibold truncate">
          {conversation.title}
        </h4>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          {conversation.agentType && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-medium">
              {getAgentTypeName(conversation.agentType)}
            </span>
          )}
          {conversation.createTime && conversation.lastMessageTime && (() => {
            const dur = new Date(conversation.lastMessageTime).getTime() - new Date(conversation.createTime).getTime();
            return dur > 0 ? (
              <span className="text-xs text-muted-foreground">
                耗时 {formatElapsed(dur)}
              </span>
            ) : null;
          })()}
          <span className="text-xs text-muted-foreground">
            {formatDatetime(conversation.createTime)}
          </span>
        </div>
      </div>

      {/* 消息内容 — 与运行中面板使用相同的 MessageTimeline 组件 */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : displayMessages.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-8">
            暂无消息记录
          </p>
        ) : (
          <MessageTimeline
            reasoningText={firstAssistant?.reasoningContent || undefined}
            reasoningDurationMs={firstAssistant?.reasoningDurationMs || undefined}
            timeline={timeline}
            streaming={false}
          />
        )}
      </div>
    </div>
  );
}

// ========== 左侧任务列表（小面板用） ==========

function PipelineTaskCard({ task }: { task: PipelineTask }) {
  const { setPanelExpanded, setExpandedTaskId } =
    usePipelineStore();

  const isRunning = task.status === "running";

  const statusIcon = {
    running: <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />,
    done: <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />,
    error: <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />,
    cancelled: <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />,
  };

  const statusText = {
    running: "运行中",
    done: "已完成",
    error: "出错",
    cancelled: "已取消",
  };

  const getLatestActivity = (): string => {
    const timeline = task.state.timeline;
    for (let i = timeline.length - 1; i >= 0; i--) {
      const item = timeline[i];
      if (item.type === "reasoning") {
        return "AI 正在思考…";
      }
      if (item.type === "tool") {
        return item.status === "calling"
          ? `正在${getToolDisplayName(item.name)}…`
          : `${getToolDisplayName(item.name)} ✓`;
      }
      if (item.type === "content") {
        return task.status === "running" ? "AI 正在输出…" : "已生成回复";
      }
    }
    if (task.state.reasoningText) return "AI 正在思考…";
    return "准备中…";
  };

  const handleOpenDetail = () => {
    setExpandedTaskId(task.id);
    setPanelExpanded(true);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8, height: 0 }}
      className={cn(
        "rounded-xl border overflow-hidden transition-colors",
        isRunning
          ? "border-blue-500/20 bg-blue-500/5"
          : task.status === "done"
            ? "border-green-500/20 bg-green-500/5"
            : task.status === "error"
              ? "border-destructive/20 bg-destructive/5"
              : "border-border/20 bg-muted/20"
      )}
    >
      <div
        className="flex items-center gap-2 px-3 py-2.5 cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
        onClick={handleOpenDetail}
      >
        {statusIcon[task.status]}
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium truncate">{task.label}</p>
          <p className="text-[10px] text-muted-foreground truncate">
            {isRunning ? getLatestActivity() : statusText[task.status]}
            {" · "}
            <ElapsedText task={task} />
          </p>
        </div>
        <div className="flex items-center shrink-0">
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        </div>
      </div>
    </motion.div>
  );
}

// ========== 大面板：master-detail 布局 ==========

type SelectedItem =
  | { type: "pipeline"; taskId: string }
  | { type: "history"; conversation: AgentConversation };

function ExpandedPanel({ onClose }: { onClose: () => void }) {
  const { tasks, clearCompleted, expandedTaskId } = usePipelineStore();
  const panelRef = useRef<HTMLDivElement>(null);
  const listEndRef = useRef<HTMLDivElement>(null);

  // 选中的项：优先使用 expandedTaskId（从小弹窗点击跳转过来的）
  const [selected, setSelected] = useState<SelectedItem | null>(() => {
    if (expandedTaskId) {
      const target = tasks.find((t) => t.id === expandedTaskId);
      if (target) return { type: "pipeline", taskId: target.id };
    }
    // 默认选中第一个运行中的 pipeline
    const running = tasks.find((t) => t.status === "running");
    return running ? { type: "pipeline", taskId: running.id } : null;
  });

  // 历史对话
  const [conversations, setConversations] = useState<AgentConversation[]>([]);
  const [historyPage, setHistoryPage] = useState(1);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const pageSize = 20;

  const hasMore = conversations.length < historyTotal;

  // 加载历史
  const loadHistory = useCallback(
    async (page: number, append = false) => {
      setHistoryLoading(true);
      try {
        const result = await listConversations({ pageNo: page, pageSize });
        // 过滤掉 status=running 的对话和当前 pipeline 中已有的对话，避免重复显示
        const currentConvIds = new Set(
          usePipelineStore.getState().tasks
            .filter((t) => t.state.conversationId)
            .map((t) => t.state.conversationId!)
        );
        const filtered = result.list.filter(
          (c) => c.status !== "running" && !currentConvIds.has(c.conversationId)
        );
        if (append) {
          setConversations((prev) => [...prev, ...filtered]);
        } else {
          setConversations(filtered);
        }
        setHistoryTotal(result.total - (result.list.length - filtered.length));
        setHistoryPage(page);
      } catch (err) {
        console.error("加载历史对话失败:", err);
      } finally {
        setHistoryLoading(false);
      }
    },
    [pageSize]
  );

  // 初始加载
  useEffect(() => {
    loadHistory(1);
  }, [loadHistory]);

  // 当有 pipeline 任务完成/出错/取消时，自动刷新历史列表 + 清除当前会话
  const prevTasksRef = useRef(tasks);
  useEffect(() => {
    const prevTasks = prevTasksRef.current;
    prevTasksRef.current = tasks;

    // 检查是否有任务从 running 变为 done/error/cancelled
    const justFinished = tasks.some((t) => {
      if (t.status === "running") return false;
      const prev = prevTasks.find((p) => p.id === t.id);
      return prev && prev.status === "running";
    });

    if (justFinished) {
      // 延迟一点让后端完成数据持久化
      const timer = setTimeout(() => {
        loadHistory(1);
        clearCompleted();
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [tasks, loadHistory, clearCompleted]);

  // 滚动加载更多
  useEffect(() => {
    if (!listEndRef.current || !hasMore) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && !historyLoading && hasMore) {
          loadHistory(historyPage + 1, true);
        }
      },
      { threshold: 0.1 }
    );
    observer.observe(listEndRef.current);
    return () => observer.disconnect();
  }, [hasMore, historyLoading, historyPage, loadHistory]);

  // ESC 关闭
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  // 找到选中的 pipeline task
  const selectedPipelineTask =
    selected?.type === "pipeline"
      ? tasks.find((t) => t.id === selected.taskId)
      : null;

  const selectedConversation =
    selected?.type === "history" ? selected.conversation : null;

  const runningTasks = tasks.filter((t) => t.status === "running");
  const completedTasks = tasks.filter((t) => t.status !== "running");

  return createPortal(
    <>
      {/* 遮罩 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-60 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />
      {/* 大面板 */}
      <motion.div
        ref={panelRef}
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className={cn(
          "fixed z-61 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2",
          "w-[900px] max-w-[92vw] h-[75vh]",
          "rounded-2xl border border-border/40",
          "bg-card/98 backdrop-blur-xl",
          "shadow-2xl shadow-black/30",
          "flex flex-col overflow-hidden"
        )}
      >
        {/* 标题栏 */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-border/20 shrink-0">
          <h3 className="text-base font-semibold">AI 任务中心</h3>
          <div className="flex items-center gap-2">
            {completedTasks.length > 0 && (
              <button
                onClick={clearCompleted}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-lg hover:bg-muted"
              >
                清除已完成
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-muted transition-colors"
            >
              <X className="h-4 w-4 text-muted-foreground" />
            </button>
          </div>
        </div>

        {/* master-detail */}
        <div className="flex flex-1 min-h-0">
          {/* 左侧：列表 */}
          <div className="w-72 shrink-0 border-r border-border/20 flex flex-col min-h-0">
            <div className="flex-1 min-h-0 overflow-y-auto">
              {/* 运行中的 pipeline */}
              {runningTasks.length > 0 && (
                <div className="px-3 pt-3 pb-1">
                  <p className="text-[10px] font-medium text-muted-foreground px-1 pb-1.5 uppercase tracking-wider">
                    运行中 ({runningTasks.length})
                  </p>
                  {runningTasks.map((task) => (
                    <TaskListItem
                      key={task.id}
                      label={task.label}
                      subtitle={<ElapsedText task={task} />}
                      statusColor="text-blue-400"
                      icon={<Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400 shrink-0" />}
                      selected={
                        selected?.type === "pipeline" &&
                        selected.taskId === task.id
                      }
                      onClick={() =>
                        setSelected({ type: "pipeline", taskId: task.id })
                      }
                    />
                  ))}
                </div>
              )}

              {/* 已完成的 pipeline */}
              {completedTasks.length > 0 && (
                <div className="px-3 pt-2 pb-1">
                  <p className="text-[10px] font-medium text-muted-foreground px-1 pb-1.5 uppercase tracking-wider">
                    当前会话 ({completedTasks.length})
                  </p>
                  {completedTasks.map((task) => (
                    <TaskListItem
                      key={task.id}
                      label={task.label}
                      subtitle={
                        (task.status === "done"
                          ? "已完成"
                          : task.status === "error"
                            ? "出错"
                            : "已取消") +
                        " · " +
                        getElapsedStr(task)
                      }
                      statusColor={
                        task.status === "done"
                          ? "text-green-400"
                          : task.status === "error"
                            ? "text-destructive"
                            : "text-muted-foreground"
                      }
                      icon={
                        task.status === "done" ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
                        ) : task.status === "error" ? (
                          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                        ) : (
                          <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                        )
                      }
                      selected={
                        selected?.type === "pipeline" &&
                        selected.taskId === task.id
                      }
                      onClick={() =>
                        setSelected({ type: "pipeline", taskId: task.id })
                      }
                    />
                  ))}
                </div>
              )}

              {/* 历史记录 */}
              <div className="px-3 pt-2 pb-3">
                <p className="text-[10px] font-medium text-muted-foreground px-1 pb-1.5 uppercase tracking-wider flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  历史记录
                </p>
                {conversations.length === 0 && !historyLoading && (
                  <p className="text-xs text-muted-foreground/60 px-1 py-3 text-center">
                    暂无历史记录
                  </p>
                )}
                {conversations.map((conv) => {
                  const isError = conv.status === "error" || conv.status === "failed";
                  const isDone = conv.status === "completed" || conv.status === "done";
                  const isCancelled = conv.status === "cancelled";
                  return (
                    <TaskListItem
                      key={conv.id}
                      label={conv.title}
                      subtitle={
                        <>
                          {conv.agentType ? getAgentTypeName(conv.agentType) : "对话"}
                          {isError ? " · 出错" : isCancelled ? " · 已取消" : isDone ? " · 已完成" : ""}
                          {conv.createTime && conv.lastMessageTime && (() => {
                            const dur = new Date(conv.lastMessageTime).getTime() - new Date(conv.createTime).getTime();
                            return dur > 0 ? ` · ${formatElapsed(dur)}` : "";
                          })()}
                        </>
                      }
                      statusColor={
                        isError
                          ? "text-destructive"
                          : isDone
                            ? "text-green-400"
                            : isCancelled
                              ? "text-muted-foreground"
                              : "text-muted-foreground"
                      }
                      icon={
                        isError ? (
                          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                        ) : isDone ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
                        ) : isCancelled ? (
                          <Ban className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                        ) : (
                          <MessageSquare className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0" />
                        )
                      }
                      selected={
                        selected?.type === "history" &&
                        selected.conversation.id === conv.id
                      }
                      onClick={() =>
                        setSelected({ type: "history", conversation: conv })
                      }
                    />
                  );
                })}

                {/* 加载更多触发器 */}
                <div ref={listEndRef} className="h-1" />
                {historyLoading && (
                  <div className="flex items-center justify-center py-3">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 右侧：详情 */}
          <div className="flex-1 min-w-0 min-h-0 bg-muted/10">
            {selectedPipelineTask ? (
              <PipelineDetailPanel task={selectedPipelineTask} />
            ) : selectedConversation ? (
              <HistoryDetailPanel conversation={selectedConversation} />
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                <div className="text-center">
                  <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-30" />
                  <p className="text-sm">选择一个任务查看详情</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </>,
    document.body
  );
}

/** 列表条目 */
function TaskListItem({
  label,
  subtitle,
  icon,
  selected,
  onClick,
}: {
  label: string;
  subtitle: React.ReactNode;
  statusColor: string;
  icon: React.ReactNode;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-colors mb-0.5",
        selected
          ? "bg-foreground/10"
          : "hover:bg-foreground/5"
      )}
    >
      {icon}
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium truncate">{label}</p>
        <p className="text-[10px] text-muted-foreground truncate">{subtitle}</p>
      </div>
    </button>
  );
}

// ========== 主组件 ==========

interface NotificationPanelProps {
  anchorRef: React.RefObject<HTMLButtonElement | null>;
}

export function NotificationPanel({ anchorRef }: NotificationPanelProps) {
  const { tasks, notificationOpen, setNotificationOpen, clearCompleted, panelExpanded, setPanelExpanded } =
    usePipelineStore();

  const [pos, setPos] = useState({ top: 0, right: 0 });
  const panelRef = useRef<HTMLDivElement>(null);

  const completedTasks = tasks.filter((t) => t.status !== "running");
  const runningTasks = tasks.filter((t) => t.status === "running");

  // 计算面板位置（仅小模式）
  useEffect(() => {
    if (notificationOpen && anchorRef.current && !panelExpanded) {
      const rect = anchorRef.current.getBoundingClientRect();
      setPos({
        top: rect.bottom + 8,
        right: window.innerWidth - rect.right,
      });
    }
  }, [notificationOpen, anchorRef, panelExpanded]);

  // 点击外部关闭（仅小模式）
  useEffect(() => {
    if (!notificationOpen || panelExpanded) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(e.target as Node) &&
        anchorRef.current &&
        !anchorRef.current.contains(e.target as Node)
      ) {
        setNotificationOpen(false);
      }
    };
    const timer = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
    }, 10);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [notificationOpen, panelExpanded, anchorRef, setNotificationOpen]);

  // 关闭通知时重置展开状态
  useEffect(() => {
    if (!notificationOpen) setPanelExpanded(false);
  }, [notificationOpen, setPanelExpanded]);

  if (!notificationOpen) return null;

  // ========== 大面板模式 ==========
  if (panelExpanded) {
    return (
      <ExpandedPanel
        onClose={() => {
          setPanelExpanded(false);
          setNotificationOpen(false);
        }}
      />
    );
  }

  // ========== 小面板模式 ==========
  return createPortal(
    <motion.div
      ref={panelRef}
      initial={{ opacity: 0, y: -8, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.96 }}
      transition={{ duration: 0.15, ease: "easeOut" }}
      className={cn(
        "fixed z-61 w-80",
        "rounded-2xl border border-border/40",
        "bg-card/95 backdrop-blur-xl",
        "shadow-2xl shadow-black/20",
        "overflow-hidden"
      )}
      style={{ top: pos.top, right: pos.right }}
    >
      {/* 标题栏 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/20">
        <h3 className="text-sm font-semibold">AI 任务</h3>
        <div className="flex items-center gap-1">
          {completedTasks.length > 0 && (
            <button
              onClick={clearCompleted}
              className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded-lg hover:bg-muted"
            >
              清除已完成
            </button>
          )}
          <button
            onClick={() => setPanelExpanded(true)}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
            title="展开大面板"
          >
            <Maximize2 className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
          <button
            onClick={() => setNotificationOpen(false)}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
        </div>
      </div>

      {/* 内容 */}
      <div className="max-h-[60vh] overflow-y-auto">
        {tasks.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">暂无 AI 任务</p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              使用 AI 功能后，任务进度会显示在这里
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {runningTasks.length > 0 && (
              <div>
                <p className="text-[10px] font-medium text-muted-foreground px-2 py-1 uppercase tracking-wider">
                  运行中 ({runningTasks.length})
                </p>
                <div className="space-y-1.5">
                  <AnimatePresence>
                    {runningTasks.map((task) => (
                      <PipelineTaskCard key={task.id} task={task} />
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            )}
            {completedTasks.length > 0 && (
              <div>
                <p className="text-[10px] font-medium text-muted-foreground px-2 py-1 uppercase tracking-wider">
                  已完成 ({completedTasks.length})
                </p>
                <div className="space-y-1.5">
                  <AnimatePresence>
                    {completedTasks.map((task) => (
                      <PipelineTaskCard key={task.id} task={task} />
                    ))}
                  </AnimatePresence>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>,
    document.body
  );
}
