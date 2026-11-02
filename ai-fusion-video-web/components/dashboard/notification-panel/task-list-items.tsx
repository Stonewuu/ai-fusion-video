"use client";

import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Ban, CheckCircle2, ChevronRight, Loader2, XCircle } from "lucide-react";
import { usePipelineStore, type PipelineTask } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import { getToolDisplayName } from "./constants";
import { ElapsedText } from "./hooks";

export function TaskListItem({
  label,
  subtitle,
  icon,
  selected,
  onClick,
}: {
  label: string;
  subtitle: ReactNode;
  icon: ReactNode;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-colors mb-0.5 outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none",
        selected ? "bg-foreground/10" : "hover:bg-foreground/5"
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

export function PipelineTaskCard({ task }: { task: PipelineTask }) {
  const reducedMotion = useReducedMotion();
  const setPanelExpanded = usePipelineStore((state) => state.setPanelExpanded);
  const setExpandedTaskId = usePipelineStore((state) => state.setExpandedTaskId);
  const isRunning = task.status === "running";
  const cancelling = !!task.state.cancelRequested;

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

  const getLatestActivity = () => {
    if (cancelling) {
      return "取消中…";
    }
    const timeline = task.state.timeline;
    const isTaskStream = task.cancellable === false;
    for (let index = timeline.length - 1; index >= 0; index--) {
      const item = timeline[index];
      if (item.type === "reasoning") {
        return "AI 正在思考…";
      }
      if (item.type === "tool") {
        return item.status === "calling"
          ? `正在${getToolDisplayName(item.name)}…`
          : `${getToolDisplayName(item.name)} 已完成`;
      }
      if (item.type === "content") {
        if (isTaskStream) {
          return task.status === "running" ? "任务执行中…" : "已记录任务结果";
        }
        return task.status === "running" ? "AI 正在输出…" : "已生成回复";
      }
    }

    if (task.state.reasoningText) {
      return "AI 正在思考…";
    }
    return isTaskStream ? "任务准备中…" : "准备中…";
  };

  const handleOpenDetail = () => {
    setExpandedTaskId(task.id);
    setPanelExpanded(true);
  };

  return (
    <motion.div
      layout={!reducedMotion}
      transition={{ duration: reducedMotion ? 0 : 0.2, ease: "easeOut" }}
      initial={reducedMotion ? false : { opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={reducedMotion ? undefined : { opacity: 0, y: -8, height: 0 }}
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
