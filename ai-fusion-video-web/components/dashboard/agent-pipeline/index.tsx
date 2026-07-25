"use client";

import { Ban, CheckCircle2, Loader2, XCircle } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useAgentPipeline } from "./use-agent-pipeline";
import { AgentPipelineTimeline } from "./timeline";
import type { AgentPipelineProps } from "./types";

export function AgentPipeline(props: AgentPipelineProps) {
  const { state, isActive, startStream, cancelStream } = useAgentPipeline(props);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isActive && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
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
            {state.status === "cancelling" && "取消中..."}
            {state.status === "done" && "解析完成"}
            {state.status === "error" && "解析出错"}
            {state.status === "cancelled" && "已取消"}
          </span>
        </div>
        {isActive && (
          <button
            onClick={cancelStream}
            disabled={state.status === "cancelling"}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-medium",
              "border border-border/40 hover:bg-destructive/10 hover:text-destructive",
              "transition-colors disabled:pointer-events-none disabled:opacity-60"
            )}
          >
            {state.status === "cancelling" ? "取消中..." : "取消"}
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

      <AgentPipelineTimeline timeline={state.timeline} isActive={isActive} />

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
