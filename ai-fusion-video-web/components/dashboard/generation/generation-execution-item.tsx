"use client";

import { Button } from "@/components/ui/button";
import { ExecutionRecordButton } from "@/components/dashboard/notification-panel/execution-record-button";
import type { GenerationExecution, GenerationExecutionItem, GenerationReviewAction } from "@/lib/api/generation-execution";
import { generationPhaseLabels, generationReviewLabels, generationStateLabels } from "@/lib/generation-execution-view";
import { cn } from "@/lib/utils";
import { executionGuidance, executionProviderTask, executionTime } from "./generation-execution-presentation";

export function GenerationExecutionItemSection({ item, data, busy, onRetry, onClose }: {
  item: GenerationExecutionItem;
  data: GenerationExecution;
  busy: boolean;
  onRetry: () => void;
  onClose: () => void;
}) {
  const uncertain = item.state === "UNKNOWN_SUBMISSION";
  const guidance = executionGuidance(item);
  return (
    <section className="flex min-w-0 flex-col gap-3 border-t border-border/20 pt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium">
          {item.itemIds.length === 1 ? `结果 ${item.index + 1}` : `结果 ${item.index + 1}–${item.index + item.itemIds.length}`}
        </h3>
        <span className={cn("text-xs", uncertain ? "text-amber-700 dark:text-amber-300" : "text-muted-foreground")}>
          {uncertain ? "受理待确认" : generationStateLabels[item.state] ?? "状态无法识别"}
          {item.phase !== "FINISHED" && ` · ${generationPhaseLabels[item.phase] ?? "阶段无法识别"}`}
        </span>
      </div>
      {guidance && <p className="text-sm leading-6">{guidance}</p>}
      <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-xs text-muted-foreground">最后更新</dt>
          <dd className="mt-1">{executionTime(item.updatedAt)}</dd>
        </div>
        <div className="min-w-0">
          <dt className="text-xs text-muted-foreground">供应商任务编号</dt>
          <dd className="mt-1 break-all">{executionProviderTask(item)}</dd>
        </div>
      </dl>
      <div className="flex flex-col gap-3">
        <h4 className="text-xs font-medium text-muted-foreground">提交记录</h4>
        {item.attempts.length === 0 && <p className="text-sm text-muted-foreground">暂无提交记录</p>}
        {item.attempts.map((attempt) => (
          <div key={attempt.id} className="flex flex-col gap-2">
            {item.attempts.length > 1 && <p className="text-sm">第 {attempt.number} 次提交</p>}
            <dl className="grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
              <div><dt className="text-muted-foreground">准备开始</dt><dd className="mt-1">{executionTime(attempt.createdAt)}</dd></div>
              <div><dt className="text-muted-foreground">发起提交</dt><dd className="mt-1">{attempt.submittedAt ? executionTime(attempt.submittedAt) : "尚未提交"}</dd></div>
              <div><dt className="text-muted-foreground">收到回执</dt><dd className="mt-1">{attempt.acknowledgedAt ? executionTime(attempt.acknowledgedAt) : "未收到"}</dd></div>
            </dl>
            {attempt.resolution && (
              <p className="text-xs text-muted-foreground">
                {generationReviewLabels[attempt.resolution as GenerationReviewAction] ?? "已处理"} · {executionTime(attempt.resolvedAt)}
                {attempt.resolutionNote && <span className="mt-1 block break-words">{attempt.resolutionNote}</span>}
              </p>
            )}
          </div>
        ))}
      </div>
      {(item.availableActions.includes("RETRY_INGEST") || ["UNKNOWN_SUBMISSION", "FAILED", "NEEDS_REVIEW"].includes(item.state)) && (
        <div className="flex flex-wrap gap-2">
          {item.availableActions.includes("RETRY_INGEST") && <Button variant="outline" size="sm" disabled={busy} onClick={onRetry}>重新保存结果</Button>}
          {["UNKNOWN_SUBMISSION", "FAILED", "NEEDS_REVIEW"].includes(item.state) && <ExecutionRecordButton target={{ kind: "GENERATION", id: item.id }} onOpen={onClose} />}
        </div>
      )}
      <details open={item.state === "FAILED"} className="text-xs">
        <summary className="cursor-pointer rounded-lg py-1 text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring">诊断信息</summary>
        <dl className="mt-2 grid min-w-0 grid-cols-1 gap-3 rounded-lg border border-border/20 bg-background/70 p-3 sm:grid-cols-2">
          {[
            ["任务编号", data.taskId], ["执行编号", item.id], ["模型标识", data.modelCode],
            ["请求协议", data.modelProtocol ?? "—"], ["错误代码", item.errorCode ?? "—"], ["进度查询", `${item.pollAttempts} 次`],
          ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-muted-foreground">{label}</dt><dd className="mt-1 break-all">{value}</dd></div>)}
          {item.errorMessage && <div className="min-w-0 sm:col-span-2"><dt className="text-muted-foreground">错误信息</dt><dd className="mt-1 whitespace-pre-wrap break-words">{item.errorMessage}</dd></div>}
        </dl>
      </details>
    </section>
  );
}
