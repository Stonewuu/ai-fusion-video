"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { ExecutionRecordButton } from "@/components/dashboard/notification-panel/execution-record-button";
import type {
  ComfyUiTestAction,
  ComfyUiTestRun,
} from "@/lib/api/comfyui-workflow";
import {
  comfyUiTestActive,
  comfyUiTestStatus,
} from "@/lib/comfyui-test-request";
import { ComfyUiTestOutput } from "./comfyui-test-output";

export function ComfyUiTestHistory({
  runs,
  busy,
  error,
  workflowHash,
  onCancel,
  onReview,
}: {
  runs: ComfyUiTestRun[];
  busy: boolean;
  error: string | null;
  workflowHash: string;
  onCancel: (run: ComfyUiTestRun) => Promise<boolean>;
  onReview: (
    run: ComfyUiTestRun,
    action: ComfyUiTestAction,
    note: string,
  ) => Promise<boolean>;
}) {
  const [decision, setDecision] = useState<ComfyUiTestRun | null>(null);
  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium">试运行记录</h4>
      {!runs.length && (
        <p className="text-sm text-muted-foreground">尚未提交试运行。</p>
      )}
      <div className="divide-y divide-border/20">
        {runs.map((run) => (
          <article key={run.id} className="space-y-3 py-4">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <p className="text-sm font-medium" role="status">
                  {comfyUiTestStatus(run)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {new Date(run.createdAt).toLocaleString()} ·{" "}
                  {run.outputs.length} 个结果
                </p>
              </div>
              {(comfyUiTestActive(run) || run.status === "NEEDS_REVIEW") &&
                !run.cancelRequested && (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busy}
                    onClick={() => setDecision(run)}
                  >
                    取消试运行
                  </Button>
                )}
            </div>
            {run.workflowHash !== workflowHash && (
              <p className="text-xs text-muted-foreground">
                此结果来自较早的草稿，当前草稿需要重新校验和试运行。
              </p>
            )}
            {run.inputSummary.prompt && (
              <p className="line-clamp-3 text-sm">{run.inputSummary.prompt}</p>
            )}
            {run.error && (
              <p className="text-sm text-destructive">{run.error}</p>
            )}
            <div className="flex flex-wrap gap-2">
              {run.availableActions.includes("RETRY_INGEST") && (
                <Button variant="outline" size="sm" disabled={busy}
                  onClick={() => void onReview(run, "RETRY_INGEST", "重新保存已收到的结果")}>
                  重新保存结果
                </Button>
              )}
              {["UNKNOWN_SUBMISSION", "FAILED", "NEEDS_REVIEW"].includes(run.status) && (
                <ExecutionRecordButton target={{ kind: "COMFY_TEST", id: run.id }} />
              )}
            </div>
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer rounded-lg focus-visible:outline-2 focus-visible:outline-ring">
                任务信息
              </summary>
              <dl className="grid gap-2 pt-2">
                {Object.entries(run.inputSummary).map(([name, value]) => (
                  <div key={name}>
                    <dt>{name === "prompt" ? "提示词" : name}</dt>
                    <dd className="break-words">{value}</dd>
                  </div>
                ))}
                <div>
                  <dt>任务编号</dt>
                  <dd className="break-all">{run.id}</dd>
                </div>
                <div>
                  <dt>远端编号</dt>
                  <dd className="break-all">{run.promptId ?? "尚未提交"}</dd>
                </div>
                <div>
                  <dt>受理回执</dt>
                  <dd>{run.acknowledged ? "已收到" : "尚未收到直接回执"}</dd>
                </div>
              </dl>
            </details>
            {run.outputs.length > 0 && (
              <div className="grid gap-3 sm:grid-cols-2">
                {run.outputs.map((output, index) => (
                  <ComfyUiTestOutput
                    key={`${output.url}:${index}`}
                    output={output}
                    index={index}
                  />
                ))}
              </div>
            )}
          </article>
        ))}
      </div>
      {decision && (
        <Dialog open onOpenChange={(open) => {
          if (!open && !busy) {
            setDecision(null);
          }
        }}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>取消试运行</DialogTitle>
              <DialogDescription>
                停止尚未提交的任务；已经受理的任务会请求远端取消，超时后结束本地等待。
              </DialogDescription>
            </DialogHeader>
            {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
            <DialogFooter>
              <Button variant="outline" disabled={busy} onClick={() => setDecision(null)}>返回</Button>
              <Button variant="destructive" disabled={busy} onClick={async () => {
                if (await onCancel(decision)) {
                  setDecision(null);
                }
              }}>{busy ? "处理中…" : "取消试运行"}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
