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
import { Textarea } from "@/components/ui/textarea";
import type {
  ComfyUiTestAction,
  ComfyUiTestRun,
} from "@/lib/api/comfyui-workflow";
import {
  comfyUiTestActive,
  comfyUiTestStatus,
} from "@/lib/comfyui-test-request";
import { ComfyUiTestOutput } from "./comfyui-test-output";

const actions: Record<ComfyUiTestAction, string> = {
  RECHECK: "查询原任务",
  RETRY_INGEST: "继续保存结果",
  CONFIRM_FINISHED: "确认远端已结束",
};

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
  const [decision, setDecision] = useState<{
    run: ComfyUiTestRun;
    action: ComfyUiTestAction | "CANCEL";
  } | null>(null);
  const [note, setNote] = useState("");
  const choose = (
    run: ComfyUiTestRun,
    action: ComfyUiTestAction | "CANCEL",
  ) => {
    setDecision({ run, action });
    setNote("");
  };
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
                    onClick={() => choose(run, "CANCEL")}
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
            {run.permitHeld && run.status === "NEEDS_REVIEW" && (
              <p className="text-xs text-muted-foreground">
                仍保留此供应商的试运行占用，核实远端状态后才能继续。
              </p>
            )}
            {run.availableActions.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {run.availableActions.map((action) => (
                  <Button
                    key={action}
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    onClick={() => {
                      if (action === "CONFIRM_FINISHED") choose(run, action);
                      else void onReview(run, action, action === "RECHECK" ? "用户请求继续查询原任务" : "用户请求继续保存原任务结果");
                    }}
                  >
                    {actions[action]}
                  </Button>
                ))}
              </div>
            )}
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
        <Dialog
          open
          onOpenChange={(open) => {
            if (!open && !busy) setDecision(null);
          }}
        >
          <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden">
            <DialogHeader className="shrink-0">
              <DialogTitle>
                {decision.action === "CANCEL"
                  ? "取消试运行"
                  : actions[decision.action]}
              </DialogTitle>
              <DialogDescription>
                {decision.action === "CANCEL"
                  ? "停止尚未提交的任务；已经受理的任务会请求远端取消，实际结束前仍保留占用。"
                  : decision.action === "CONFIRM_FINISHED"
                    ? "请先在 ComfyUI 确认任务已经结束。这会结束本地跟踪并释放占用；再次试运行会创建新的付费请求。"
                    : "使用原任务与输入继续查询或保存结果，不会重新提交工作流。"}
              </DialogDescription>
            </DialogHeader>
            {decision.action !== "CANCEL" && (
              <Textarea
                className="min-h-24 max-h-60 overflow-y-auto"
                aria-label="核实说明"
                value={note}
                maxLength={2000}
                onChange={(event) => setNote(event.target.value)}
                placeholder="记录核实依据"
              />
            )}
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
            <DialogFooter className="shrink-0">
              <Button
                variant="outline"
                disabled={busy}
                onClick={() => setDecision(null)}
              >
                返回
              </Button>
              <Button
                disabled={
                  busy || (decision.action !== "CANCEL" && !note.trim())
                }
                onClick={async () => {
                  const done =
                    decision.action === "CANCEL"
                      ? await onCancel(decision.run)
                      : await onReview(
                          decision.run,
                          decision.action,
                          note.trim(),
                        );
                  if (done) setDecision(null);
                }}
              >
                {busy ? "处理中…" : "确认"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
