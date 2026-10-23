"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import type { useGenerationSubmission } from "./use-generation-submission";

export function GenerationSubmissionNotice({
  submission,
  onConfirmed,
}: {
  submission: ReturnType<typeof useGenerationSubmission>;
  onConfirmed: () => void;
}) {
  const [clearing, setClearing] = useState(false);
  const [withdrawing, setWithdrawing] = useState(false);
  if (submission.ready && !submission.pending && !submission.error) return null;
  return (
    <section
      role="status"
      className="mb-3 space-y-3 rounded-lg border border-border/30 bg-card p-4"
    >
      <div className="space-y-1">
        <h2 className="flex items-center gap-2 text-sm font-medium">
          {submission.busy && (
            <Loader2 className="size-4 animate-spin motion-reduce:animate-none" />
          )}
          {submission.pending
            ? submission.busy
              ? "正在确认提交…"
              : "上次提交需要确认"
            : submission.error
              ? "暂时无法提交"
              : "正在确认账号与团队…"}
        </h2>
        {submission.pending && (
          <p className="text-xs text-muted-foreground">
            {submission.pending.modelName} · {submission.pending.request.count}{" "}
            个结果 ·{" "}
            {new Date(submission.pending.createdAt).toLocaleString("zh-CN")}
          </p>
        )}
        {submission.pending && (
          <p className="line-clamp-2 break-words text-sm">
            {submission.pending.request.prompt}
          </p>
        )}
        {submission.error && (
          <p role="alert" className="text-sm text-destructive">
            {submission.error}
          </p>
        )}
        {submission.pending && (
          <p className="text-xs text-muted-foreground">
            确认会使用原模型、参数和提交标识。当前编辑的内容不会替换这次请求。
          </p>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {submission.pending ? (
          <>
            <Button
              size="sm"
              disabled={submission.busy}
              onClick={() =>
                void submission.retry().then((receipt) => {
                  if (receipt) onConfirmed();
                })
              }
            >
              确认提交结果
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={submission.busy}
              onClick={() => setWithdrawing(true)}
            >
              撤回并修改
            </Button>
          </>
        ) : (
          submission.error && (
            <Button variant="outline" size="sm" onClick={submission.reload}>
              重新读取状态
            </Button>
          )
        )}
        {submission.corrupt && (
          <Button variant="ghost" size="sm" onClick={() => setClearing(true)}>
            清理本地记录
          </Button>
        )}
      </div>
      <Dialog
        open={withdrawing}
        onOpenChange={(open) => {
          if (!submission.busy) setWithdrawing(open);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>撤回这次提交</DialogTitle>
            <DialogDescription>
              尚未受理的请求会被阻止；已受理的任务会请求取消。供应商已经开始的任务可能仍需等待结束，已产生的费用不会因此撤销。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={submission.busy}
              onClick={() => setWithdrawing(false)}
            >
              返回
            </Button>
            <Button
              disabled={submission.busy}
              onClick={() =>
                void submission.withdraw().then((receipt) => {
                  if (receipt) {
                    setWithdrawing(false);
                    onConfirmed();
                  }
                })
              }
            >
              撤回提交
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={clearing} onOpenChange={setClearing}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>清理损坏的本地记录</DialogTitle>
            <DialogDescription>
              这只移除浏览器中的损坏记录，不会取消服务端任务。请先查看创作记录，确认上次任务是否已经受理，再重新生成，避免重复付费。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setClearing(false)}>
              返回核对
            </Button>
            <Button
              onClick={() =>
                void submission
                  .clearCorruptRecord()
                  .then(() => setClearing(false))
              }
            >
              已核对，清理记录
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
