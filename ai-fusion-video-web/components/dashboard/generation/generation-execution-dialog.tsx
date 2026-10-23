"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  generationExecutionApi,
  type GenerationExecution,
  type GenerationExecutionItem,
  type GenerationReviewAction,
} from "@/lib/api/generation-execution";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import {
  generationPhaseLabels,
  generationReviewLabels,
  generationStateLabels,
  generationTerminalStates,
} from "@/lib/generation-execution-view";
import { GenerationReviewDialog } from "./generation-review-dialog";

export function GenerationExecutionDialog({
  id,
  onClose,
  onChanged,
}: {
  id: string;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [data, setData] = useState<GenerationExecution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [revision, refresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [decision, setDecision] = useState<{
    item: GenerationExecutionItem;
    action: GenerationReviewAction;
  } | null>(null);
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const read = async () => {
      let complete = false;
      try {
        const result = await generationExecutionApi.get(id);
        if (active) {
          setData(result);
          setError(null);
        }
        complete = generationTerminalStates.has(result.state);
      } catch (failure) {
        if (active) {
          setError(getApiErrorMessage(failure));
          if ([401, 403, 404].includes(getApiErrorStatus(failure) ?? 0)) {
            setData(null);
            complete = true;
          }
        }
      } finally {
        if (active && !complete) timer = setTimeout(() => void read(), 3000);
      }
    };
    void read();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [id, revision]);
  const review = async (
    item: GenerationExecutionItem,
    action: GenerationReviewAction,
    note: string,
  ) => {
    if (busy || error) return false;
    setBusy(true);
    setReviewError(null);
    try {
      await generationExecutionApi.review(item.id, {
        expectedRevision: item.revision,
        action,
        note,
      });
      refresh((value) => value + 1);
      onChanged();
      return true;
    } catch (failure) {
      setReviewError(getApiErrorMessage(failure));
      refresh((value) => value + 1);
      return false;
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <Dialog
        open
        onOpenChange={(open) => {
          if (!open && !busy) onClose();
        }}
      >
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
          <DialogHeader className="shrink-0">
            <DialogTitle>生成执行详情</DialogTitle>
            <DialogDescription>
              {data
                ? `${data.modelName} · ${generationStateLabels[data.state] ?? "状态无法识别"}`
                : "正在读取任务状态…"}
            </DialogDescription>
          </DialogHeader>
          <DialogScrollBody className="space-y-4">
            {error && (
              <div
                role="alert"
                className="flex items-center gap-2 text-sm text-destructive"
              >
                <p className="flex-1">{error}</p>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => refresh((value) => value + 1)}
                >
                  <RefreshCw />
                  重试读取
                </Button>
              </div>
            )}
            {reviewError && !decision && (
              <p role="alert" className="text-sm text-destructive">
                {reviewError}
              </p>
            )}
            {data?.items.map((item) => (
              <section
                key={item.id}
                className="space-y-3 border-b border-border/20 pb-4 last:border-b-0"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-sm font-medium">
                    {item.itemIds.length === 1
                      ? `结果 ${item.index + 1}`
                      : `结果 ${item.index + 1}–${item.index + item.itemIds.length}`}
                  </h3>
                  <span className="text-xs text-muted-foreground">
                    {generationStateLabels[item.state] ?? "状态无法识别"} ·{" "}
                    {generationPhaseLabels[item.phase] ?? "阶段无法识别"}
                  </span>
                </div>
                {item.errorMessage && (
                  <p className="break-words text-sm text-destructive">
                    {item.errorMessage}
                  </p>
                )}
                {item.cancelRequested &&
                  !generationTerminalStates.has(item.state) && (
                    <p className="text-sm text-muted-foreground">
                      已请求取消，正在核实供应商是否结束。
                    </p>
                  )}
                {item.remoteCapacityHeld && (
                  <p className="text-xs text-muted-foreground">
                    远端任务尚未确认结束，暂时保留生成名额。
                  </p>
                )}
                {item.availableActions.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {item.availableActions.map((action) => (
                      <Button
                        key={action}
                        variant="outline"
                        size="sm"
                        disabled={busy || !!error}
                        onClick={() => {
                          setReviewError(null);
                          if (
                            action === "CONFIRM_NOT_ACCEPTED_AND_RETRY" ||
                            action === "CONFIRM_PROVIDER_FINISHED"
                          )
                            setDecision({ item, action });
                          else
                            void review(
                              item,
                              action,
                              `用户在执行详情中选择：${generationReviewLabels[action]}`,
                            );
                        }}
                      >
                        {generationReviewLabels[action]}
                      </Button>
                    ))}
                  </div>
                )}
                {item.attempts.length > 0 && (
                  <details className="text-xs text-muted-foreground">
                    <summary className="cursor-pointer rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring">
                      查看提交记录（{item.attempts.length} 次）
                    </summary>
                    <ul className="mt-3 space-y-3">
                      {item.attempts.map((attempt) => (
                        <li key={attempt.id} className="space-y-1">
                          <p>
                            第 {attempt.number} 次 ·{" "}
                            {attempt.resolution
                              ? "已处理"
                              : attempt.acknowledged
                              ? "已收到响应"
                              : attempt.submitted
                                ? "已记录提交，受理待确认"
                                : "正在准备，尚未提交"}
                          </p>
                          {attempt.providerTaskId && (
                            <p className="break-all">
                              供应商任务编号：{attempt.providerTaskId}
                            </p>
                          )}
                          {attempt.resolution && (
                            <p>
                              核实操作：
                              {generationReviewLabels[
                                attempt.resolution as GenerationReviewAction
                              ] ?? attempt.resolution}
                            </p>
                          )}
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </section>
            ))}
          </DialogScrollBody>
        </DialogContent>
      </Dialog>
      {decision && (
        <GenerationReviewDialog
          item={decision.item}
          action={decision.action}
          busy={busy}
          error={reviewError}
          onClose={() => setDecision(null)}
          onConfirm={(note) => review(decision.item, decision.action, note)}
        />
      )}
    </>
  );
}
