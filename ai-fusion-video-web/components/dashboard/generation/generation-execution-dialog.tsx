"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import {
  generationExecutionApi,
  type GenerationExecution,
  type GenerationExecutionItem,
  type GenerationReviewAction,
} from "@/lib/api/generation-execution";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { generationTerminalStates } from "@/lib/generation-execution-view";
import { GenerationExecutionInputSection } from "./generation-execution-input";
import { GenerationExecutionItemSection } from "./generation-execution-item";
import { executionStatus, executionTime } from "./generation-execution-presentation";

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
        if (active && !complete) {
          timer = setTimeout(() => void read(), 3000);
        }
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
    if (busy || error) {
      return false;
    }
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
          if (!open && !busy) {
            onClose();
          }
        }}
      >
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-3xl">
          <DialogHeader className="shrink-0">
            <DialogTitle>生成执行详情</DialogTitle>
            <DialogDescription>
              {data
                ? `${data.modelName} · ${executionStatus(data)}`
                : error ? "暂时无法读取任务详情" : "正在读取任务状态…"}
            </DialogDescription>
          </DialogHeader>
          <DialogScrollBody className="flex flex-col gap-4">
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
            {reviewError && (
              <p role="alert" className="text-sm text-destructive">
                {reviewError}
              </p>
            )}
            {data && (
              <>
                <p className="text-xs text-muted-foreground">提交时间：{executionTime(data.createdAt)}</p>
                {data.items.map((item) => (
                  <GenerationExecutionItemSection key={item.id} item={item} data={data}
                    busy={busy || !!error} onClose={onClose}
                    onRetry={() => void review(item, "RETRY_INGEST", "重新保存已收到的结果")} />
                ))}
                <GenerationExecutionInputSection data={data} />
              </>
            )}
          </DialogScrollBody>
        </DialogContent>
      </Dialog>
    </>
  );
}
