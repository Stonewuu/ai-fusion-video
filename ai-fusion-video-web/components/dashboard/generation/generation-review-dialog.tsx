"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { generationReviewLabels } from "@/lib/generation-execution-view";
import type {
  GenerationExecutionItem,
  GenerationReviewAction,
} from "@/lib/api/generation-execution";

export function GenerationReviewDialog({
  item,
  action,
  busy,
  error,
  onClose,
  onConfirm,
}: {
  item: GenerationExecutionItem;
  action: GenerationReviewAction;
  busy: boolean;
  error: string | null;
  onClose: () => void;
  onConfirm: (note: string) => Promise<boolean>;
}) {
  const [note, setNote] = useState("");
  const retry = action === "CONFIRM_NOT_ACCEPTED_AND_RETRY";
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{generationReviewLabels[action]}</DialogTitle>
          <DialogDescription>
            {retry
              ? "请先在供应商侧确认原请求没有被受理。确认后会重新提交同一生成内容，可能产生新的费用。"
              : "请先在供应商侧确认任务已经结束。此操作会结束本地等待并释放并发名额，不会生成结果或重新发送请求。"}
          </DialogDescription>
        </DialogHeader>
        {item.providerTaskId && (
          <p className="break-all text-xs text-muted-foreground">
            供应商任务编号：{item.providerTaskId}
          </p>
        )}
        <label className="space-y-2 text-sm">
          <span>核实依据</span>
          <Textarea
            disabled={busy}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            maxLength={2000}
            placeholder="例如：供应商记录中的状态、核实时间或客服确认信息"
          />
        </label>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            返回
          </Button>
          <Button
            disabled={busy || !note.trim()}
            onClick={() =>
              void onConfirm(note.trim()).then((success) => {
                if (success) onClose();
              })
            }
          >
            {busy
              ? "正在提交…"
              : retry
                ? "已核实，重新提交"
                : "已核实，结束等待"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
