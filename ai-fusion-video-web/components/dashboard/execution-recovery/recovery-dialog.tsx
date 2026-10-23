"use client";

import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  executionRecoveryApi,
  recoveryKindNames,
  type RecoveryDecision,
  type RecoveryExecution,
} from "@/lib/api/execution-recovery";
import { getApiErrorMessage } from "@/lib/api/api-error";

export function ExecutionRecoveryDialog({
  target,
  adminUserId,
  onClose,
  onCompleted,
}: {
  target: RecoveryExecution;
  adminUserId: number;
  onClose: () => void;
  onCompleted: () => void;
}) {
  const [note, setNote] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [pending, setPending] = useState<RecoveryDecision | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();
  const submit = async () => {
    if (busy) return;
    const decision = pending ?? {
      adminUserId,
      expectedRevision: target.revision,
      expectedFactsHash: target.factsHash,
      requestId: crypto.randomUUID(),
      note: note.trim(),
      providerFinished: confirmed,
    };
    setPending(decision);
    setBusy(true);
    setError(null);
    try {
      await executionRecoveryApi.confirm(target, decision);
      onCompleted();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>处理未完成调用</DialogTitle>
          <DialogDescription>
            {target.kind === "AGENT_MODEL"
              ? "连接中断后，系统未收到模型的结束响应。请先确认模型服务中的原请求已结束。"
              : "原发起人的权限已失效。"}
            核实后结束本地等待并释放额度，已有内容和结果保留，后续执行停止。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="flex flex-col gap-4">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">任务类型</dt>
            <dd>{recoveryKindNames[target.kind]}</dd>
            <dt className="text-muted-foreground">任务编号</dt>
            <dd className="break-all font-mono text-xs">{target.id}</dd>
            <dt className="text-muted-foreground">原发起人</dt>
            <dd>
              用户 {target.actorUserId}
              {target.actorTeamId
                ? ` · 团队 ${target.actorTeamId}`
                : " · 个人范围"}
            </dd>
            <dt className="text-muted-foreground">当前占用</dt>
            <dd>
              {target.capacityHeld ? "远端并发名额仍保留" : "已释放并发名额"}
            </dd>
          </dl>
          <div className="flex flex-col gap-2">
            <h3 className="text-sm font-medium">原提交记录</h3>
            {target.remote.length ? (
              <ul className="divide-y divide-border/20 rounded-lg border border-border/20 bg-background/70">
                {target.remote.map((remote) => (
                  <li
                    key={remote.attemptId}
                    className="flex flex-col gap-1 p-3 text-xs"
                  >
                    <span className="font-medium">
                      {remote.step} ·{" "}
                      {remote.responseRecorded
                        ? "终态响应已记录"
                        : "终态尚未记录"}
                    </span>
                    <span className="break-all text-muted-foreground">
                      {remote.providerTaskId ??
                        "供应商未返回任务编号，请在供应商记录中核实原提交"}
                    </span>
                    <span className="break-all font-mono text-muted-foreground">
                      尝试编号：{remote.attemptId}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-muted-foreground">
                未记录远端受理标识，请依据供应商记录核实。
              </p>
            )}
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-note`}>核实依据</Label>
            <Textarea
              id={`${id}-note`}
              value={note}
              onChange={(event) => setNote(event.target.value)}
              maxLength={2000}
              disabled={busy || !!pending}
              placeholder="记录核实时间、供应商原任务状态及确认依据"
            />
          </div>
          <label className="flex items-start gap-2 text-sm">
            <Checkbox
              checked={confirmed}
              onCheckedChange={(value) => setConfirmed(value === true)}
              disabled={busy || !!pending}
            />
            我已核实上方所有原远端请求均已结束，可以释放并发名额
          </label>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          {pending && error && (
            <p className="text-xs text-muted-foreground">
              重试会提交同一份决定。需要查看最新状态时，可返回任务列表刷新。
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={busy} onClick={onClose}>
            返回
          </Button>
          <Button
            disabled={busy || !target.canResolve || !confirmed || !note.trim()}
            onClick={() => void submit()}
          >
            {busy ? "提交中…" : pending ? "重试提交" : "确认结束"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
