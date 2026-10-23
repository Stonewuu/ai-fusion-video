"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { ExecutionRecoveryDialog } from "@/components/dashboard/execution-recovery/recovery-dialog";
import {
  executionRecoveryApi,
  type RecoveryExecution,
} from "@/lib/api/execution-recovery";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";

export function CanvasAgentModelRecovery({
  requestIds,
  onChanged,
}: {
  requestIds: string[];
  onChanged: () => void;
}) {
  const user = useAuthStore((state) => state.user);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [target, setTarget] = useState<RecoveryExecution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isAdmin = !!user?.roles.includes("admin");
  useEffect(() => {
    if (!requestId || !isAdmin) return;
    const controller = new AbortController();
    void executionRecoveryApi
      .get({ kind: "AGENT_MODEL", id: requestId }, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setTarget(result);
      })
      .catch((failure) => {
        if (!controller.signal.aborted) {
          setError(getApiErrorMessage(failure));
          setRequestId(null);
        }
      });
    return () => controller.abort();
  }, [requestId, isAdmin, user?.id]);

  const close = () => {
    setTarget(null);
    setRequestId(null);
    onChanged();
  };
  return (
    <div
      role="status"
      className="flex flex-col gap-3 rounded-lg border border-border/30 bg-muted/40 p-3 text-sm"
    >
      <p>
        有 {requestIds.length}{" "}
        次模型调用在连接中断前没有返回结束状态，暂时保留并发名额，避免重复执行。
      </p>
      {isAdmin ? (
        <div className="flex flex-wrap gap-2">
          {requestIds.map((id, index) => (
            <Button
              key={id}
              variant="outline"
              size="sm"
              disabled={requestId !== null}
              onClick={() => {
                setError(null);
                setRequestId(id);
              }}
            >
              {requestId === id && !target
                ? "读取中…"
                : requestIds.length === 1
                  ? "处理未完成调用"
                  : `处理第 ${index + 1} 次调用`}
            </Button>
          ))}
        </div>
      ) : (
        <p className="text-muted-foreground">
          请管理员打开此画布的任务详情，在这里确认原调用已结束。
        </p>
      )}
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
      {isAdmin && user && target && (
        <ExecutionRecoveryDialog
          key={`${user.id}:${target.id}`}
          target={target}
          adminUserId={user.id}
          onClose={close}
          onCompleted={close}
        />
      )}
    </div>
  );
}
