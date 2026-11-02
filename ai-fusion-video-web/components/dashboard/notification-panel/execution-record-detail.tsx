"use client";

import { useEffect, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { executionRecordApi, executionRecordStatus, type ExecutionRecordDetail, type ExecutionRecordRef } from "@/lib/api/execution-records";
import { useAuthStore } from "@/lib/store/auth-store";
import { formatDatetime } from "./utils";

export function ExecutionRecordDetailPanel({ target }: { target: ExecutionRecordRef }) {
  const userId = useAuthStore((state) => state.user?.id);
  const [revision, setRevision] = useState(0);
  const [loaded, setLoaded] = useState<{ key: string; value: ExecutionRecordDetail } | null>(null);
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const key = `${userId}:${target.kind}:${target.id}:${revision}`;
  const data = loaded?.key === key ? loaded.value : null;
  const error = failure?.key === key ? failure.message : null;
  useEffect(() => {
    const controller = new AbortController();
    void executionRecordApi.get(target, controller.signal).then((value) => {
      if (!controller.signal.aborted) {
        setLoaded({ key, value });
      }
    }).catch((value) => {
      if (!controller.signal.aborted) {
        setFailure({ key, message: getApiErrorMessage(value) });
      }
    });
    return () => controller.abort();
  }, [key, target]);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 items-start justify-between gap-2 border-b border-border/20 p-4">
        <div className="min-w-0">
          <h4 className="text-sm font-semibold">{data?.task.title ?? "任务详情"}</h4>
          {data && <p className="mt-1 text-xs text-muted-foreground">{executionRecordStatus(data.task)} · {formatDatetime(data.task.updatedAt)}</p>}
        </div>
        <Button variant="ghost" size="icon" aria-label="刷新任务记录" onClick={() => setRevision((value) => value + 1)}><RefreshCw /></Button>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        {!data && !error && <div role="status" aria-label="加载任务详情" className="flex justify-center p-6"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>}
        {data && <>
          <div className="rounded-xl border border-border/30 bg-card/50 p-4">
            <p className="text-sm font-medium">{data.task.localFinished ? "本地执行已结束" : "系统正在自动处理"}</p>
            <p className="mt-2 text-sm text-muted-foreground">{data.task.remoteUnknown
              ? "未取得可确认的远端结果，原请求不会自动重发。"
              : "已有的响应和处理记录会保留在此任务中。"}</p>
            <p className="mt-2 text-xs text-muted-foreground">{data.capacityHeld ? "任务仍占用并发名额，结束本地执行后会自动释放。" : "此任务未占用本地并发名额。"}</p>
          </div>
          {data.task.message && <p className="whitespace-pre-wrap break-words text-sm">{data.task.message}</p>}
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 text-xs">
            <dt className="text-muted-foreground">任务编号</dt><dd className="break-all font-mono">{data.task.id}</dd>
            <dt className="text-muted-foreground">创建时间</dt><dd>{formatDatetime(data.task.createdAt)}</dd>
            {data.projectId !== null && <><dt className="text-muted-foreground">所属项目</dt><dd>{data.projectId}</dd></>}
          </dl>
          {data.remote.length > 0 && <section className="flex flex-col gap-3">
            <h5 className="text-sm font-medium">远端调用记录</h5>
            {data.remote.map((remote) => <div key={remote.attemptId} className="flex flex-col gap-2 border-t border-border/20 pt-3 text-xs">
              <p className="font-medium">{remote.step} · {remote.responseRecorded ? "响应已记录" : remote.acknowledged ? "已取得受理回执" : "未取得受理回执"}</p>
              <p className="break-all font-mono text-muted-foreground">{remote.providerTaskId ?? remote.attemptId}</p>
            </div>)}
          </section>}
          {data.history.length > 0 && <section className="flex flex-col gap-3">
            <h5 className="text-sm font-medium">历史处理记录</h5>
            {data.history.map((item, index) => <div key={`${item.createdAt}:${index}`} className="border-t border-border/20 pt-3 text-xs">
              <p className="text-muted-foreground">{formatDatetime(item.createdAt)} · 管理员 {item.adminUserId}</p>
              <p className="mt-2 whitespace-pre-wrap break-words">{item.note}</p>
            </div>)}
          </section>}
        </>}
      </div>
    </div>
  );
}
