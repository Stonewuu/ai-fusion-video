import { AlertCircle } from "lucide-react";
import type { ExecutionRecordDetail } from "@/lib/api/execution-records";
import { GenerationExecutionInputSection } from "../generation/generation-execution-input";
import { executionTime } from "../generation/generation-execution-presentation";
import { executionRecordDuration, executionRecordStage } from "./execution-record-presentation";

export function ExecutionRecordOverview({ data }: { data: ExecutionRecordDetail }) {
  const generation = data.generation;
  return (
    <>
      {data.task.message && (
        <section aria-label="错误信息" className="flex min-w-0 items-start gap-3 rounded-lg border border-destructive/20 bg-destructive/5 p-4">
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
          <div className="flex min-w-0 flex-col gap-2">
            <h5 className="text-xs font-medium text-destructive">错误信息</h5>
            <p className="whitespace-pre-wrap break-words text-sm leading-6">{data.task.message}</p>
          </div>
        </section>
      )}
      <dl className="grid min-w-0 grid-cols-2 gap-4 text-sm">
        {generation && (
          <div className="col-span-2 flex min-w-0 flex-col gap-1">
            <dt className="text-xs text-muted-foreground">模型</dt>
            <dd className="break-words font-medium">{generation.modelName}</dd>
          </div>
        )}
        <div className="flex min-w-0 flex-col gap-1">
          <dt className="text-xs text-muted-foreground">处理阶段</dt>
          <dd>{executionRecordStage(data)}</dd>
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <dt className="text-xs text-muted-foreground">{data.task.localFinished ? "总耗时" : "已等待"}</dt>
          <dd>{executionRecordDuration(data)}</dd>
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <dt className="text-xs text-muted-foreground">创建时间</dt>
          <dd>{executionTime(data.task.createdAt)}</dd>
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <dt className="text-xs text-muted-foreground">{data.task.localFinished ? "结束时间" : "更新时间"}</dt>
          <dd>{executionTime(data.task.updatedAt)}</dd>
        </div>
        {generation && (
          <>
            <div className="flex min-w-0 flex-col gap-1">
              <dt className="text-xs text-muted-foreground">生成结果</dt>
              <dd>{generation.resultReceived ? "已收到" : "未收到"}</dd>
            </div>
            <div className="flex min-w-0 flex-col gap-1">
              <dt className="text-xs text-muted-foreground">查询与重试</dt>
              <dd>{generation.pollAttempts} 次</dd>
            </div>
          </>
        )}
      </dl>
      {data.task.remoteUnknown && (
        <p className="text-sm text-muted-foreground">供应商受理状态尚未确认，请先核对原任务再重新提交。</p>
      )}
      {generation && <GenerationExecutionInputSection data={generation} />}
      {data.remote.length > 0 && (
        <section className="flex min-w-0 flex-col gap-3 border-t border-border/20 pt-4">
          <h5 className="text-sm font-medium">调用记录</h5>
          <ol className="flex flex-col gap-3">
            {data.remote.map((remote, index) => {
              const attempt = generation?.attempts.find((item) => item.id === remote.attemptId);
              return (
                <li key={remote.attemptId} className="flex min-w-0 flex-col gap-2 text-xs">
                  <p className="font-medium">
                    {index + 1}. {remote.step} · {remote.responseRecorded ? "已收到响应" : remote.acknowledged ? "已受理" : "未取得回执"}
                  </p>
                  {attempt && (
                    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-muted-foreground">
                      <dt>提交时间</dt><dd>{executionTime(attempt.submittedAt)}</dd>
                      <dt>响应时间</dt><dd>{executionTime(attempt.acknowledgedAt)}</dd>
                    </dl>
                  )}
                  {remote.providerTaskId && (
                    <p className="break-all text-muted-foreground">供应商任务编号：{remote.providerTaskId}</p>
                  )}
                </li>
              );
            })}
          </ol>
        </section>
      )}
      <details className="min-w-0 border-t border-border/20 pt-4 text-xs">
        <summary className="cursor-pointer rounded-lg text-muted-foreground outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50">任务标识</summary>
        <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2">
          <dt className="text-muted-foreground">任务编号</dt><dd className="break-all font-mono">{data.task.id}</dd>
          {data.projectId !== null && <><dt className="text-muted-foreground">所属项目</dt><dd>{data.projectId}</dd></>}
          {generation && <><dt className="text-muted-foreground">模型代码</dt><dd className="break-all">{generation.modelCode}</dd></>}
        </dl>
      </details>
    </>
  );
}
