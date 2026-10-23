"use client";

import { Clock3, Film, ImageIcon, Music2, PackageCheck } from "lucide-react";
import type { CanvasActivity } from "@/lib/api/canvas-activity";

const count = (value: number | null) => value?.toLocaleString("zh-CN") ?? "未提供";
const seconds = (value: number) => (value / 1000).toLocaleString("zh-CN", { maximumFractionDigits: 1 });
const bytes = (value: number) => value < 1024 ? `${value} B` : value < 1024 * 1024
  ? `${(value / 1024).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} KiB`
  : `${(value / 1024 / 1024).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} MiB`;
export function CanvasActivitySummary({ value }: { value: CanvasActivity }) {
  const success = value.states.SUCCEEDED ?? 0;
  const failed = (value.states.FAILED ?? 0) + (value.states.PARTIAL_FAILED ?? 0) + (value.states.BLOCKED ?? 0);
  const stopped = value.states.CANCELLED ?? 0;
  const finished = success + failed + stopped;
  const pending = value.runs - finished;
  return (
    <div className="space-y-5">
      <section className="space-y-3" aria-label="运行汇总">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <p className="text-sm text-muted-foreground">共 <strong className="text-3xl font-semibold text-foreground">{count(value.runs)}</strong> 次运行</p>
          <p className="text-sm">成功率 <strong className="text-lg text-emerald-700 dark:text-emerald-300">{finished ? `${Math.round(success / finished * 100)}%` : "—"}</strong></p>
        </div>
        {value.runs > 0 && <div className="flex h-2 overflow-hidden rounded-full bg-muted" aria-hidden>
          <span className="bg-emerald-500" style={{ width: `${success / value.runs * 100}%` }} />
          <span className="bg-destructive" style={{ width: `${failed / value.runs * 100}%` }} />
          <span className="bg-amber-500" style={{ width: `${stopped / value.runs * 100}%` }} />
          <span className="bg-primary" style={{ width: `${pending / value.runs * 100}%` }} />
        </div>}
        <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs">
          <span className="text-emerald-700 dark:text-emerald-300">成功 {count(success)}</span>
          <span className="text-destructive">失败或部分失败 {count(failed)}</span>
          <span className="text-amber-700 dark:text-amber-300">停止 {count(stopped)}</span>
          <span className="text-sky-700 dark:text-sky-300">未结束 {count(pending)}</span>
        </div>
        <p className="text-xs text-muted-foreground">成功率为成功运行数 ÷ 已结束运行数，停止计入分母。</p>
      </section>
      <section className="space-y-3 border-t border-border/20 pt-4" aria-label="媒体与执行用量">
        <div className="flex flex-wrap gap-4 text-sm">
          <span className="flex items-center gap-2 text-cyan-700 dark:text-cyan-300"><ImageIcon className="size-4" />图片 {count(value.media.images)}</span>
          <span className="flex items-center gap-2 text-violet-700 dark:text-violet-300"><Film className="size-4" />视频 {count(value.media.videos)}</span>
          <span className="flex items-center gap-2 text-amber-700 dark:text-amber-300"><Music2 className="size-4" />音频 {count(value.media.audios)}</span>
        </div>
        <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-2 text-sm [&>dd]:text-right [&>dd]:tabular-nums">
          <dt className="flex items-center gap-2 text-muted-foreground"><Clock3 className="size-4" />累计运行时长</dt><dd>{seconds(value.elapsedMs)} 秒</dd>
          <dt className="text-muted-foreground">成功 / 全部节点</dt><dd>{count(value.nodes.succeeded)} / {count(value.nodes.total)}</dd>
          <dt className="text-muted-foreground">复用节点</dt><dd>{count(value.nodes.reused)}</dd>
          <dt className="text-muted-foreground">批量尝试 / 重试</dt><dd>{count(value.nodes.batchAttempts)} / {count(value.nodes.batchRetries)}</dd>
          <dt className="text-muted-foreground">视频 / 音频总时长</dt><dd>{seconds(value.media.videoDurationMs)} / {seconds(value.media.audioDurationMs)} 秒</dd>
          <dt className="text-muted-foreground">新增媒体大小</dt><dd>{bytes(value.media.bytes)}</dd>
          <dt className="text-muted-foreground">输入 / 输出 Token</dt><dd>{count(value.tokens.input)} / {count(value.tokens.output)}</dd>
          <dt className="text-muted-foreground">已知总 Token</dt><dd>{count(value.tokens.total)}</dd>
        </dl>
        <p className="text-xs leading-relaxed text-muted-foreground">累计时长包含排队与确认时间；同时运行的任务分别计时。同一份图片、视频或音频只计一次。</p>
        <p className="text-xs text-muted-foreground">{value.tokens.submittedCalls} 次文本模型请求中，{value.tokens.measuredCalls} 次返回完整用量。未返回的用量不按零计算，费用未配置计价。</p>
        {value.media.unavailableNodes > 0 && <p className="text-xs text-muted-foreground">{value.media.unavailableNodes} 个节点的媒体已删除或无权访问，未计入媒体汇总。</p>}
        {value.media.unknownDurations > 0 && <p className="text-xs text-muted-foreground">{value.media.unknownDurations} 个文件未提供时长。</p>}
      </section>
      <section className="space-y-2 border-t border-border/20 pt-4" aria-label="交付汇总">
        <h3 className="flex items-center gap-2 text-sm font-medium"><PackageCheck className="size-4 text-sky-700 dark:text-sky-300" />导出记录</h3>
        <p className="text-sm">{value.deliveries.states.SUCCEEDED ?? 0} / {value.deliveries.total} 次完成 · {bytes(value.deliveries.bytes)}</p>
        <p className="text-xs text-muted-foreground">失败 {value.deliveries.states.FAILED ?? 0} 次 · 停止 {value.deliveries.states.CANCELLED ?? 0} 次。大小仅包含有权访问的成功交付。</p>
        {value.deliveries.unavailable > 0 && <p className="text-xs text-muted-foreground">{value.deliveries.unavailable} 条交付内容受限，未计入大小。</p>}
      </section>
    </div>
  );
}
