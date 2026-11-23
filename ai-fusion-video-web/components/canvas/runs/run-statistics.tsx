"use client";

import { Clock3, Film, ImageIcon, Music2 } from "lucide-react";
import type { CanvasRunStatistics } from "@/lib/api/canvas-runs";

function duration(ms: number) {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return `${seconds} 秒`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes} 分 ${seconds % 60} 秒`;
  }
  return `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
}

export function CanvasRunStatistics({ value }: { value: CanvasRunStatistics }) {
  const { nodes, media, tokens } = value;
  const ended = nodes.total - nodes.pending;
  const successRate = ended ? Math.round((nodes.succeeded / ended) * 100) : null;
  return (
    <section
      aria-label="本次运行统计"
      className="flex flex-col gap-2.5 rounded-[12px] border border-foreground/[0.07] bg-foreground/[0.02] p-3 dark:border-white/[0.08] dark:bg-white/[0.025]"
    >
      <div className="flex items-center justify-between gap-2 text-xs">
        <h3 className="font-medium text-foreground/85">运行统计</h3>
        <span
          className="flex items-center gap-1 tabular-nums text-muted-foreground"
          title="从受理到结束的总时长，包含排队、确认和等待；进行中为最近刷新时的耗时"
        >
          <Clock3 className="size-3.5" aria-hidden />
          {duration(value.elapsedMs)}
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between gap-2">
          <p className="text-[13px] tabular-nums">
            <strong className="font-semibold text-foreground">
              {nodes.succeeded}
            </strong>{" "}
            / {nodes.total} 个节点成功
          </p>
          <span
            className="text-[11px] tabular-nums text-muted-foreground"
            title="成功节点 ÷ 已结束节点；含复用，不重复计入批量子项"
          >
            成功率 {successRate === null ? "—" : `${successRate}%`}
          </span>
        </div>
        <div className="flex flex-wrap gap-x-2.5 gap-y-1 text-[11px] text-muted-foreground">
          {nodes.pending > 0 && <span>未结束 {nodes.pending}</span>}
          {nodes.failed > 0 && (
            <span className="text-destructive">
              失败或部分失败 {nodes.failed}
            </span>
          )}
          {nodes.blocked > 0 && <span>阻断 {nodes.blocked}</span>}
          {nodes.cancelled > 0 && <span>已停止 {nodes.cancelled}</span>}
          {nodes.reused > 0 && <span>复用 {nodes.reused}</span>}
          {nodes.batchAttempts > 0 && (
            <span>
              批量子项 {nodes.batchAttempts} 次 · 重试 {nodes.batchRetries} 次
            </span>
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 text-[11px]" aria-label="本次新增媒体">
        <span className="inline-flex items-center gap-1 rounded-[6px] bg-sky-500/10 px-2 py-0.5 tabular-nums text-sky-700 dark:text-sky-300">
          <ImageIcon className="size-3" aria-hidden />
          图片 {media.images}
        </span>
        <span className="inline-flex items-center gap-1 rounded-[6px] bg-purple-500/10 px-2 py-0.5 tabular-nums text-purple-700 dark:text-purple-300">
          <Film className="size-3" aria-hidden />
          视频 {media.videos}
        </span>
        <span className="inline-flex items-center gap-1 rounded-[6px] bg-amber-500/10 px-2 py-0.5 tabular-nums text-amber-700 dark:text-amber-300">
          <Music2 className="size-3" aria-hidden />
          音频 {media.audios}
        </span>
      </div>
      {media.unavailableNodes > 0 && (
        <p className="text-[11px] text-muted-foreground">
          {media.unavailableNodes} 个节点内容已不可访问，未计入媒体数量。
        </p>
      )}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer rounded-[6px] py-0.5 text-[11px] font-medium outline-none focus-visible:ring-2 focus-visible:ring-primary/30">
          用量明细
        </summary>
        <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1.5 pt-2 text-[11px] tabular-nums">
          <dt>视频总时长</dt>
          <dd>
            {(media.videoDurationMs / 1000).toLocaleString("zh-CN", {
              maximumFractionDigits: 2,
            })}{" "}
            秒
          </dd>
          <dt>音频总时长</dt>
          <dd>
            {(media.audioDurationMs / 1000).toLocaleString("zh-CN", {
              maximumFractionDigits: 2,
            })}{" "}
            秒
          </dd>
          <dt>新增媒体大小</dt>
          <dd>
            {(media.bytes / 1024 / 1024).toLocaleString("zh-CN", {
              maximumFractionDigits: 2,
            })}{" "}
            MiB
          </dd>
          <dt>输入 Token</dt>
          <dd>{tokens.input?.toLocaleString("zh-CN") ?? "未提供"}</dd>
          <dt>输出 Token</dt>
          <dd>{tokens.output?.toLocaleString("zh-CN") ?? "未提供"}</dd>
          <dt>已知总 Token</dt>
          <dd>{tokens.total?.toLocaleString("zh-CN") ?? "未提供"}</dd>
          <dt>费用</dt>
          <dd>未配置计价</dd>
        </dl>
        <p className="pt-2 text-[11px] leading-relaxed">
          媒体仅统计本次已发布的新产物，同一文件只计一次。
        </p>
        {media.unknownDurations > 0 && (
          <p className="pt-1 text-[11px]">
            另有 {media.unknownDurations} 个文件未提供时长。
          </p>
        )}
        <p className="pt-1 text-[11px] leading-relaxed">
          {tokens.submittedCalls === 0
            ? "本次没有提交文本模型请求。"
            : `共 ${tokens.submittedCalls} 次文本请求，${tokens.measuredCalls} 次返回用量。`}
        </p>
      </details>
    </section>
  );
}
