"use client";
import { Clock3, ImageIcon, Film, Music2 } from "lucide-react";
import type { CanvasRunStatistics } from "@/lib/api/canvas-runs";

function duration(ms: number) {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds} 秒`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} 分 ${seconds % 60} 秒`;
  return `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
}

export function CanvasRunStatistics({ value }: { value: CanvasRunStatistics }) {
  const { nodes, media, tokens } = value;
  const ended = nodes.total - nodes.pending;
  const successRate = ended ? Math.round(nodes.succeeded / ended * 100) : null;
  return (
    <section aria-label="本次运行统计" className="space-y-3 border-y border-border/20 py-4">
      <div className="flex items-center justify-between gap-3 text-xs">
        <h3 className="font-medium">本次运行</h3>
        <span className="flex items-center gap-1.5 text-muted-foreground" title="从受理到结束的总时长，包含排队、确认和等待；进行中为最近刷新时的耗时">
          <Clock3 className="size-3.5" />{duration(value.elapsedMs)}
        </span>
      </div>
      <div className="space-y-2">
        <div className="flex items-baseline justify-between gap-3">
          <p className="text-sm"><strong className="text-primary">{nodes.succeeded}</strong> / {nodes.total} 个节点成功</p>
          <span className="text-xs text-muted-foreground" title="成功节点 ÷ 已结束节点；含复用，不重复计入批量子项">
            节点成功率 {successRate === null ? "—" : `${successRate}%`}
          </span>
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
          {nodes.pending > 0 && <span>未结束 {nodes.pending}</span>}
          {nodes.failed > 0 && <span className="text-destructive">失败或部分失败 {nodes.failed}</span>}
          {nodes.blocked > 0 && <span>阻断 {nodes.blocked}</span>}
          {nodes.cancelled > 0 && <span>已停止 {nodes.cancelled}</span>}
          {nodes.reused > 0 && <span>其中复用 {nodes.reused}</span>}
          {nodes.batchAttempts > 0 && <span>批量子项 {nodes.batchAttempts} 次尝试 · 重试 {nodes.batchRetries} 次</span>}
        </div>
      </div>
      <div className="flex flex-wrap gap-3 text-xs" aria-label="本次新增媒体">
        <span className="flex items-center gap-1.5 text-cyan-700 dark:text-cyan-300"><ImageIcon className="size-3.5" />图片 {media.images}</span>
        <span className="flex items-center gap-1.5 text-violet-700 dark:text-violet-300"><Film className="size-3.5" />视频 {media.videos}</span>
        <span className="flex items-center gap-1.5 text-amber-700 dark:text-amber-300"><Music2 className="size-3.5" />音频 {media.audios}</span>
      </div>
      {media.unavailableNodes > 0 && (
        <p className="text-xs text-muted-foreground">
          {media.unavailableNodes} 个节点内容已不可访问，媒体数量不包含这些节点的结果。
        </p>
      )}
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer py-1">用量明细</summary>
        <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-2 pt-3">
          <dt>视频总时长</dt><dd>{(media.videoDurationMs / 1000).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 秒</dd>
          <dt>音频总时长</dt><dd>{(media.audioDurationMs / 1000).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 秒</dd>
          <dt>新增媒体大小</dt><dd>{(media.bytes / 1024 / 1024).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} MiB</dd>
          <dt>输入 Token</dt><dd>{tokens.input?.toLocaleString("zh-CN") ?? "未提供"}</dd>
          <dt>输出 Token</dt><dd>{tokens.output?.toLocaleString("zh-CN") ?? "未提供"}</dd>
          <dt>已知总 Token</dt><dd>{tokens.total?.toLocaleString("zh-CN") ?? "未提供"}</dd>
          <dt>费用</dt><dd>未配置计价</dd>
        </dl>
        <p className="pt-3 leading-relaxed">媒体仅统计本次已发布的新产物，同一文件只计一次。复用、来源素材和后续步骤转发不重复计入。</p>
        {media.unknownDurations > 0 && <p className="pt-2">另有 {media.unknownDurations} 个文件未提供时长。</p>}
        <p className="pt-2 leading-relaxed">
          {tokens.submittedCalls === 0 ? "本次没有提交文本模型请求。" : `共 ${tokens.submittedCalls} 次文本请求，${tokens.measuredCalls} 次返回用量；停止或失败请求的已知用量也计入。未返回的用量不按零计算。`}
        </p>
      </details>
    </section>
  );
}
