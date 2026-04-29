"use client";

import { Clapperboard, Image as ImageIcon, Info, RefreshCcw } from "lucide-react";

type CapabilityResultRecord = Record<string, unknown>;

function toText(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function toNumber(value: unknown): number | undefined {
  return typeof value === "number" && !Number.isNaN(value) ? value : undefined;
}

function toBoolean(value: unknown): boolean {
  return value === true;
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    : [];
}

function formatList(values: string[]): string {
  return values.length > 0 ? values.join(" / ") : "未声明";
}

function formatLimit(prefix: string, value: number | undefined, unit: string): string {
  return value !== undefined ? `${prefix}${value}${unit}` : `${prefix}未限制`;
}

function CapabilityBadge({
  label,
  tone = "muted",
}: {
  label: string;
  tone?: "positive" | "muted" | "info";
}) {
  const className = tone === "positive"
    ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-600"
    : tone === "info"
      ? "border-sky-500/20 bg-sky-500/10 text-sky-600"
      : "border-border/40 bg-muted/50 text-muted-foreground";

  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] leading-none ${className}`}>
      {label}
    </span>
  );
}

function SectionBlock({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/70 p-3 space-y-2">
      <div className="flex items-center gap-2 text-xs font-medium text-foreground">
        {icon}
        <span>{title}</span>
      </div>
      {children}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2 text-[11px] leading-5">
      <span className="shrink-0 text-muted-foreground/75">{label}:</span>
      <span className="text-foreground/85">{value}</span>
    </div>
  );
}

function ImageCapabilitySection({ data }: { data: CapabilityResultRecord }) {
  const configured = toBoolean(data.configured);
  if (!configured) {
    return (
      <SectionBlock title="图片模型能力" icon={<ImageIcon className="h-3.5 w-3.5 text-sky-500" />}>
        <p className="text-xs text-muted-foreground">{toText(data.summary) || "当前未配置默认图片模型。"}</p>
      </SectionBlock>
    );
  }

  const supportsReferenceImages = toBoolean(data.supportsReferenceImages);
  const asyncMode = toBoolean(data.asyncMode);
  const minReferenceImages = toNumber(data.minReferenceImages);
  const maxReferenceImages = toNumber(data.maxReferenceImages);
  const supportedAspectRatios = toStringArray(data.supportedAspectRatios);
  const supportedSizes = data.supportedSizes && typeof data.supportedSizes === "object"
    ? Object.keys(data.supportedSizes as Record<string, unknown>)
    : [];

  return (
    <SectionBlock title="图片模型能力" icon={<ImageIcon className="h-3.5 w-3.5 text-sky-500" />}>
      <div className="space-y-1">
        <p className="text-xs font-medium text-foreground">{toText(data.modelName) || "默认图片模型"}</p>
        <p className="text-[11px] text-muted-foreground/80 leading-5">{toText(data.summary) || ""}</p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <CapabilityBadge
          label={supportsReferenceImages ? (maxReferenceImages !== undefined ? `参考图 ≤${maxReferenceImages}` : "支持参考图") : "仅文生图"}
          tone={supportsReferenceImages ? "positive" : "muted"}
        />
        <CapabilityBadge label={`最少参考图 ${minReferenceImages ?? 0} 张`} tone="info" />
        {asyncMode && <CapabilityBadge label="异步任务" tone="positive" />}
        {supportedAspectRatios.length > 0 && <CapabilityBadge label={`${supportedAspectRatios.length} 种比例`} tone="info" />}
        {supportedSizes.length > 0 && <CapabilityBadge label={`${supportedSizes.length} 个尺寸档`} tone="info" />}
      </div>

      <div className="space-y-1">
        <DetailRow label="默认尺寸" value={`${toNumber(data.defaultWidth) ?? "?"} × ${toNumber(data.defaultHeight) ?? "?"}`} />
        <DetailRow label="异步模式" value={asyncMode ? "已启用" : "未启用"} />
        <DetailRow label="支持比例" value={formatList(supportedAspectRatios)} />
        <DetailRow label="尺寸档位" value={supportedSizes.length > 0 ? supportedSizes.join(" / ") : "未声明"} />
        <DetailRow label="调用建议" value={toText(data.toolGuidance) || "按模型支持情况组织 imageUrls。"} />
      </div>
    </SectionBlock>
  );
}

function VideoCapabilitySection({ data }: { data: CapabilityResultRecord }) {
  const configured = toBoolean(data.configured);
  if (!configured) {
    return (
      <SectionBlock title="视频模型能力" icon={<Clapperboard className="h-3.5 w-3.5 text-emerald-500" />}>
        <p className="text-xs text-muted-foreground">{toText(data.summary) || "当前未配置默认视频模型。"}</p>
      </SectionBlock>
    );
  }

  const supportsFirstFrame = toBoolean(data.supportsFirstFrame);
  const supportsLastFrame = toBoolean(data.supportsLastFrame);
  const supportsReferenceImages = toBoolean(data.supportsReferenceImages);
  const supportsReferenceVideos = toBoolean(data.supportsReferenceVideos);
  const supportsReferenceAudios = toBoolean(data.supportsReferenceAudios);
  const supportedAspectRatios = toStringArray(data.supportedAspectRatios);
  const supportedResolutions = toStringArray(data.supportedResolutions);

  return (
    <SectionBlock title="视频模型能力" icon={<Clapperboard className="h-3.5 w-3.5 text-emerald-500" />}>
      <div className="space-y-1">
        <p className="text-xs font-medium text-foreground">{toText(data.modelName) || "默认视频模型"}</p>
        <p className="text-[11px] text-muted-foreground/80 leading-5">{toText(data.summary) || ""}</p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <CapabilityBadge label={supportsFirstFrame ? "支持首帧" : "无首帧"} tone={supportsFirstFrame ? "positive" : "muted"} />
        <CapabilityBadge label={supportsLastFrame ? "支持尾帧" : "无尾帧"} tone={supportsLastFrame ? "positive" : "muted"} />
        <CapabilityBadge label={supportsReferenceImages ? formatLimit("参考图 ≤", toNumber(data.maxReferenceImages), " 张") : "无参考图"} tone={supportsReferenceImages ? "positive" : "muted"} />
        <CapabilityBadge label={supportsReferenceVideos ? formatLimit("参考视频 ≤", toNumber(data.maxReferenceVideos), " 个") : "无参考视频"} tone={supportsReferenceVideos ? "positive" : "muted"} />
        <CapabilityBadge label={supportsReferenceAudios ? formatLimit("参考音频 ≤", toNumber(data.maxReferenceAudios), " 个") : "无参考音频"} tone={supportsReferenceAudios ? "positive" : "muted"} />
        <CapabilityBadge label={`${toNumber(data.minImageInputs) ?? 0} - ${toNumber(data.maxImageInputs) ?? "∞"} 张图片输入`} tone="info" />
      </div>

      <div className="space-y-1">
        <DetailRow label="支持比例" value={formatList(supportedAspectRatios)} />
        <DetailRow label="支持分辨率" value={formatList(supportedResolutions)} />
        <DetailRow label="时长范围" value={`${toNumber(data.minDuration) ?? "?"} - ${toNumber(data.maxDuration) ?? "?"} 秒，默认 ${toNumber(data.defaultDuration) ?? "?"} 秒`} />
        <DetailRow label="调用建议" value={toText(data.toolGuidance) || "按模型支持情况组织多模态输入。"} />
      </div>
    </SectionBlock>
  );
}

export function GenerationModelCapabilitiesResult({ data }: { data: unknown }) {
  const obj = (typeof data === "object" && data !== null ? data : {}) as CapabilityResultRecord;
  const requestedModelType = toText(obj.requestedModelType) || "all";
  const imageData = typeof obj.image === "object" && obj.image !== null ? (obj.image as CapabilityResultRecord) : null;
  const videoData = typeof obj.video === "object" && obj.video !== null ? (obj.video as CapabilityResultRecord) : null;

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-primary/15 bg-primary/5 p-3 space-y-2">
        <div className="flex items-center gap-2 text-xs font-medium text-foreground">
          <Info className="h-3.5 w-3.5 text-primary" />
          <span>生成模型能力查询</span>
          <span className="rounded-full bg-background px-2 py-0.5 text-[10px] text-muted-foreground">
            {requestedModelType === "all" ? "图片 + 视频" : requestedModelType === "image" ? "图片" : "视频"}
          </span>
        </div>
        {toText(obj.usageHint) && <p className="text-[11px] leading-5 text-muted-foreground/85">{toText(obj.usageHint)}</p>}
        {toText(obj.retryPolicy) && (
          <div className="flex items-start gap-2 rounded-md bg-background/70 px-2.5 py-2 text-[11px] leading-5 text-muted-foreground/85">
            <RefreshCcw className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
            <span>{toText(obj.retryPolicy)}</span>
          </div>
        )}
      </div>

      <div className="grid gap-3">
        {imageData && <ImageCapabilitySection data={imageData} />}
        {videoData && <VideoCapabilitySection data={videoData} />}
      </div>
    </div>
  );
}