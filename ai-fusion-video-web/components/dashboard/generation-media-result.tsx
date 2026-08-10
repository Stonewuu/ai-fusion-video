"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { CheckCircle2, Expand, Play, X } from "lucide-react";

import { resolveMediaUrl } from "@/lib/api/client";

type Obj = Record<string, unknown>;

function ImageLightbox({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return createPortal(
    <div
      className="modal-overlay fixed inset-0 z-200 flex items-center justify-center px-4 py-6"
      onClick={onClose}
    >
      <button
        type="button"
        onClick={onClose}
        className="absolute right-4 top-4 rounded-full border border-border/40 bg-background/80 p-2 text-foreground shadow-xl backdrop-blur-xl transition-colors hover:bg-background"
        aria-label="关闭预览"
      >
        <X className="h-5 w-5" />
      </button>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        className="max-h-[88vh] max-w-[92vw] rounded-xl object-contain shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      />
    </div>,
    document.body
  );
}

function PromptBlock({ prompt }: { prompt: string }) {
  return (
    <div className="space-y-1 rounded-lg border border-border/40 bg-muted/25 p-2.5">
      <p className="text-[10px] font-medium tracking-[0.08em] text-muted-foreground/70">提示词</p>
      <p className="line-clamp-3 text-[11px] leading-5 text-muted-foreground/90">{prompt}</p>
    </div>
  );
}

export function ThumbImage({
  src,
  label,
  previewClassName = "h-[168px] w-[168px] sm:h-[184px] sm:w-[184px]",
}: {
  src: string;
  label: string;
  previewClassName?: string;
}) {
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  return (
    <>
      <div className="space-y-1.5">
        <p className="text-[10px] font-medium text-muted-foreground/70">{label}</p>
        <button
          type="button"
          onClick={() => setIsPreviewOpen(true)}
          className="group relative block rounded-2xl border border-border/50 bg-muted/15 p-1 text-left transition-all hover:border-border/80 hover:bg-muted/25"
          aria-label={`查看${label}`}
        >
          <div className={`flex items-center justify-center overflow-hidden rounded-lg bg-background/80 ${previewClassName}`}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={src}
              alt={label}
              className="h-full w-full object-contain"
              loading="lazy"
            />
          </div>
          <div className="pointer-events-none absolute inset-x-1 inset-y-1 rounded-[14px] bg-gradient-to-t from-background/16 via-transparent to-background/4 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100" />
          <div className="pointer-events-none absolute right-3 top-3 inline-flex items-center gap-1 rounded-full border border-white/15 bg-black/45 px-2.5 py-1 text-[10px] font-medium text-white opacity-0 shadow-lg backdrop-blur-md transition-all duration-200 group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100 translate-y-1">
            <Expand className="h-3 w-3" />
            <span>放大查看</span>
          </div>
        </button>
      </div>
      {isPreviewOpen ? <ImageLightbox src={src} alt={label} onClose={() => setIsPreviewOpen(false)} /> : null}
    </>
  );
}

export function ImageGenerateResult({ data }: { data: unknown }) {
  const obj = data as Obj;
  const resolved = resolveMediaUrl(obj.imageUrl as string);
  const prompt = obj.prompt as string | undefined;
  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-1.5 text-xs">
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
        <span className="font-medium text-foreground">生成成功</span>
      </div>
      {resolved && <ThumbImage src={resolved} label="生成图片" />}
      {prompt && <PromptBlock prompt={prompt} />}
    </div>
  );
}

export function VideoGenerateResult({ data }: { data: unknown }) {
  const obj = data as Obj;
  const videoUrl = resolveMediaUrl(obj.videoUrl as string);
  const coverUrl = resolveMediaUrl(obj.coverUrl as string);
  const prompt = obj.prompt as string | undefined;
  const duration = obj.duration as number | undefined;
  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-1.5 text-xs">
        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
        <span className="font-medium text-foreground">生成成功</span>
        {duration != null && (
          <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
            {duration}s
          </span>
        )}
      </div>
      {coverUrl && (
        <ThumbImage
          src={coverUrl}
          label="封面图"
          previewClassName="h-[104px] w-[184px] sm:h-[116px] sm:w-[206px]"
        />
      )}
      {videoUrl && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground/70">
            <Play className="h-3 w-3" />
            <span>视频预览</span>
          </div>
          <video
            src={videoUrl}
            controls
            className="w-[206px] max-w-full rounded-xl border border-border/50 bg-muted/20 sm:w-[232px]"
          />
        </div>
      )}
      {prompt && <PromptBlock prompt={prompt} />}
    </div>
  );
}
