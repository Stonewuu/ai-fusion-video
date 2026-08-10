"use client";

import { useRef, useState } from "react";
import { toast } from "sonner";
import {
  CircleHelp,
  FileVideo2,
  ImageIcon,
  Link2,
  Loader2,
  Music2,
  Plus,
  Trash2,
} from "lucide-react";
import { uploadFile } from "@/lib/api/storage";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { resolveMediaUrl } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SafeImage } from "@/components/ui/safe-image";

type ReferenceMediaType = "image" | "video" | "audio";

interface GenerationReferenceFieldProps {
  label: string;
  hint?: string;
  values: string[];
  maxCount: number;
  mediaType: ReferenceMediaType;
  onChange: (values: string[]) => void;
  uploadDisabledReason?: string;
  manualInputAllowed?: boolean;
}

const MEDIA_CONFIG: Record<
  ReferenceMediaType,
  { accept: string; folder: string; emptyLabel: string }
> = {
  image: {
    accept: "image/*",
    folder: "generation-references",
    emptyLabel: "添加图片",
  },
  video: {
    accept: "video/*",
    folder: "generation-video-references",
    emptyLabel: "添加视频",
  },
  audio: {
    accept: "audio/*",
    folder: "generation-audio-references",
    emptyLabel: "添加音频",
  },
};

export function GenerationReferenceField({
  label,
  hint,
  values,
  maxCount,
  mediaType,
  onChange,
  uploadDisabledReason,
  manualInputAllowed = true,
}: GenerationReferenceFieldProps) {
  const config = MEDIA_CONFIG[mediaType];
  const inputRef = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [showUrl, setShowUrl] = useState(false);
  const [uploading, setUploading] = useState(false);
  const canAdd = values.length < maxCount;
  const canUpload = canAdd && !uploadDisabledReason;

  const uploadFiles = async (files: File[]) => {
    if (!canUpload) return;
    const remaining = maxCount - values.length;
    const selected = files.slice(0, remaining);
    if (!selected.length) return;

    setUploading(true);
    try {
      const urls = await Promise.all(
        selected.map((file) => uploadFile(file, config.folder)),
      );
      onChange([...values, ...urls]);
    } catch (error) {
      toast.error(getApiErrorMessage(error));
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const addUrl = () => {
    const value = url.trim();
    if (!value || !canAdd) return;
    onChange([...values, value]);
    setUrl("");
    setShowUrl(false);
  };

  return (
    <section className="rounded-xl border border-border/40 bg-background/52 p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-1.5">
          <h4 className="text-xs font-semibold">{label}</h4>
          {hint && (
            <span title={hint}>
              <CircleHelp className="h-3.5 w-3.5 text-muted-foreground" />
            </span>
          )}
        </div>
        <span className="text-[10px] tabular-nums text-muted-foreground">
          {values.length}/{maxCount}
        </span>
      </div>

      <div className="mt-2.5 flex gap-2 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {values.map((value, index) => (
          <div
            key={`${value}-${index}`}
            className="group relative grid h-16 w-16 shrink-0 place-items-center overflow-hidden rounded-xl border border-border/45 bg-muted/25"
          >
            {mediaType === "image" ? (
              <SafeImage
                src={resolveMediaUrl(value) || undefined}
                alt={`${label}${index + 1}`}
                className="h-full w-full object-cover"
                fallbackType="image"
              />
            ) : mediaType === "video" ? (
              <FileVideo2 className="h-6 w-6 text-cyan-500/70" />
            ) : (
              <Music2 className="h-6 w-6 text-violet-500/70" />
            )}
            <button
              type="button"
              onClick={() =>
                onChange(values.filter((_, itemIndex) => itemIndex !== index))
              }
              title={`移除${label}`}
              aria-label={`移除${label}`}
              className="absolute right-1 top-1 grid h-6 w-6 place-items-center rounded-lg bg-black/60 text-white opacity-0 transition-opacity group-hover:opacity-100"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        ))}

        {canAdd && (
          <button
            type="button"
            disabled={!canUpload}
            onClick={() => inputRef.current?.click()}
            title={uploadDisabledReason}
            className="flex h-16 w-16 shrink-0 flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-border/55 text-[10px] text-muted-foreground transition-colors hover:border-primary/35 hover:bg-primary/[0.035] hover:text-primary disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:border-border/55 disabled:hover:bg-transparent disabled:hover:text-muted-foreground"
          >
            {uploading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : mediaType === "image" ? (
              <ImageIcon className="h-4 w-4" />
            ) : mediaType === "video" ? (
              <FileVideo2 className="h-4 w-4" />
            ) : (
              <Music2 className="h-4 w-4" />
            )}
            {config.emptyLabel}
          </button>
        )}
      </div>

      {uploadDisabledReason && (
        <p className="mt-2 text-[10px] leading-4 text-amber-600">
          {uploadDisabledReason}
          {manualInputAllowed ? "，仍可添加模型支持的公网图片 URL。" : "。"}
        </p>
      )}

      {showUrl ? (
        <div className="mt-3 flex gap-2">
          <Input
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addUrl();
              }
            }}
            placeholder={
              mediaType === "image" ? "输入图片 URL 或 Data URI" : "输入媒体 URL"
            }
            className="h-8 rounded-lg text-xs"
            autoFocus
          />
          <Button size="xs" onClick={addUrl} disabled={!url.trim()}>
            <Plus className="h-3 w-3" />
            添加
          </Button>
        </div>
      ) : (
        canAdd && manualInputAllowed && (
          <div className="mt-2 flex justify-end">
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={() => setShowUrl(true)}
            >
              <Link2 className="h-3 w-3" />
              通过 URL 添加
            </Button>
          </div>
        )
      )}

      <input
        ref={inputRef}
        hidden
        multiple={maxCount > 1}
        type="file"
        accept={config.accept}
        onChange={(event) => void uploadFiles(Array.from(event.target.files || []))}
      />
    </section>
  );
}
