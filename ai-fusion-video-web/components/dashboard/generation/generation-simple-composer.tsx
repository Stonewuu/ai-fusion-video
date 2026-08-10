"use client";

import { useRef, useState, type ClipboardEvent, type DragEvent } from "react";
import { toast } from "sonner";
import {
  CircleAlert,
  FileVideo2,
  ImagePlus,
  Loader2,
  Music2,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import type { AiModel } from "@/lib/api/ai-model";
import type { GenerationCapabilities } from "@/lib/generation-capabilities";
import { uploadFile } from "@/lib/api/storage";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { resolveMediaUrl } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/ui/safe-image";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type {
  GenerationFormState,
  SimpleAttachment,
  SimpleAttachmentKind,
  SimpleAttachmentMediaType,
  SimpleAttachmentUploadOption,
  WorkbenchMode,
} from "./generation-types";
import { GenerationAttachmentUploadMenu } from "./generation-attachment-upload-menu";
import { GenerationQuickSettings } from "./generation-quick-settings";

const MEDIA_UPLOAD_CONFIG: Record<
  SimpleAttachmentMediaType,
  { accept: string; folder: string }
> = {
  image: { accept: "image/*", folder: "generation-references" },
  video: { accept: "video/*", folder: "generation-video-references" },
  audio: { accept: "audio/*", folder: "generation-audio-references" },
};

interface GenerationSimpleComposerProps {
  mode: WorkbenchMode;
  models: AiModel[];
  modelId?: number;
  loadingModels: boolean;
  capabilities: GenerationCapabilities | null;
  form: GenerationFormState;
  attachments: SimpleAttachment[];
  attachmentOptions: SimpleAttachmentUploadOption[];
  maxImageAttachments: number;
  attachmentCapacity: number;
  minAttachments: number;
  submitting: boolean;
  onModelChange: (modelId: number) => void;
  onFormChange: (patch: Partial<GenerationFormState>) => void;
  onAddAttachments: (urls: string[], kind?: SimpleAttachmentKind) => void;
  onRemoveAttachment: (url: string) => void;
  onSubmit: () => void;
  onAdvanced: () => void;
  attachmentDisabledReason?: string;
}

export function GenerationSimpleComposer({
  mode,
  models,
  modelId,
  loadingModels,
  capabilities,
  form,
  attachments,
  attachmentOptions,
  maxImageAttachments,
  attachmentCapacity,
  minAttachments,
  submitting,
  onModelChange,
  onFormChange,
  onAddAttachments,
  onRemoveAttachment,
  onSubmit,
  onAdvanced,
  attachmentDisabledReason = "",
}: GenerationSimpleComposerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const selectedUploadOptionRef = useRef<SimpleAttachmentUploadOption | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const prompt = form.prompt;
  const imageAttachmentCount = attachments.filter(
    (attachment) => attachment.mediaType === "image",
  ).length;
  const canPasteImages = attachmentOptions.some(
    (option) =>
      option.mediaType === "image" &&
      option.remaining > 0 &&
      !option.disabledReason,
  );
  const missingRequiredAttachments = imageAttachmentCount < minAttachments;
  const canSubmit = Boolean(
    modelId &&
    prompt.trim() &&
    !missingRequiredAttachments &&
    !submitting &&
    !uploading,
  );

  const uploadFiles = async (
    files: File[],
    option?: SimpleAttachmentUploadOption,
  ) => {
    const mediaType = option?.mediaType || "image";
    const config = MEDIA_UPLOAD_CONFIG[mediaType];
    const remaining = option?.remaining ?? Math.max(
      0,
      maxImageAttachments - imageAttachmentCount,
    );
    const selected = files
      .filter((file) => file.type.startsWith(`${mediaType}/`))
      .slice(0, remaining);
    if (!selected.length) return;

    setUploading(true);
    setError("");
    try {
      const urls = await Promise.all(
        selected.map((file) => uploadFile(file, config.folder)),
      );
      onAddAttachments(urls, option?.kind);
    } catch (error) {
      const message = getApiErrorMessage(error);
      setError(message);
      toast.error(message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
      selectedUploadOptionRef.current = null;
    }
  };

  const handleSelectUpload = (option: SimpleAttachmentUploadOption) => {
    const input = inputRef.current;
    if (!input) return;
    const config = MEDIA_UPLOAD_CONFIG[option.mediaType];
    selectedUploadOptionRef.current = option;
    input.accept = config.accept;
    input.multiple = option.remaining > 1;
    input.click();
  };

  const handlePaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const imageFiles = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((file): file is File => Boolean(file));
    if (!imageFiles.length || !canPasteImages) return;
    event.preventDefault();
    void uploadFiles(imageFiles);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (!canPasteImages) return;
    void uploadFiles(Array.from(event.dataTransfer.files));
  };

  return (
    <div className="w-full">
      <div
        onDragEnter={(event) => {
          event.preventDefault();
          if (canPasteImages) setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "relative isolate overflow-hidden rounded-[28px] border bg-card/76 shadow-[0_6px_20px_-12px_rgba(15,23,42,.4)] backdrop-blur-2xl backdrop-saturate-150 transition-[border-color,box-shadow] duration-150 supports-[backdrop-filter]:bg-card/70 motion-reduce:transition-none",
          dragging
            ? "border-primary/45 ring-4 ring-primary/8"
            : "border-border/55 focus-within:border-primary/30 focus-within:shadow-[0_8px_24px_-14px_rgba(15,23,42,.46)]",
        )}
      >
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={onAdvanced}
          className="absolute right-3 top-3 z-10 rounded-xl bg-background/65 text-muted-foreground shadow-sm backdrop-blur-md hover:bg-background hover:text-foreground"
        >
          <SlidersHorizontal className="h-3 w-3" />
          高级模式
        </Button>

        {attachmentDisabledReason && (
          <div className="flex items-start gap-1.5 px-5 pr-32 pt-4 text-[11px] leading-5 text-amber-600">
            <CircleAlert className="mt-0.5 size-3.5 shrink-0" />
            <span>{attachmentDisabledReason}</span>
          </div>
        )}

        {attachments.length > 0 && (
          <div className="flex gap-2 overflow-x-auto px-4 pr-32 pt-4 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            {attachments.map((attachment) => (
              <div
                key={`${attachment.label}-${attachment.url}`}
                className="group relative grid h-16 w-16 shrink-0 place-items-center overflow-hidden rounded-2xl border border-border/50 bg-muted/30"
              >
                {attachment.mediaType === "image" ? (
                  <SafeImage
                    src={resolveMediaUrl(attachment.url) || undefined}
                    alt={attachment.label}
                    className="h-full w-full object-cover"
                    fallbackType="image"
                  />
                ) : attachment.mediaType === "video" ? (
                  <FileVideo2 className="size-6 text-cyan-500/70" />
                ) : (
                  <Music2 className="size-6 text-violet-500/70" />
                )}
                <span className="absolute inset-x-1 bottom-1 truncate rounded-md bg-black/55 px-1 py-0.5 text-center text-[9px] text-white backdrop-blur-sm">
                  {attachment.label}
                </span>
                <button
                  type="button"
                  onClick={() => onRemoveAttachment(attachment.url)}
                  title={`移除${attachment.label}`}
                  aria-label={`移除${attachment.label}`}
                  className="absolute right-1 top-1 grid h-5 w-5 place-items-center rounded-full bg-black/60 text-white opacity-0 transition-opacity group-hover:opacity-100"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        <Textarea
          value={prompt}
          onChange={(event) => onFormChange({ prompt: event.target.value })}
          onPaste={handlePaste}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && canSubmit) {
              event.preventDefault();
              onSubmit();
            }
          }}
          rows={4}
          maxLength={5000}
          aria-label={mode === "image" ? "图片提示词" : "视频提示词"}
          placeholder={
            mode === "image"
              ? "描述你想生成的画面，也可以直接粘贴参考图片…"
              : "描述画面、动作与镜头，也可以粘贴首帧或参考图片…"
          }
          className={cn(
            "min-h-32 resize-none rounded-none border-0 bg-transparent px-5 pb-4 pr-32 text-[15px] leading-7 shadow-none focus-visible:ring-0",
            attachmentDisabledReason || attachments.length > 0 ? "pt-3" : "pt-5",
          )}
        />

        <div className="mx-2 mb-2 flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-white/15 bg-background/42 p-1.5 shadow-sm backdrop-blur-xl dark:border-white/6">
          <div className="flex min-w-0 flex-wrap items-center gap-1">
            <GenerationAttachmentUploadMenu
              options={attachmentOptions}
              uploading={uploading}
              disabledReason={attachmentDisabledReason}
              onSelect={handleSelectUpload}
            />
            <GenerationQuickSettings
              mode={mode}
              models={models}
              modelId={modelId}
              loadingModels={loadingModels}
              capabilities={capabilities}
              form={form}
              onModelChange={onModelChange}
              onFormChange={onFormChange}
            />
            {attachmentCapacity > 0 && (
              <span className="hidden text-[10px] text-muted-foreground sm:inline">
                {missingRequiredAttachments
                  ? `至少 ${minAttachments} 张图片`
                  : `${attachments.length}/${attachmentCapacity}`}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 pl-1">
            {error && (
              <span
                className="max-w-72 truncate text-[10px] text-destructive"
                title={error}
              >
                {error}
              </span>
            )}
            <span className="hidden text-[10px] tabular-nums text-muted-foreground sm:inline">
              {prompt.length}/5000
            </span>
            <Button
              variant={mode === "image" ? "image" : "video"}
              size="sm"
              disabled={!canSubmit}
              onClick={onSubmit}
              title={
                missingRequiredAttachments
                  ? `至少添加 ${minAttachments} 张图片`
                  : mode === "image"
                    ? "生成图片"
                    : "生成视频"
              }
              aria-label={mode === "image" ? "生成图片" : "生成视频"}
              className="min-w-[76px] rounded-xl px-4"
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              {submitting ? "生成中" : "生成"}
            </Button>
          </div>
        </div>

        {dragging && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-background/88 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-sm font-medium text-primary">
              <ImagePlus className="h-4 w-4" />
              松开即可添加图片
            </div>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        hidden
        type="file"
        accept="image/*,video/*,audio/*"
        onChange={(event) =>
          void uploadFiles(
            Array.from(event.target.files || []),
            selectedUploadOptionRef.current || undefined,
          )
        }
      />
    </div>
  );
}
