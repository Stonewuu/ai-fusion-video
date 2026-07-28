"use client";

import React, { useRef, useState } from "react";
import { toast } from "sonner";
import { Upload, Link, Images, X, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { resolveMediaUrl } from "@/lib/api/client";
import { uploadFile } from "@/lib/api/storage";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { cn } from "@/lib/utils";

interface ImageInputProps {
  value: string;
  onChange: (url: string) => void;
  className?: string;
  /** 预览区高度 class，默认 h-32 */
  previewHeight?: string;
  previewContainerClassName?: string;
  previewImageClassName?: string;
  onPreviewClick?: () => void;
  placeholder?: string;
  uploadSubDir?: string;
  beforeUpload?: () => boolean;
}

/**
 * 图片输入组件：支持 URL 填写 / 本地文件上传 切换
 */
export default function ImageInput({
  value,
  onChange,
  className,
  previewHeight = "h-32",
  previewContainerClassName,
  previewImageClassName,
  onPreviewClick,
  placeholder = "粘贴图片链接...",
  uploadSubDir = "images",
  beforeUpload,
}: ImageInputProps) {
  const [mode, setMode] = useState<"url" | "upload">("upload");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      return;
    }
    try {
      setUploading(true);
      const uploadedUrl = await uploadFile(file, uploadSubDir);
      onChange(uploadedUrl);
    } catch (error) {
      console.error("图片上传失败:", error);
      toast.error(getApiErrorMessage(error));
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (beforeUpload && !beforeUpload()) return;
    void handleUpload(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    if (beforeUpload && !beforeUpload()) return;
    void handleUpload(file);
  };

  const handleClear = () => {
    onChange("");
    if (fileRef.current) fileRef.current.value = "";
  };

  const commitUrl = (rawUrl: string) => {
    const nextUrl = rawUrl.trim();
    if (nextUrl !== value) {
      onChange(nextUrl);
    }
  };

  const handlePreviewKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (!onPreviewClick) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onPreviewClick();
    }
  };

  return (
    <div className={cn("space-y-2", className)}>
      {/* 模式切换标签 */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => setMode("upload")}
          className={cn(
            "flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium transition-all",
            mode === "upload"
              ? "bg-foreground/10 text-foreground"
              : "text-muted-foreground/60 hover:text-muted-foreground"
          )}
        >
          <Upload className="h-2.5 w-2.5" />
          上传
        </button>
        <button
          type="button"
          onClick={() => setMode("url")}
          className={cn(
            "flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium transition-all",
            mode === "url"
              ? "bg-foreground/10 text-foreground"
              : "text-muted-foreground/60 hover:text-muted-foreground"
          )}
        >
          <Link className="h-2.5 w-2.5" />
          链接
        </button>
      </div>

      {/* 预览区 / 拖拽区 */}
      {value ? (
        <div
          className={cn(
            "relative rounded-lg overflow-hidden border border-border/20 bg-muted/10 group",
            onPreviewClick && "cursor-zoom-in",
            previewHeight,
            previewContainerClassName
          )}
          role={onPreviewClick ? "button" : undefined}
          tabIndex={onPreviewClick ? 0 : undefined}
          onClick={onPreviewClick}
          onKeyDown={handlePreviewKeyDown}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={resolveMediaUrl(value) || ""}
            alt="preview"
            className={cn("w-full h-full", previewImageClassName || "object-cover")}
          />
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              handleClear();
            }}
            className="absolute top-1.5 right-1.5 p-1 rounded-md bg-black/50 text-white/80 hover:bg-black/70 opacity-0 group-hover:opacity-100 transition-all backdrop-blur-sm"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ) : mode === "upload" ? (
        <div
          className={cn(
            "rounded-lg border-2 border-dashed border-border/30 bg-muted/5 flex flex-col items-center justify-center cursor-pointer",
            "hover:border-primary/30 hover:bg-primary/5 transition-all",
            uploading && "cursor-wait opacity-70",
            previewHeight
          )}
          onClick={() => !uploading && fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {uploading ? (
            <Loader2 className="h-5 w-5 text-muted-foreground/40 mb-1.5 animate-spin" />
          ) : (
            <Upload className="h-5 w-5 text-muted-foreground/30 mb-1.5" />
          )}
          <p className="text-[10px] text-muted-foreground/40">
            {uploading ? "上传中..." : "点击或拖拽图片到此处"}
          </p>
        </div>
      ) : (
        <div className={cn(
          "rounded-lg border border-dashed border-border/30 bg-muted/5 flex flex-col items-center justify-center",
          previewHeight
        )}>
          <Images className="h-5 w-5 text-muted-foreground/20 mb-1" />
          <p className="text-[10px] text-muted-foreground/30">暂无图片</p>
        </div>
      )}

      {/* 隐藏的 file input */}
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* URL 输入框 */}
      {mode === "url" && (
        <Input
          key={value || "empty-url"}
          defaultValue={value}
          onBlur={(e) => commitUrl(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commitUrl(e.currentTarget.value);
              e.currentTarget.blur();
            }
          }}
          placeholder={placeholder}
          className="h-7 text-xs"
        />
      )}
    </div>
  );
}
