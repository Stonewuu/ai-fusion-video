"use client";

import { useRef, useState, type Ref } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ImagePlus, Images, Loader2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { resolveMediaUrl } from "@/lib/api/client";
import { cn } from "@/lib/utils";
import { AssetItemMessage } from "./asset-item-feedback";

interface Props {
  imageUrl: string;
  name: string;
  assetType: string;
  uploading: boolean;
  saving: boolean;
  generationDisabled: boolean;
  generationLabel: string;
  error: string;
  onUpload: (file: File) => void;
  onRemove: () => void;
  onGenerate: () => void;
  uploadButtonRef: Ref<HTMLButtonElement>;
}

export function AssetItemImage({
  imageUrl,
  name,
  assetType,
  uploading,
  saving,
  generationDisabled,
  generationLabel,
  error,
  onUpload,
  onRemove,
  onGenerate,
  uploadButtonRef,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const reduceMotion = useReducedMotion();
  const busy = saving || uploading;
  const fallbackType =
    assetType === "character"
      ? "avatar"
      : assetType === "scene"
        ? "scene"
        : assetType === "prop"
          ? "prop"
          : "image";
  const chooseFile = () => {
    if (!busy) {
      inputRef.current?.click();
    }
  };
  return (
    <div className="min-w-0 p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">资产图</h3>
          <span
            className={cn(
              "rounded-md px-1.5 py-0.5 text-xs font-medium",
              imageUrl
                ? "bg-emerald-500/10 text-emerald-600"
                : "bg-amber-500/10 text-amber-600",
            )}
          >
            {imageUrl ? "已添加" : "待添加"}
          </span>
        </div>
        <Button
          type="button"
          variant="ghost"
          disabled={!imageUrl || busy}
          onClick={onRemove}
        >
          移除图片
        </Button>
      </div>
      <div
        role="button"
        tabIndex={busy ? -1 : 0}
        aria-label={imageUrl ? "资产图片" : "拖入图片或点击上传"}
        aria-disabled={busy}
        onClick={() => {
          if (!imageUrl) {
            chooseFile();
          }
        }}
        onKeyDown={(event) => {
          if (!imageUrl && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            chooseFile();
          }
        }}
        onDragEnter={(event) => {
          event.preventDefault();
          if (!busy) {
            setDragging(true);
          }
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file && !busy) {
            onUpload(file);
          }
        }}
        className={cn(
          "group relative mx-auto aspect-[16/10] w-full max-w-[480px] overflow-hidden rounded-lg border bg-background/70 outline-none transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-ring/50 motion-reduce:transition-none",
          dragging
            ? "border-primary bg-primary/5 ring-2 ring-primary/15"
            : "border-input",
          !imageUrl &&
            !busy &&
            "cursor-pointer border-dashed hover:border-primary/35 hover:bg-primary/5",
        )}
      >
        <AnimatePresence initial={false}>
          <motion.div
            key={imageUrl || "empty"}
            className="absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.2 }}
          >
            {imageUrl ? (
              <SafeImage
                src={resolveMediaUrl(imageUrl) || undefined}
                alt={name || "子资产"}
                className="h-full w-full object-contain p-3"
                fallbackType={fallbackType}
              />
            ) : (
              <div className="flex h-full flex-col items-center justify-center text-center text-muted-foreground">
                <span className="grid h-12 w-12 place-items-center rounded-xl bg-primary/10 text-primary">
                  <Images className="size-5" />
                </span>
                <p className="mt-3 text-sm font-medium text-foreground/75">
                  {dragging ? "松开即可上传" : "拖入图片或点击上传"}
                </p>
              </div>
            )}
          </motion.div>
          {uploading && (
            <motion.div
              key="uploading"
              className="absolute inset-0 flex items-center justify-center gap-2 bg-background/80 text-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: reduceMotion ? 0 : 0.15 }}
            >
              <Loader2 className="size-4 animate-spin" />
              正在上传
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      <div className="mx-auto w-full max-w-[480px]">
        <AssetItemMessage message={error} />
      </div>
      <div className="mx-auto mt-3 grid w-full max-w-[480px] grid-cols-2 gap-2">
        <Button
          ref={uploadButtonRef}
          type="button"
          variant="outline"
          onClick={chooseFile}
          disabled={busy}
        >
          {uploading ? <Loader2 className="animate-spin" /> : <Upload />}
          {imageUrl ? "替换图片" : "上传图片"}
        </Button>
        <Button
          type="button"
          variant="image"
          disabled={generationDisabled}
          onClick={onGenerate}
        >
          {saving ? <Loader2 className="animate-spin" /> : <ImagePlus />}
          {generationLabel}
        </Button>
      </div>
      <input
        ref={inputRef}
        hidden
        disabled={busy}
        type="file"
        accept="image/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onUpload(file);
          }
          event.target.value = "";
        }}
      />
    </div>
  );
}
