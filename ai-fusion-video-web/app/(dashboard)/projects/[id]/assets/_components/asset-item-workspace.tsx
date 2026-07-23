"use client";

import { useMemo, useRef, useState } from "react";
import {
  Check,
  ImagePlus,
  Images,
  Loader2,
  Plus,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import type { AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { uploadFile } from "@/lib/api/storage";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SafeImage } from "@/components/ui/safe-image";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { getAppearance, getAppearanceHints } from "./asset-types";

interface AssetItemWorkspaceProps {
  mode: "create" | "edit";
  item: AssetItem | null;
  assetType: string;
  saving: boolean;
  onCreate: (
    name: string,
    appearance: string,
    imageUrl: string,
    generateAfterCreate: boolean,
  ) => Promise<void>;
  onSave: (
    item: AssetItem,
    name: string,
    appearance: string,
    imageUrl: string,
  ) => Promise<void>;
  onDelete: () => void;
  onCancelCreate: () => void;
  onGenerate: () => void;
}

export function AssetItemWorkspace({
  mode,
  item,
  assetType,
  saving,
  onCreate,
  onSave,
  onDelete,
  onCancelCreate,
  onGenerate,
}: AssetItemWorkspaceProps) {
  const initialName = item?.name || "";
  const initialAppearance = getAppearance(item);
  const initialImageUrl = item?.imageUrl || "";
  const [name, setName] = useState(initialName);
  const [appearance, setAppearance] = useState(initialAppearance);
  const [imageUrl, setImageUrl] = useState(initialImageUrl);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const appearanceHints = useMemo(
    () => getAppearanceHints(assetType),
    [assetType],
  );
  const fallbackType =
    assetType === "character"
      ? "avatar"
      : assetType === "scene"
        ? "scene"
        : assetType === "prop"
          ? "prop"
          : "image";
  const dirty =
    name !== initialName ||
    appearance !== initialAppearance ||
    imageUrl !== initialImageUrl;
  const disabled = saving || uploading || !name.trim();

  const upload = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("请选择图片文件");
      return;
    }

    setUploading(true);
    setError("");
    try {
      setImageUrl(await uploadFile(file, "assets"));
    } catch {
      setError("图片上传失败");
    } finally {
      setUploading(false);
    }
  };

  const appendHint = (hint: string) => {
    const token = `${hint}：`;
    setAppearance((current) => {
      if (current.includes(token)) return current;
      return current.trim() ? `${current.trim()}\n${token}` : token;
    });
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const save = async () => {
    if (!item || disabled) return;
    await onSave(item, name, appearance, imageUrl);
  };

  const create = async (generateAfterCreate: boolean) => {
    if (disabled) return;
    await onCreate(name, appearance, imageUrl, generateAfterCreate);
  };

  const generate = async () => {
    if (mode === "create") {
      await create(true);
      return;
    }
    if (item && dirty) await onSave(item, name, appearance, imageUrl);
    onGenerate();
  };

  return (
    <section className="grid min-h-[360px] min-w-0 bg-[radial-gradient(circle_at_18%_20%,color-mix(in_oklab,var(--primary)_5%,transparent),transparent_32%)] xl:grid-cols-[minmax(330px,0.9fr)_minmax(420px,1.1fr)]">
      <div className="min-w-0 p-5">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold">资产图</h3>
            <span
              className={cn(
                "rounded-md px-1.5 py-0.5 text-[10px] font-medium",
                imageUrl
                  ? "bg-emerald-500/10 text-emerald-600"
                  : "bg-amber-500/10 text-amber-600",
              )}
            >
              {imageUrl ? "已添加" : "待添加"}
            </span>
          </div>
          {imageUrl && (
            <Button
              variant="ghost"
              size="xs"
              onClick={() => setImageUrl("")}
            >
              移除图片
            </Button>
          )}
        </div>

        <div
          role="button"
          tabIndex={0}
          onClick={() => !imageUrl && inputRef.current?.click()}
          onKeyDown={(event) => {
            if (!imageUrl && (event.key === "Enter" || event.key === " ")) {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files?.[0];
            if (file) void upload(file);
          }}
          className={cn(
            "group relative mx-auto aspect-[16/10] w-full max-w-[480px] overflow-hidden rounded-2xl border bg-muted/15 outline-none transition-all focus-visible:ring-2 focus-visible:ring-ring/50",
            dragging
              ? "border-primary bg-primary/5 ring-2 ring-primary/15"
              : "border-border/50",
            !imageUrl && "cursor-pointer border-dashed hover:border-primary/35 hover:bg-primary/[0.025]",
          )}
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
              <span className="grid h-12 w-12 place-items-center rounded-2xl bg-primary/8 text-primary/65 transition-transform group-hover:scale-105">
                {uploading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Images className="h-5 w-5" />
                )}
              </span>
              <p className="mt-3 text-sm font-medium text-foreground/75">
                {dragging ? "松开即可上传" : "拖入图片或点击上传"}
              </p>
            </div>
          )}
        </div>

        {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

        <div className="mx-auto mt-3 grid w-full max-w-[480px] grid-cols-2 gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Upload className="h-3.5 w-3.5" />
            )}
            {imageUrl ? "替换图片" : "上传图片"}
          </Button>
          <Button
            size="sm"
            variant="image"
            disabled={disabled}
            onClick={() => void generate()}
          >
            {saving ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <ImagePlus className="h-3.5 w-3.5" />
            )}
            {mode === "create"
              ? "创建并生图"
              : dirty
                ? "保存并生图"
                : "生成图片"}
          </Button>
        </div>
        <input
          ref={inputRef}
          hidden
          type="file"
          accept="image/*"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
            event.target.value = "";
          }}
        />
      </div>

      <div className="flex min-w-0 flex-col border-t border-border/40 bg-background/35 p-5 xl:border-l xl:border-t-0">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-primary">
                {mode === "create" ? "新增" : "子资产"}
              </p>
              <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                {mode === "create" ? "草稿" : imageUrl ? "有图片" : "待补图"}
              </span>
            </div>
            <h2 className="mt-1 truncate text-xl font-semibold tracking-tight">
              {mode === "create" ? "新建子资产" : name || "未命名"}
            </h2>
          </div>
          <Button
            variant={mode === "create" ? "ghost" : "destructive-ghost"}
            size="icon-sm"
            onClick={mode === "create" ? onCancelCreate : onDelete}
            title={mode === "create" ? "取消新建" : "删除子资产"}
            aria-label={mode === "create" ? "取消新建" : "删除子资产"}
          >
            {mode === "create" ? (
              <X className="h-4 w-4" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
          </Button>
        </div>

        <div className="mt-4 space-y-3.5">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">名称</label>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="子资产名称"
              className="h-9 rounded-lg bg-background/85"
              autoFocus={mode === "create"}
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs font-medium text-muted-foreground">
                外观描写
              </label>
              <span className="text-[10px] tabular-nums text-muted-foreground/70">
                {appearance.length} 字
              </span>
            </div>
            <Textarea
              ref={textareaRef}
              value={appearance}
              onChange={(event) => setAppearance(event.target.value)}
              rows={6}
              placeholder="描述材质、颜色、形态和细节"
              className="min-h-32 rounded-lg bg-background/85 text-sm leading-5"
            />
          </div>

          <div>
            <p className="mb-2 text-[10px] font-medium text-muted-foreground">
              描写要点
            </p>
            <div className="flex flex-wrap gap-1.5">
              {appearanceHints.map((hint) => (
                <button
                  key={hint}
                  type="button"
                  onClick={() => appendHint(hint)}
                  className="h-6 rounded-md border border-border/50 bg-background/65 px-2 text-[10px] text-muted-foreground transition-colors hover:border-primary/30 hover:bg-primary/5 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                >
                  + {hint}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-auto flex items-center justify-between gap-3 border-t border-border/40 pt-4">
          <span className="text-[10px] text-muted-foreground">
            {mode === "create" ? "尚未创建" : dirty ? "有未保存修改" : "已保存"}
          </span>
          <div className="flex items-center gap-2">
            {mode === "create" && (
              <Button variant="outline" size="sm" onClick={onCancelCreate}>
                取消
              </Button>
            )}
            <Button
              size="sm"
              disabled={disabled || (mode === "edit" && !dirty)}
              onClick={() => void (mode === "create" ? create(false) : save())}
            >
              {saving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : mode === "create" ? (
                <Plus className="h-3.5 w-3.5" />
              ) : (
                <Check className="h-3.5 w-3.5" />
              )}
              {mode === "create" ? "创建" : "保存"}
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
