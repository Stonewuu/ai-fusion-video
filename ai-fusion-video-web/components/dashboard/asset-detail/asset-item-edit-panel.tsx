"use client";
import { AssetPropertiesEditor } from "./asset-properties-editor";
import { Button } from "@/components/ui/button";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { Input } from "@/components/ui/input";
import { SafeImage } from "@/components/media/managed-image";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { assetApi, type AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { uploadFile } from "@/lib/api/storage";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import {
  Check,
  Images,
  Link,
  Loader2,
  Maximize2,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { ImageLightbox } from "./image-lightbox";
import { itemTypeOptions, parseProps } from "./shared";
export function AssetItemEditPanel({
  confirm,
  item,
  assetId,
  assetType,
  projectId,
  onClose,
  onUpdated,
  onDeleted,
}: {
  confirm: ReturnType<typeof useConfirm>["confirm"];
  item: AssetItem;
  assetId: number;
  assetType: string;
  projectId: number | null;
  onClose: () => void;
  onUpdated: () => void;
  onDeleted: () => void;
}) {
  const requests = useCommandRequestKey();
  const deletions = useCommandRequestKey();
  const [version, setVersion] = useState(item.version);
  const [parentVersion, setParentVersion] = useState(item.assetVersion);
  const [name, setName] = useState(item.name || "");
  const [itemType, setItemType] = useState(item.itemType || "");
  const [imageUrl, setImageUrl] = useState(item.imageUrl || "");
  const [itemProperties, setItemProperties] = useState<Record<string, unknown>>(
    parseProps(item.properties),
  );
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [showLightbox, setShowLightbox] = useState(false);
  const [urlMode, setUrlMode] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (
      dirty ||
      saving ||
      item.version < version ||
      item.assetVersion < parentVersion
    )
      return;
    setVersion(item.version);
    setParentVersion(item.assetVersion);
    setName(item.name || "");
    setItemType(item.itemType || "");
    setImageUrl(item.imageUrl || "");
    const p = parseProps(item.properties);
    setItemProperties(p);
    setDirty(false);
    setShowLightbox(false);
    setUrlMode(false);
  }, [item, dirty, saving, version, parentVersion]);
  const handleItemPropertyChange = (key: string, value: string | null) => {
    setItemProperties((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };
  const handleSave = async () => {
    if (saving || uploading || !dirty) return;
    try {
      setSaving(true);
      const payload = {
        id: item.id,
        version,
        assetVersion: parentVersion,
        name: name || null,
        itemType: itemType || undefined,
        imageUrl: imageUrl || null,
        properties: itemProperties,
      };
      const updated = await assetApi.updateItem(
        payload,
        requests.forPayload(payload),
      );
      requests.clear();
      setVersion(updated.version);
      setParentVersion(updated.assetVersion);
      setName(updated.name || "");
      setImageUrl(updated.imageUrl || "");
      setItemProperties(parseProps(updated.properties));
      setDirty(false);
      onUpdated();
    } catch (err) {
      console.error("更新子资产失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };
  const handleDelete = async () => {
    if (saving || uploading) return;
    const ok = await confirm({
      title: "删除子资产",
      description: "确定删除该子资产吗？已有引用会显示失效，媒体文件会保留。",
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!ok) return;
    try {
      await assetApi.deleteItem(
        item.id,
        version,
        parentVersion,
        deletions.forPayload({ id: item.id, version, parentVersion }),
      );
      deletions.clear();
      onDeleted();
    } catch (err) {
      console.error("删除子资产失败:", err);
      toast.error(getApiErrorMessage(err));
    }
  };
  const uploadItemImage = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    try {
      setUploading(true);
      const uploadedUrl = await uploadFile(file, "assets");
      setImageUrl(uploadedUrl);
      setDirty(true);
    } catch (err) {
      console.error("上传子资产图片失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setUploading(false);
    }
  }, []);
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    void uploadItemImage(file);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (saving || uploading || !file || !file.type.startsWith("image/")) return;
    void uploadItemImage(file);
  };
  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/30 bg-card/60 backdrop-blur-sm shrink-0">
        <Button variant="ghost" size="icon-sm" title="关闭" onClick={onClose}>
          <X className="h-3.5 w-3.5" />
        </Button>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold truncate">
            {item.name || "未命名子资产"}
          </h3>
          <span className="text-[10px] text-muted-foreground/50">
            子资产编辑
          </span>
        </div>
        <Button
          variant="destructive-ghost"
          size="icon-sm"
          onClick={handleDelete}
          title="删除子资产"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="default"
          size="sm"
          title="保存"
          onClick={handleSave}
          disabled={!dirty || saving || uploading}
        >
          {saving || uploading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Check className="h-3 w-3" />
          )}
          保存
        </Button>
      </div>

      {/* 编辑内容 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-5 space-y-5">
        {/* 图片预览 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            图片
          </h4>
          <div
            className="group/img relative aspect-square w-full rounded-xl border border-border/30 overflow-hidden bg-muted/5"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            {imageUrl ? (
              <SafeImage
                src={resolveMediaUrl(imageUrl) || undefined}
                alt={name || "子资产图片"}
                className="w-full h-full object-contain"
                fallbackType={
                  assetType === "character"
                    ? "avatar"
                    : assetType === "scene"
                      ? "scene"
                      : assetType === "prop"
                        ? "prop"
                        : "image"
                }
              />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground/20">
                <Images className="h-8 w-8 mb-1" />
                <p className="text-[10px]">拖拽图片到此处</p>
              </div>
            )}

            {/* 悬浮操作遮罩 */}
            <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/40 transition-colors duration-200 flex items-center justify-center gap-2 opacity-0 group-hover/img:opacity-100 z-20">
              {imageUrl && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowLightbox(true)}
                  title="查看大图"
                >
                  <Maximize2 className="h-4 w-4" />
                  大图
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                disabled={saving || uploading}
                onClick={() => fileInputRef.current?.click()}
                title="上传图片"
              >
                <Upload className="h-4 w-4" />
                上传
              </Button>
              <Button
                variant="image"
                size="sm"
                onClick={() => {
                  if (projectId === null) return;
                  const { addPipeline, setNotificationOpen } =
                    usePipelineStore.getState();
                  addPipeline({
                    label: `生成图片: ${item.name || "子资产"}`,
                    projectId,
                    request: {
                      agentType: "asset_image_gen",
                      toolExecutionMode: "FULL_ACCESS",
                      projectId,
                      context: {
                        selectedAssetIds: [assetId],
                        selectedAssetItemIds: [item.id],
                      },
                    },
                    onComplete: () => {
                      onUpdated();
                    },
                  });
                  setNotificationOpen(true);
                }}
                disabled={projectId === null || dirty || saving || uploading}
                title={
                  projectId === null
                    ? "关联项目后使用资产生图"
                    : dirty
                      ? "请先保存修改"
                      : "AI 生图"
                }
              >
                <Sparkles className="h-4 w-4" />
                生图
              </Button>
            </div>

            {/* 隐藏文件输入 */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileUpload}
            />
          </div>
        </section>

        {/* URL 输入切换 */}
        <div className="space-y-1">
          <Button
            variant="ghost"
            size="sm"
            title="使用链接"
            onClick={() => setUrlMode(!urlMode)}
          >
            <Link className="h-2.5 w-2.5" />
            {urlMode ? "收起链接" : "使用链接"}
          </Button>
          {urlMode && (
            <Input
              disabled={saving || uploading}
              value={imageUrl}
              onChange={(e) => {
                setImageUrl(e.target.value);
                setDirty(true);
              }}
              placeholder="粘贴图片链接..."
              className="h-7 text-xs"
            />
          )}
        </div>

        {/* 基本信息 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            基本信息
          </h4>
          <div className="rounded-xl border border-border/30 bg-card/40 p-3.5 space-y-3">
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">
                名称
              </label>
              <Input
                disabled={saving || uploading}
                value={name}
                onChange={(e) => {
                  setName(e.target.value);
                  setDirty(true);
                }}
                placeholder="子资产名称"
                className="h-8 text-xs"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">
                类型
              </label>
              <Select
                disabled={saving || uploading}
                value={itemType || ""}
                onValueChange={(v) => {
                  if (v) {
                    setItemType(v);
                    setDirty(true);
                  }
                }}
                items={itemTypeOptions}
              >
                <SelectTrigger size="sm" className="text-xs">
                  <SelectValue placeholder="选择类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {itemTypeOptions.map((opt) => (
                      <SelectItem
                        key={opt.value}
                        value={opt.value}
                        className="text-xs"
                      >
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        <AssetPropertiesEditor
          assetType={assetType}
          value={itemProperties}
          onChange={handleItemPropertyChange}
          disabled={saving || uploading}
        />
      </div>

      {/* 大图弹窗 */}
      {showLightbox && imageUrl && (
        <ImageLightbox
          src={resolveMediaUrl(imageUrl) || ""}
          alt={name || "子资产图片"}
          onClose={() => setShowLightbox(false)}
        />
      )}
    </div>
  );
}
