"use client";
import { AssetPropertiesEditor } from "./asset-properties-editor";
import { Button } from "@/components/ui/button";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { Input } from "@/components/ui/input";
import { SafeImage } from "@/components/media/managed-image";
import { Textarea } from "@/components/ui/textarea";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { assetApi, type AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import { uploadFile } from "@/lib/api/storage";
import { cn } from "@/lib/utils";
import {
  Images,
  Library,
  Link,
  Loader2,
  Maximize2,
  Plus,
  Save,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { AssetItemCreatePanel } from "./asset-item-create-panel";
import { AssetItemEditPanel } from "./asset-item-edit-panel";
import { CoverSelectorDialog } from "./cover-selector-dialog";
import { ImageLightbox } from "./image-lightbox";
import {
  itemTypeOptions,
  parseProps,
  typeColorMap,
  typeLabelMap,
  type EditProps,
} from "./shared";
export default function AssetEditPanel(props: EditProps) {
  const { confirm } = useConfirm();
  const { onClose, onSaved } = props;
  const asset = props.asset;
  const saveRequest = useCommandRequestKey();
  const deleteRequest = useCommandRequestKey();
  const itemDeleteRequest = useCommandRequestKey();
  const [baseVersion, setBaseVersion] = useState<number | null>(
    asset?.version ?? null,
  );
  // 编辑模式的状态
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [coverUrl, setCoverUrl] = useState("");
  const [properties, setProperties] = useState<Record<string, unknown>>({});
  const [items, setItems] = useState<AssetItem[]>([]);
  const [itemsLoading, setItemsLoading] = useState(false);
  const [itemsError, setItemsError] = useState("");
  const itemRequestSequence = useRef(0);
  const [saving, setSaving] = useState(false);
  const [coverUploading, setCoverUploading] = useState(false);
  const [dirty, setDirty] = useState(false);
  const coverFileRef = useRef<HTMLInputElement>(null);
  const [showCoverLightbox, setShowCoverLightbox] = useState(false);
  const [isCoverSelectorOpen, setIsCoverSelectorOpen] = useState(false);
  const [coverUrlMode, setCoverUrlMode] = useState(false);
  // 选中的子资产 / 创建子资产模式
  const [selectedItem, setSelectedItem] = useState<AssetItem | null>(null);
  const [isCreatingItem, setIsCreatingItem] = useState(false);
  const isItemPanelOpen = !!selectedItem || isCreatingItem;
  useEffect(() => {
    if (!asset) return;
    setBaseVersion(asset.version);
    setName(asset.name);
    setDescription(asset.description || "");
    setCoverUrl(asset.coverUrl || "");
    const p = parseProps(asset.properties);
    setProperties(p);
    setDirty(false);
    setSelectedItem(null);
    setIsCreatingItem(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asset?.id]);
  useEffect(() => {
    if (
      !asset ||
      dirty ||
      (baseVersion !== null && asset.version <= baseVersion)
    )
      return;
    setBaseVersion(asset.version);
    setName(asset.name);
    setDescription(asset.description || "");
    setCoverUrl(asset.coverUrl || "");
    setProperties(parseProps(asset.properties));
  }, [asset, dirty, baseVersion]);
  const loadItems = useCallback(async (assetId: number) => {
    const sequence = ++itemRequestSequence.current;
    try {
      setItemsLoading(true);
      setItemsError("");
      const data = await assetApi.listItems(assetId);
      if (sequence !== itemRequestSequence.current) return;
      setItems(data);
      setSelectedItem((current) =>
        current ? data.find((item) => item.id === current.id) || null : null,
      );
    } catch (error) {
      if (sequence === itemRequestSequence.current)
        setItemsError(getApiErrorMessage(error));
    } finally {
      if (sequence === itemRequestSequence.current) setItemsLoading(false);
    }
  }, []);
  useEffect(() => {
    void loadItems(asset.id);
    const sequenceRef = itemRequestSequence;
    return () => {
      sequenceRef.current++;
    };
  }, [asset.id, asset.version, loadItems]);
  const handlePropertyChange = (key: string, value: string | null) => {
    setProperties((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };
  const uploadCoverImage = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    try {
      setCoverUploading(true);
      const uploadedUrl = await uploadFile(file, "assets");
      setCoverUrl(uploadedUrl);
      setDirty(true);
    } catch (err) {
      console.error("上传资产封面失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setCoverUploading(false);
    }
  }, []);
  const handleSave = async () => {
    if (!asset || baseVersion === null || saving || coverUploading || !dirty)
      return;
    try {
      setSaving(true);
      const payload = {
        id: asset.id,
        version: baseVersion,
        name,
        description: description || null,
        coverUrl: coverUrl || null,
        properties,
      };
      const updated = await assetApi.update(
        payload,
        saveRequest.forPayload(payload),
      );
      saveRequest.clear();
      setBaseVersion(updated.version);
      setDirty(false);
      onSaved();
    } catch (err) {
      console.error("保存资产失败:", err);
      toast.error(getApiErrorMessage(err));
    } finally {
      setSaving(false);
    }
  };
  const handleDelete = async () => {
    if (
      !asset ||
      !props.onDeleted ||
      baseVersion === null ||
      saving ||
      coverUploading
    )
      return;
    const ok = await confirm({
      title: "删除主资产",
      description:
        "确定要删除该资产及其所有子资产吗？已有引用会显示失效，媒体文件会保留。",
      variant: "destructive",
      confirmText: "确定删除",
    });
    if (!ok) return;
    try {
      await assetApi.delete(
        asset.id,
        baseVersion,
        deleteRequest.forPayload({ id: asset.id, version: baseVersion }),
      );
      deleteRequest.clear();
      props.onDeleted();
    } catch (err) {
      console.error("删除资产失败:", err);
      toast.error(getApiErrorMessage(err));
    }
  };
  const typeColor = asset
    ? typeColorMap[asset.type] || "text-muted-foreground bg-muted/50"
    : "";
  // ========== 编辑模式 ==========
  if (!asset) return null;
  return (
    <div className="h-full relative overflow-hidden">
      {/* ======= 左侧：主资产编辑 ======= */}
      <div
        className="h-full flex flex-col min-h-0 overflow-hidden border-r border-border/20 transition-colors duration-200 motion-reduce:transition-none"
        style={{ width: isItemPanelOpen ? "50%" : "100%" }}
      >
        {/* 头部 */}
        <div className="flex items-center gap-3 px-5 py-3.5 border-b border-border/30 bg-card/60 backdrop-blur-sm shrink-0">
          <Button variant="ghost" size="icon-sm" title="关闭" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold truncate">{asset.name}</h3>
            <span
              className={cn(
                "inline-block px-1.5 py-0.5 rounded text-[9px] leading-none mt-0.5 font-medium",
                typeColor,
              )}
            >
              {typeLabelMap[asset.type] || asset.type}
            </span>
          </div>
          <Button
            variant="destructive-ghost"
            size="icon-sm"
            onClick={handleDelete}
            title="删除资产"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="default"
            size="sm"
            title="保存"
            onClick={handleSave}
            disabled={!dirty || saving || coverUploading}
          >
            {saving || coverUploading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Save className="h-3 w-3" />
            )}
            保存
          </Button>
        </div>

        {/* 滚动内容 */}
        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-5 space-y-6">
          {/* ======= 基本信息 ======= */}
          <section className="space-y-3">
            <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              基本信息
            </h4>
            <div className="rounded-xl border border-border/30 bg-card/40 p-4 space-y-3">
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-medium">
                  名称
                </label>
                <Input
                  disabled={saving || coverUploading}
                  value={name}
                  onChange={(e) => {
                    setName(e.target.value);
                    setDirty(true);
                  }}
                  placeholder="资产名称"
                  className="h-8 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-medium">
                  描述
                </label>
                <Textarea
                  disabled={saving || coverUploading}
                  value={description}
                  onChange={(e) => {
                    setDescription(e.target.value);
                    setDirty(true);
                  }}
                  placeholder="资产描述..."
                  rows={3}
                  className="resize-none text-xs leading-relaxed"
                />
              </div>
            </div>
            <div className="space-y-2 mt-4">
              <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                封面
              </h4>
              <div
                className="group/cover relative h-[280px] w-full rounded-xl border border-border/30 overflow-hidden bg-muted/5"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files?.[0];
                  if (
                    saving ||
                    coverUploading ||
                    !file ||
                    !file.type.startsWith("image/")
                  )
                    return;
                  void uploadCoverImage(file);
                }}
              >
                {coverUrl ? (
                  <SafeImage
                    src={resolveMediaUrl(coverUrl) || undefined}
                    alt={name || "封面"}
                    className="w-full h-full object-contain"
                    fallbackType={
                      asset?.type === "character"
                        ? "avatar"
                        : asset?.type === "scene"
                          ? "scene"
                          : asset?.type === "prop"
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
                <div className="absolute inset-0 bg-black/0 group-hover/cover:bg-black/40 transition-colors duration-200 flex items-center justify-center gap-2 opacity-0 group-hover/cover:opacity-100 z-20">
                  {coverUrl && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowCoverLightbox(true)}
                      title="查看大图"
                    >
                      <Maximize2 className="h-4 w-4" />
                      大图
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={saving || coverUploading}
                    onClick={() => setIsCoverSelectorOpen(true)}
                    title="从资料库"
                  >
                    <Library className="h-4 w-4" />
                    资料库
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={saving || coverUploading}
                    onClick={() => coverFileRef.current?.click()}
                    title="上传图片"
                  >
                    <Upload className="h-4 w-4" />
                    上传
                  </Button>
                </div>
                <input
                  ref={coverFileRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    e.target.value = "";
                    if (!file) return;
                    void uploadCoverImage(file);
                  }}
                />
              </div>
              {/* URL 输入切换 */}
              <div className="space-y-1">
                <Button
                  variant="ghost"
                  size="sm"
                  title="使用链接"
                  onClick={() => setCoverUrlMode(!coverUrlMode)}
                >
                  <Link className="h-2.5 w-2.5" />
                  {coverUrlMode ? "收起链接" : "使用链接"}
                </Button>
                {coverUrlMode && (
                  <Input
                    disabled={saving || coverUploading}
                    value={coverUrl}
                    onChange={(e) => {
                      setCoverUrl(e.target.value);
                      setDirty(true);
                    }}
                    placeholder="粘贴封面图片链接..."
                    className="h-7 text-xs"
                  />
                )}
              </div>
            </div>
            {/* 大图弹窗 */}
            {showCoverLightbox && coverUrl && (
              <ImageLightbox
                src={resolveMediaUrl(coverUrl) || ""}
                alt={name || "封面"}
                onClose={() => setShowCoverLightbox(false)}
              />
            )}
          </section>

          <AssetPropertiesEditor
            assetType={asset.type}
            value={properties}
            onChange={handlePropertyChange}
            disabled={saving || coverUploading}
          />

          {/* ======= 子资产卡片列表 ======= */}
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                子资产
                {itemsLoading && (
                  <Loader2 className="h-2.5 w-2.5 animate-spin" />
                )}
              </h4>
              <div className="flex items-center gap-2.5">
                <span className="text-[10px] text-muted-foreground/40 tabular-nums">
                  {items.length} 个
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  title="操作"
                  onClick={() => {
                    setSelectedItem(null);
                    setIsCreatingItem(true);
                  }}
                >
                  <Plus className="h-2.5 w-2.5" />
                  新建
                </Button>
              </div>
            </div>

            {/* 子资产卡片网格 */}
            {itemsError ? (
              <div role="alert" className="space-y-2 text-sm text-destructive">
                <p>{itemsError}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => void loadItems(asset.id)}
                >
                  重试
                </Button>
              </div>
            ) : items.length === 0 && !itemsLoading ? (
              <div className="rounded-xl border border-dashed border-border/30 py-8 flex flex-col items-center text-muted-foreground/25">
                <Images className="h-6 w-6 mb-1.5" />
                <p className="text-[11px]">暂无子资产</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2.5">
                {items.map((item) => {
                  const isItemSelected = selectedItem?.id === item.id;
                  return (
                    <div
                      key={item.id}
                      className={cn(
                        "group relative rounded-xl border cursor-pointer transition-colors duration-200",
                        "bg-card/40 hover:bg-card/60 hover:shadow-sm",
                        isItemSelected
                          ? "border-primary/40 ring-1 ring-primary/20 shadow-sm shadow-primary/5"
                          : "border-border/30 hover:border-border/50",
                      )}
                      onClick={() => {
                        setIsCreatingItem(false);
                        setSelectedItem(item);
                      }}
                    >
                      {/* 缩略图 */}
                      <div className="aspect-square overflow-hidden rounded-t-[7px]">
                        <SafeImage
                          src={
                            resolveMediaUrl(
                              item.thumbnailUrl || item.imageUrl,
                            ) || undefined
                          }
                          alt={item.name || ""}
                          className="w-full h-full object-cover"
                          fallbackType={
                            asset?.type === "character"
                              ? "avatar"
                              : asset?.type === "scene"
                                ? "scene"
                                : asset?.type === "prop"
                                  ? "prop"
                                  : "image"
                          }
                        />
                        {/* 删除按钮 */}
                        <Button
                          variant="destructive-ghost"
                          size="icon-sm"
                          title="删除"
                          onClick={async (e) => {
                            e.stopPropagation();
                            const ok = await confirm({
                              title: "删除子资产",
                              description:
                                "确定删除该子资产吗？已有引用会显示失效，媒体文件会保留。",
                              variant: "destructive",
                              confirmText: "确定删除",
                            });
                            if (!ok) return;
                            assetApi
                              .deleteItem(
                                item.id,
                                item.version,
                                item.assetVersion,
                                itemDeleteRequest.forPayload({
                                  id: item.id,
                                  version: item.version,
                                  parentVersion: item.assetVersion,
                                }),
                              )
                              .then(() => {
                                itemDeleteRequest.clear();
                                onSaved();
                                if (selectedItem?.id === item.id)
                                  setSelectedItem(null);
                                loadItems(asset.id);
                              })
                              .catch((error) =>
                                toast.error(getApiErrorMessage(error)),
                              );
                          }}
                          className="absolute top-1 right-1 z-10"
                        >
                          <Trash2 className="h-2.5 w-2.5" />
                        </Button>
                      </div>
                      {/* 信息 */}
                      <div className="px-2.5 py-2">
                        <p className="text-[11px] font-medium truncate">
                          {item.name || "未命名"}
                        </p>
                        <span className="inline-block mt-1 px-1.5 py-0.5 rounded-md text-[9px] font-medium bg-foreground/5 text-muted-foreground/50">
                          {itemTypeOptions.find(
                            (o) => o.value === item.itemType,
                          )?.label ||
                            item.itemType ||
                            "未分类"}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>
      </div>

      {/* ======= 右侧：子资产编辑/创建面板（absolute 定位，从右侧滑入） ======= */}
      <div
        className={cn(
          "absolute top-0 right-0 h-full min-h-0 overflow-hidden bg-card/30",
          "transition-colors duration-200 motion-reduce:transition-none",
        )}
        style={{
          width: isItemPanelOpen ? "50%" : "0%",
          opacity: isItemPanelOpen ? 1 : 0,
        }}
      >
        {isCreatingItem ? (
          <AssetItemCreatePanel
            assetId={asset.id}
            assetType={asset.type}
            parentVersion={asset.version}
            onClose={() => setIsCreatingItem(false)}
            onCreated={() => {
              setIsCreatingItem(false);
              loadItems(asset.id);
              onSaved();
            }}
          />
        ) : selectedItem ? (
          <AssetItemEditPanel
            confirm={confirm}
            key={selectedItem.id}
            item={selectedItem}
            assetId={asset.id}
            assetType={asset.type}
            projectId={asset.projectId}
            onClose={() => setSelectedItem(null)}
            onUpdated={() => {
              loadItems(asset.id);
              onSaved();
            }}
            onDeleted={() => {
              setSelectedItem(null);
              loadItems(asset.id);
              onSaved();
            }}
          />
        ) : null}
      </div>

      <CoverSelectorDialog
        open={isCoverSelectorOpen}
        onOpenChange={setIsCoverSelectorOpen}
        projectId={asset.projectId}
        currentAssetId={asset.id}
        onSelect={(url) => {
          setCoverUrl(url);
          setDirty(true);
        }}
      />
    </div>
  );
}
