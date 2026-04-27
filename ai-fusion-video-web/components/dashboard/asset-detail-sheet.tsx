"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { createPortal } from "react-dom";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import AssetTypePlaceholder from "@/components/dashboard/asset-type-placeholder";
import {
  Save,
  Plus,
  Trash2,
  Loader2,
  Images,
  X,
  Check,
  Upload,
  Maximize2,
  Sparkles,
  Link,
  Library,
} from "lucide-react";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import {
  assetApi,
  type Asset,
  type AssetItem,
  type AssetWithItems,
  type FieldDef,
} from "@/lib/api/asset";

const typeColorMap: Record<string, string> = {
  character: "text-blue-400 bg-blue-500/10",
  scene: "text-green-400 bg-green-500/10",
  prop: "text-amber-400 bg-amber-500/10",
};

const typeLabelMap: Record<string, string> = {
  character: "角色",
  scene: "场景",
  prop: "道具",
};

/** 需要多行编辑的长文本字段 key */
const LONG_TEXT_KEYS = new Set(["description", "relationship", "appearance", "personality"]);

/** 将后端返回的 properties（JSON 字符串或已解析对象）统一解析为 Record */
function parseProps(raw: unknown): Record<string, string> {
  if (!raw) return {};
  if (typeof raw === "string") {
    try { return JSON.parse(raw); } catch { return {}; }
  }
  if (typeof raw === "object") return raw as Record<string, string>;
  return {};
}

const assetTypeOptions = [
  { value: "character", label: "角色" },
  { value: "scene", label: "场景" },
  { value: "prop", label: "道具" },
];

const itemTypeOptions = [
  { value: "initial", label: "初始图" },
  { value: "three_view", label: "三视图" },
  { value: "variant", label: "变体" },
];

// ========== Props 类型 ==========
interface EditProps {
  isCreating?: false;
  asset: Asset;
  projectId?: never;
  onClose: () => void;
  onSaved: () => void;
  onDeleted: () => void;
  onCreated?: never;
}

interface CreateProps {
  isCreating: true;
  asset?: never;
  projectId: number;
  onClose: () => void;
  onSaved: () => void;
  onDeleted?: never;
  onCreated: (created: Asset) => void;
}

type Props = EditProps | CreateProps;

// ========== 全屏大图弹窗 ==========
function ImageLightbox({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
  return createPortal(
    <div
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/80 backdrop-blur-sm cursor-pointer"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-4 right-4 p-2 rounded-full bg-white/10 text-white/80 hover:bg-white/20 transition-colors z-10"
      >
        <X className="h-5 w-5" />
      </button>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
    </div>,
    document.body
  );
}

// ========== 覆盖从库选择弹窗 ==========
function CoverSelectorDialog({ 
  open, 
  onOpenChange, 
  projectId, 
  currentAssetId,
  onSelect 
}: { 
  open: boolean; 
  onOpenChange: (open: boolean) => void; 
  projectId: number; 
  currentAssetId: number;
  onSelect: (url: string) => void; 
}) {
  const [assets, setAssets] = useState<AssetWithItems[]>([]);
  const [loadedProjectId, setLoadedProjectId] = useState<number | null>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const loading = open && projectId > 0 && loadedProjectId !== projectId;

  useEffect(() => {
    if (open && projectId) {
      assetApi.listWithItems(projectId)
        .then(res => {
          setAssets(res);
          setLoadedProjectId(projectId);
          // 默认展开所有资产
          const allIds = new Set(res.map(a => a.id));
          setExpandedIds(allIds);
        })
        .catch(err => {
          console.error(err);
          setAssets([]);
          setLoadedProjectId(projectId);
        });
    }
  }, [open, projectId]);

  const handleExpandToggle = (assetId: number) => {
    setExpandedIds(prev => {
       const next = new Set(prev);
       if (next.has(assetId)) {
         next.delete(assetId);
       } else {
         next.add(assetId);
       }
       return next;
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[calc(100vh-2rem)] sm:max-h-[80vh] flex flex-col overflow-hidden bg-card/95 backdrop-blur-md rounded-2xl p-0 gap-0">
        <DialogHeader className="px-6 py-5 border-b border-border/30 shrink-0">
          <DialogTitle className="text-base font-semibold">从资产库选择封面</DialogTitle>
          <DialogDescription className="text-xs">点击展开项目中的资产，点击图片以选取该图片为当前主资产封面</DialogDescription>
        </DialogHeader>
        <div className="flex-1 overflow-y-auto min-h-0 p-5 space-y-2.5">
          {loading ? (
             <div className="py-12 flex justify-center"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground/40" /></div>
          ) : assets.length === 0 ? (
             <div className="py-12 flex flex-col items-center justify-center text-muted-foreground/30">
               <Images className="h-8 w-8 mb-2" />
               <p className="text-xs">项目中暂无资产</p>
             </div>
          ) : (
            assets.map(a => {
              const isExpanded = expandedIds.has(a.id);
              return (
              <div key={a.id} className="border border-border/30 rounded-xl overflow-hidden bg-card/40 transition-all hover:bg-card/60">
                <div 
                  className="px-4 py-3 flex items-center justify-between cursor-pointer"
                  onClick={() => handleExpandToggle(a.id)}
                >
                  <div className="flex items-center gap-2.5">
                    <span className={cn("inline-block px-1.5 py-0.5 rounded text-[10px] leading-none font-medium", typeColorMap[a.type] || "bg-muted text-muted-foreground")}>
                      {typeLabelMap[a.type] || a.type}
                    </span>
                    <span className="text-[13px] font-medium">{a.name}</span>
                  </div>
                  <div className="text-[11px] text-muted-foreground/60 font-medium">{isExpanded ? '收起' : '展开'}</div>
                </div>
                
                {isExpanded && (
                  <div className="p-4 border-t border-border/30 bg-black/10">
                    {!a.items?.length ? (
                      <div className="py-6 text-center text-xs text-muted-foreground/40">该资产暂无子资产</div>
                    ) : (
                      <div className="grid grid-cols-4 sm:grid-cols-5 gap-3">
                        {a.items.map(item => {
                           const urlStr = item.thumbnailUrl || item.imageUrl;
                           return (
                             <div 
                               key={item.id} 
                               className={cn(
                                 "group relative aspect-square rounded-lg overflow-hidden border border-border/30 transition-all",
                                 urlStr 
                                   ? "cursor-pointer hover:border-primary/50 hover:ring-2 hover:ring-primary/20 hover:shadow-lg" 
                                   : "opacity-60 cursor-not-allowed bg-muted/20"
                               )}
                               onClick={() => {
                                 if(urlStr) {
                                   onSelect(urlStr);
                                   onOpenChange(false);
                                 }
                               }}
                             >
                               {urlStr ? (
                                 // eslint-disable-next-line @next/next/no-img-element
                                 <img src={resolveMediaUrl(urlStr) || ""} alt={item.name || ""} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                               ) : (
                                 <div className="w-full h-full flex flex-col items-center justify-center p-2 text-center text-muted-foreground/60 p-2">
                                   <Images className="h-5 w-5 mb-1.5 opacity-20" />
                                   <span className="text-[10px] break-all line-clamp-2 leading-tight">{item.name || "无名资产"}</span>
                                 </div>
                               )}
                               {urlStr && (
                                 <div className="absolute inset-x-0 bottom-0 p-1.5 bg-gradient-to-t from-black/80 to-transparent text-[10px] text-white/90 truncate opacity-0 group-hover:opacity-100 transition-opacity font-medium">
                                   {item.name || "未命名"}
                                 </div>
                               )}
                             </div>
                           );
                         })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )})
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ========== 子资产编辑面板（滑入式） ==========
function AssetItemEditPanel({
  item,
  assetId,
  assetType,
  projectId,
  onClose,
  onUpdated,
  onDeleted,
}: {
  item: AssetItem;
  assetId: number;
  assetType: string;
  projectId: number;
  onClose: () => void;
  onUpdated: () => void;
  onDeleted: () => void;
}) {
  const [name, setName] = useState(item.name || "");
  const [itemType, setItemType] = useState(item.itemType || "");
  const [imageUrl, setImageUrl] = useState(item.imageUrl || "");
  const [itemProperties, setItemProperties] = useState<Record<string, string>>({});
  const [itemFields, setItemFields] = useState<FieldDef[]>([]);
  const [itemMetaLoading, setItemMetaLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [showLightbox, setShowLightbox] = useState(false);
  const [urlMode, setUrlMode] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setName(item.name || "");
    setItemType(item.itemType || "");
    setImageUrl(item.imageUrl || "");
    const p = parseProps(item.properties);
    setItemProperties(p);
    setDirty(false);
    setShowLightbox(false);
    setUrlMode(false);
  }, [item]);

  useEffect(() => {
    if (!assetType) return;
    let cancelled = false;
    (async () => {
      try {
        setItemMetaLoading(true);
        const resp = await assetApi.getMetadata(assetType);
        if (!cancelled) setItemFields(resp.fields || []);
      } catch {
        if (!cancelled) setItemFields([]);
      } finally {
        if (!cancelled) setItemMetaLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [assetType]);

  const handleItemPropertyChange = (key: string, value: string) => {
    setItemProperties((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      await assetApi.updateItem({
        id: item.id,
        name: name || undefined,
        itemType: itemType || undefined,
        imageUrl: imageUrl || undefined,
        properties: JSON.stringify(itemProperties),
      });
      setDirty(false);
      onUpdated();
    } catch (err) {
      console.error("更新子资产失败:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm("确定删除该子资产？")) return;
    try {
      await assetApi.deleteItem(item.id);
      onDeleted();
    } catch (err) {
      console.error("删除子资产失败:", err);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setImageUrl(result);
      setDirty(true);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImageUrl(reader.result as string);
      setDirty(true);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/30 bg-card/60 backdrop-blur-sm shrink-0">
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-muted/50 transition-colors text-muted-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold truncate">{item.name || "未命名子资产"}</h3>
          <span className="text-[10px] text-muted-foreground/50">子资产编辑</span>
        </div>
        <button
          onClick={handleDelete}
          className="p-1.5 rounded-lg hover:bg-destructive/10 hover:text-destructive transition-colors text-muted-foreground/40"
          title="删除子资产"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={handleSave}
          disabled={!dirty || saving}
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all",
            dirty
              ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm shadow-primary/20"
              : "bg-muted/30 text-muted-foreground/40 cursor-not-allowed"
          )}
        >
          {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
          保存
        </button>
      </div>

      {/* 编辑内容 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-5 space-y-5">
        {/* 图片预览 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <div className="h-1 w-1 rounded-full bg-orange-400" />
            图片
          </h4>
          <div
            className="group/img relative aspect-square w-full rounded-xl border border-border/30 overflow-hidden bg-muted/5"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {imageUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={resolveMediaUrl(imageUrl) || ""}
              alt={name || "子资产图片"}
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground/20">
              <Images className="h-8 w-8 mb-1" />
              <p className="text-[10px]">拖拽图片到此处</p>
            </div>
          )}

          {/* 悬浮操作遮罩 */}
          <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/40 transition-all duration-200 flex items-center justify-center gap-2 opacity-0 group-hover/img:opacity-100">
            {imageUrl && (
              <button
                onClick={() => setShowLightbox(true)}
                className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg bg-white/15 text-white/90 hover:bg-white/25 backdrop-blur-sm transition-all text-[10px] font-medium"
                title="查看大图"
              >
                <Maximize2 className="h-4 w-4" />
                大图
              </button>
            )}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg bg-white/15 text-white/90 hover:bg-white/25 backdrop-blur-sm transition-all text-[10px] font-medium"
              title="上传图片"
            >
              <Upload className="h-4 w-4" />
              上传
            </button>
            <button
              onClick={() => {
                const { addPipeline, setNotificationOpen } = usePipelineStore.getState();
                addPipeline({
                  label: `生成图片: ${item.name || '子资产'}`,
                  projectId,
                  request: {
                    agentType: 'asset_image_gen',
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
              className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg bg-white/15 text-white/90 hover:bg-white/25 backdrop-blur-sm transition-all text-[10px] font-medium"
              title="AI 生图"
            >
              <Sparkles className="h-4 w-4" />
              生图
            </button>
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
          <button
            onClick={() => setUrlMode(!urlMode)}
            className="flex items-center gap-1 text-[10px] text-muted-foreground/60 hover:text-muted-foreground transition-colors"
          >
            <Link className="h-2.5 w-2.5" />
            {urlMode ? "收起链接" : "使用链接"}
          </button>
          {urlMode && (
            <Input
              value={imageUrl.startsWith("data:") ? "" : imageUrl}
              onChange={(e) => { setImageUrl(e.target.value); setDirty(true); }}
              placeholder="粘贴图片链接..."
              className="h-7 text-xs"
            />
          )}
        </div>

        {/* 基本信息 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <div className="h-1 w-1 rounded-full bg-primary" />
            基本信息
          </h4>
          <div className="rounded-xl border border-border/30 bg-card/40 p-3.5 space-y-3">
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">名称</label>
              <Input
                value={name}
                onChange={(e) => { setName(e.target.value); setDirty(true); }}
                placeholder="子资产名称"
                className="h-8 text-xs"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">类型</label>
              <Select value={itemType || ""} onValueChange={(v) => { if (v) { setItemType(v); setDirty(true); } }} items={itemTypeOptions}>
                <SelectTrigger size="sm" className="text-xs">
                  <SelectValue placeholder="选择类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {itemTypeOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value} className="text-xs">{opt.label}</SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        {/* 自定义属性 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <div className="h-1 w-1 rounded-full bg-purple-400" />
            属性
            {itemMetaLoading && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
          </h4>
          {itemFields.length === 0 && !itemMetaLoading ? (
            <div className="rounded-xl border border-dashed border-border/30 py-5 flex justify-center">
              <p className="text-[11px] text-muted-foreground/40">该类型无可编辑属性</p>
            </div>
          ) : (
            <div className="rounded-xl border border-border/30 bg-card/40 p-3.5 grid grid-cols-1 gap-3">
              {itemFields.map((fd) => (
                <div key={fd.key} className="space-y-1.5">
                  <label className="text-xs text-muted-foreground font-medium flex items-center gap-0.5">
                    {fd.label}
                    {fd.required && <span className="text-orange-400 text-[10px]">*</span>}
                  </label>
                  {fd.type === "select" && fd.options ? (
                    <Select
                      value={itemProperties[fd.key] || ""}
                      onValueChange={(v) => handleItemPropertyChange(fd.key, v ?? "")}
                      items={fd.options}
                    >
                      <SelectTrigger size="sm" className="text-xs">
                        <SelectValue placeholder="选择" />
                      </SelectTrigger>
                      <SelectContent className="text-xs">
                        <SelectGroup>
                          {fd.options.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value} className="text-xs">{opt.label}</SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  ) : LONG_TEXT_KEYS.has(fd.key) ? (
                    <Textarea
                      value={itemProperties[fd.key] || ""}
                      onChange={(e) => handleItemPropertyChange(fd.key, e.target.value)}
                      placeholder={fd.label}
                      rows={3}
                      className="resize-none text-xs leading-relaxed"
                    />
                  ) : (
                    <Input
                      value={itemProperties[fd.key] || ""}
                      onChange={(e) => handleItemPropertyChange(fd.key, e.target.value)}
                      placeholder={fd.label}
                      className="h-8 text-xs"
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
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

// ========== 子资产创建面板（滑入式） ==========
function AssetItemCreatePanel({
  assetId,
  assetType,
  onClose,
  onCreated,
}: {
  assetId: number;
  assetType: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [itemType, setItemType] = useState("");
  const [creating, setCreating] = useState(false);
  const [newItemProperties, setNewItemProperties] = useState<Record<string, string>>({});
  const [newItemFields, setNewItemFields] = useState<FieldDef[]>([]);
  const [newItemMetaLoading, setNewItemMetaLoading] = useState(false);

  useEffect(() => {
    if (!assetType) return;
    let cancelled = false;
    (async () => {
      try {
        setNewItemMetaLoading(true);
        const resp = await assetApi.getMetadata(assetType);
        if (!cancelled) setNewItemFields(resp.fields || []);
      } catch {
        if (!cancelled) setNewItemFields([]);
      } finally {
        if (!cancelled) setNewItemMetaLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [assetType]);

  const handleNewItemPropertyChange = (key: string, value: string) => {
    setNewItemProperties((prev) => ({ ...prev, [key]: value }));
  };

  const handleCreate = async () => {
    if (!name.trim()) return;
    try {
      setCreating(true);
      await assetApi.createItem({
        assetId,
        name: name.trim(),
        itemType: itemType || undefined,
        properties: JSON.stringify(newItemProperties),
      });
      onCreated();
    } catch (err) {
      console.error("创建子资产失败:", err);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* 头部 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/30 bg-card/60 backdrop-blur-sm shrink-0">
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-muted/50 transition-colors text-muted-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold truncate">新建子资产</h3>
        </div>
        <button
          onClick={handleCreate}
          disabled={!name.trim() || creating}
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all",
            name.trim()
              ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm shadow-primary/20"
              : "bg-muted/30 text-muted-foreground/40 cursor-not-allowed"
          )}
        >
          {creating ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
          创建
        </button>
      </div>

      {/* 创建表单 */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-5 space-y-5">
        {/* 基本信息 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <div className="h-1 w-1 rounded-full bg-primary" />
            基本信息
          </h4>
          <div className="rounded-xl border border-border/30 bg-card/40 p-3.5 space-y-3">
            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">名称</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="子资产名称"
                className="h-8 text-xs"
                onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
                autoFocus
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-muted-foreground font-medium">类型</label>
              <Select value={itemType || ""} onValueChange={(v) => { if (v) setItemType(v); }} items={itemTypeOptions}>
                <SelectTrigger size="sm" className="text-xs">
                  <SelectValue placeholder="选择类型" />
                </SelectTrigger>
                <SelectContent className="text-xs">
                  <SelectGroup>
                    {itemTypeOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value} className="text-xs">{opt.label}</SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          </div>
        </section>

        {/* 自定义属性 */}
        <section className="space-y-2">
          <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <div className="h-1 w-1 rounded-full bg-purple-400" />
            属性
            {newItemMetaLoading && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
          </h4>
          {newItemFields.length === 0 && !newItemMetaLoading ? (
            <div className="rounded-xl border border-dashed border-border/30 py-5 flex justify-center">
              <p className="text-[11px] text-muted-foreground/40">该类型无可编辑属性</p>
            </div>
          ) : (
            <div className="rounded-xl border border-border/30 bg-card/40 p-3.5 grid grid-cols-1 gap-3">
              {newItemFields.map((fd) => (
                <div key={fd.key} className="space-y-1.5">
                  <label className="text-xs text-muted-foreground font-medium flex items-center gap-0.5">
                    {fd.label}
                    {fd.required && <span className="text-orange-400 text-[10px]">*</span>}
                  </label>
                  {fd.type === "select" && fd.options ? (
                    <Select
                      value={newItemProperties[fd.key] || ""}
                      onValueChange={(v) => handleNewItemPropertyChange(fd.key, v ?? "")}
                      items={fd.options}
                    >
                      <SelectTrigger size="sm" className="text-xs">
                        <SelectValue placeholder="选择" />
                      </SelectTrigger>
                      <SelectContent className="text-xs">
                        <SelectGroup>
                          {fd.options.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value} className="text-xs">{opt.label}</SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  ) : LONG_TEXT_KEYS.has(fd.key) ? (
                    <Textarea
                      value={newItemProperties[fd.key] || ""}
                      onChange={(e) => handleNewItemPropertyChange(fd.key, e.target.value)}
                      placeholder={fd.label}
                      rows={3}
                      className="resize-none text-xs leading-relaxed"
                    />
                  ) : (
                    <Input
                      value={newItemProperties[fd.key] || ""}
                      onChange={(e) => handleNewItemPropertyChange(fd.key, e.target.value)}
                      placeholder={fd.label}
                      className="h-8 text-xs"
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// ========== 主面板 ==========
export default function AssetDetailPanel(props: Props) {
  const { onClose, onSaved } = props;
  const isCreating = props.isCreating === true;
  const asset = isCreating ? null : props.asset;

  // 创建模式的状态
  const [createName, setCreateName] = useState("");
  const [createType, setCreateType] = useState("character");
  const [createDescription, setCreateDescription] = useState("");
  const [createSaving, setCreateSaving] = useState(false);

  // 编辑模式的状态
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [coverUrl, setCoverUrl] = useState("");
  const [properties, setProperties] = useState<Record<string, string>>({});
  const [fields, setFields] = useState<FieldDef[]>([]);
  const [metaLoading, setMetaLoading] = useState(false);
  const [items, setItems] = useState<AssetItem[]>([]);
  const [itemsLoading, setItemsLoading] = useState(false);
  const [saving, setSaving] = useState(false);
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
    setName(asset.name);
    setDescription(asset.description || "");
    setCoverUrl(asset.coverUrl || "");
    const p = parseProps(asset.properties);
    setProperties(p);
    setDirty(false);
    setSelectedItem(null);
    setIsCreatingItem(false);
    loadMetadata(asset.type);
    loadItems(asset.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asset?.id]);

  const loadMetadata = useCallback(async (type: string) => {
    try {
      setMetaLoading(true);
      const resp = await assetApi.getMetadata(type);
      setFields(resp.fields || []);
    } catch {
      setFields([]);
    } finally {
      setMetaLoading(false);
    }
  }, []);

  const loadItems = useCallback(async (assetId: number) => {
    try {
      setItemsLoading(true);
      const data = await assetApi.listItems(assetId);
      setItems(data || []);
    } catch {
      setItems([]);
    } finally {
      setItemsLoading(false);
    }
  }, []);

  const handlePropertyChange = (key: string, value: string) => {
    setProperties((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  const handleSave = async () => {
    if (!asset) return;
    try {
      setSaving(true);
      await assetApi.update({
        id: asset.id,
        name,
        description: description || undefined,
        coverUrl: coverUrl || undefined,
        properties: JSON.stringify(properties),
      });
      setDirty(false);
      onSaved();
    } catch (err) {
      console.error("保存资产失败:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!asset || !props.onDeleted) return;
    if (!confirm("确定要删除该资产及其所有子资产吗？")) return;
    try {
      await assetApi.delete(asset.id);
      props.onDeleted();
    } catch (err) {
      console.error("删除资产失败:", err);
    }
  };

  const handleCreate = async () => {
    if (!isCreating || !createName.trim()) return;
    try {
      setCreateSaving(true);
      const created = await assetApi.create({
        projectId: props.projectId,
        type: createType,
        name: createName.trim(),
        description: createDescription || undefined,
      });
      props.onCreated(created);
    } catch (err) {
      console.error("创建资产失败:", err);
    } finally {
      setCreateSaving(false);
    }
  };

  const typeColor = asset ? (typeColorMap[asset.type] || "text-muted-foreground bg-muted/50") : "";

  // ========== 创建模式 ==========
  if (isCreating) {
    return (
      <div className="h-full flex flex-col">
        {/* 头部 */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border/20 shrink-0">
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-muted/50 transition-colors text-muted-foreground"
          >
            <X className="h-4 w-4" />
          </button>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold">新建资产</h3>
          </div>
          <button
            onClick={handleCreate}
            disabled={!createName.trim() || createSaving}
            className={cn(
              "flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
              createName.trim()
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted/30 text-muted-foreground/40 cursor-not-allowed"
            )}
          >
            {createSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
            创建
          </button>
        </div>

        {/* 表单 */}
        <div className="flex-1 min-h-0 overflow-y-auto px-4 py-4 space-y-4">
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">名称</label>
            <Input
              value={createName}
              onChange={(e) => setCreateName(e.target.value)}
              placeholder="资产名称"
              className="h-8 text-xs"
              onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              autoFocus
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">类型</label>
            <Select value={createType} onValueChange={(v) => { if (v) setCreateType(v); }} items={assetTypeOptions}>
              <SelectTrigger size="sm" className="text-xs">
                <SelectValue placeholder="选择类型" />
              </SelectTrigger>
              <SelectContent className="text-xs">
                <SelectGroup>
                  {assetTypeOptions.map((t) => (
                    <SelectItem key={t.value} value={t.value} className="text-xs">{t.label}</SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <label className="text-xs text-muted-foreground">描述</label>
            <Textarea
              value={createDescription}
              onChange={(e) => setCreateDescription(e.target.value)}
              placeholder="资产描述（可选）..."
              rows={3}
              className="resize-none text-xs"
            />
          </div>
        </div>
      </div>
    );
  }

  // ========== 编辑模式 ==========
  if (!asset) return null;

  return (
    <div className="h-full relative overflow-hidden">
      {/* ======= 左侧：主资产编辑 ======= */}
      <div
        className="h-full flex flex-col min-h-0 overflow-hidden border-r border-border/20 transition-all duration-400 ease-in-out"
        style={{ width: isItemPanelOpen ? "50%" : "100%" }}
      >
        {/* 头部 */}
        <div className="flex items-center gap-3 px-5 py-3.5 border-b border-border/30 bg-card/60 backdrop-blur-sm shrink-0">
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-muted/50 transition-colors text-muted-foreground"
          >
            <X className="h-4 w-4" />
          </button>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold truncate">{asset.name}</h3>
            <span className={cn("inline-block px-1.5 py-0.5 rounded text-[9px] leading-none mt-0.5 font-medium", typeColor)}>
              {typeLabelMap[asset.type] || asset.type}
            </span>
          </div>
          <button
            onClick={handleDelete}
            className="p-1.5 rounded-lg hover:bg-destructive/10 hover:text-destructive transition-colors text-muted-foreground/40"
            title="删除资产"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleSave}
            disabled={!dirty || saving}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
              dirty
                ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm shadow-primary/20"
                : "bg-muted/30 text-muted-foreground/40 cursor-not-allowed"
            )}
          >
            {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
            保存
          </button>
        </div>

        {/* 滚动内容 */}
        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-5 space-y-6">
          {/* ======= 基本信息 ======= */}
          <section className="space-y-3">
            <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <div className="h-1 w-1 rounded-full bg-primary" />
              基本信息
            </h4>
            <div className="rounded-xl border border-border/30 bg-card/40 p-4 space-y-3">
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-medium">名称</label>
                <Input
                  value={name}
                  onChange={(e) => { setName(e.target.value); setDirty(true); }}
                  placeholder="资产名称"
                  className="h-8 text-xs"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs text-muted-foreground font-medium">描述</label>
                <Textarea
                  value={description}
                  onChange={(e) => { setDescription(e.target.value); setDirty(true); }}
                  placeholder="资产描述..."
                  rows={3}
                  className="resize-none text-xs leading-relaxed"
                />
              </div>
            </div>
            <div className="space-y-2 mt-4">
              <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <div className="h-1 w-1 rounded-full bg-orange-400" />
                封面
              </h4>
              <div
                className="group/cover relative h-[280px] w-full rounded-xl border border-border/30 overflow-hidden bg-muted/5"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files?.[0];
                  if (!file || !file.type.startsWith("image/")) return;
                  const reader = new FileReader();
                  reader.onload = () => { setCoverUrl(reader.result as string); setDirty(true); };
                  reader.readAsDataURL(file);
                }}
              >
                {coverUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={resolveMediaUrl(coverUrl) || ""} alt={name || "封面"} className="w-full h-full object-contain" />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-muted-foreground/20">
                    <Images className="h-8 w-8 mb-1" />
                    <p className="text-[10px]">拖拽图片到此处</p>
                  </div>
                )}
                {/* 悬浮操作遮罩 */}
                <div className="absolute inset-0 bg-black/0 group-hover/cover:bg-black/40 transition-all duration-200 flex items-center justify-center gap-2 opacity-0 group-hover/cover:opacity-100">
                  {coverUrl && (
                    <button
                      onClick={() => setShowCoverLightbox(true)}
                      className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg bg-white/15 text-white/90 hover:bg-white/25 backdrop-blur-sm transition-all text-[10px] font-medium"
                      title="查看大图"
                    >
                      <Maximize2 className="h-4 w-4" />
                      大图
                    </button>
                  )}
                  <button
                    onClick={() => setIsCoverSelectorOpen(true)}
                    className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg bg-white/15 text-white/90 hover:bg-white/25 backdrop-blur-sm transition-all text-[10px] font-medium"
                    title="从资料库"
                  >
                    <Library className="h-4 w-4" />
                    资料库
                  </button>
                  <button
                    onClick={() => coverFileRef.current?.click()}
                    className="flex flex-col items-center gap-1 px-3 py-2 rounded-lg bg-white/15 text-white/90 hover:bg-white/25 backdrop-blur-sm transition-all text-[10px] font-medium"
                    title="上传图片"
                  >
                    <Upload className="h-4 w-4" />
                    上传
                  </button>
                </div>
                <input
                  ref={coverFileRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    const reader = new FileReader();
                    reader.onload = () => { setCoverUrl(reader.result as string); setDirty(true); };
                    reader.readAsDataURL(file);
                  }}
                />
              </div>
              {/* URL 输入切换 */}
              <div className="space-y-1">
                <button
                  onClick={() => setCoverUrlMode(!coverUrlMode)}
                  className="flex items-center gap-1 text-[10px] text-muted-foreground/60 hover:text-muted-foreground transition-colors"
                >
                  <Link className="h-2.5 w-2.5" />
                  {coverUrlMode ? "收起链接" : "使用链接"}
                </button>
                {coverUrlMode && (
                  <Input
                    value={coverUrl.startsWith("data:") ? "" : coverUrl}
                    onChange={(e) => { setCoverUrl(e.target.value); setDirty(true); }}
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

          {/* ======= 动态属性 ======= */}
          <section className="space-y-3">
            <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <div className="h-1 w-1 rounded-full bg-purple-400" />
              属性
              {metaLoading && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
            </h4>
            {fields.length === 0 && !metaLoading ? (
              <div className="rounded-xl border border-dashed border-border/30 py-6 flex justify-center">
                <p className="text-[11px] text-muted-foreground/40">该类型无可编辑属性</p>
              </div>
            ) : (
              <div className="rounded-xl border border-border/30 bg-card/40 p-4 grid grid-cols-2 gap-3">
                {fields.map((fd) => (
                  <div
                    key={fd.key}
                    className={cn(
                      "space-y-1",
                      (fd.key === "description" || fd.key === "relationship" || fd.key === "appearance" || fd.key === "personality")
                        ? "col-span-2" : ""
                    )}
                  >
                    <label className="text-xs text-muted-foreground font-medium flex items-center gap-0.5">
                      {fd.label}
                      {fd.required && <span className="text-orange-400 text-[10px]">*</span>}
                    </label>
                    {fd.type === "select" && fd.options ? (
                      <Select
                        value={properties[fd.key] || ""}
                        onValueChange={(v) => handlePropertyChange(fd.key, v ?? "")}
                        items={fd.options}
                      >
                        <SelectTrigger size="sm" className="text-xs">
                          <SelectValue placeholder="选择" />
                        </SelectTrigger>
                        <SelectContent className="text-xs">
                          <SelectGroup>
                            {fd.options.map((opt) => (
                              <SelectItem key={opt.value} value={opt.value} className="text-xs">{opt.label}</SelectItem>
                            ))}
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    ) : LONG_TEXT_KEYS.has(fd.key) ? (
                      <Textarea
                        value={properties[fd.key] || ""}
                        onChange={(e) => handlePropertyChange(fd.key, e.target.value)}
                        placeholder={fd.label}
                        rows={3}
                        className="resize-none text-xs leading-relaxed"
                      />
                    ) : (
                      <Input
                        value={properties[fd.key] || ""}
                        onChange={(e) => handlePropertyChange(fd.key, e.target.value)}
                        placeholder={fd.label}
                        className="h-8 text-xs"
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* ======= 子资产卡片列表 ======= */}
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <div className="h-1 w-1 rounded-full bg-cyan-400" />
                子资产
                {itemsLoading && <Loader2 className="h-2.5 w-2.5 animate-spin" />}
              </h4>
              <div className="flex items-center gap-2.5">
                <span className="text-[10px] text-muted-foreground/40 tabular-nums">{items.length} 个</span>
                <button
                  onClick={() => { setSelectedItem(null); setIsCreatingItem(true); }}
                  className="flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                >
                  <Plus className="h-2.5 w-2.5" />
                  新建
                </button>
              </div>
            </div>

            {/* 子资产卡片网格 */}
            {items.length === 0 && !itemsLoading ? (
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
                        "group relative rounded-xl border cursor-pointer transition-all duration-200",
                        "bg-card/40 hover:bg-card/60 hover:shadow-sm",
                        isItemSelected
                          ? "border-primary/40 ring-1 ring-primary/20 shadow-sm shadow-primary/5"
                          : "border-border/30 hover:border-border/50"
                      )}
                      onClick={() => { setIsCreatingItem(false); setSelectedItem(item); }}
                    >
                      {/* 缩略图 */}
                      <div className="aspect-square overflow-hidden rounded-t-[7px]">
                        {(item.imageUrl || item.thumbnailUrl) ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={resolveMediaUrl(item.thumbnailUrl || item.imageUrl) || ""}
                            alt={item.name || ""}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <AssetTypePlaceholder type={asset.type} className="w-full h-full" iconSize="h-5 w-5" />
                        )}
                        {/* 删除按钮 */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (!confirm("确定删除该子资产？")) return;
                            assetApi.deleteItem(item.id).then(() => {
                              if (selectedItem?.id === item.id) setSelectedItem(null);
                              loadItems(asset.id);
                            });
                          }}
                          className="absolute top-1 right-1 p-0.5 rounded bg-black/40 text-white/70 hover:bg-destructive hover:text-white opacity-0 group-hover:opacity-100 transition-all backdrop-blur-sm"
                        >
                          <Trash2 className="h-2.5 w-2.5" />
                        </button>
                      </div>
                      {/* 信息 */}
                      <div className="px-2.5 py-2">
                        <p className="text-[11px] font-medium truncate">{item.name || "未命名"}</p>
                        <span className="inline-block mt-1 px-1.5 py-0.5 rounded-md text-[9px] font-medium bg-foreground/5 text-muted-foreground/50">
                          {itemTypeOptions.find((o) => o.value === item.itemType)?.label || item.itemType || "未分类"}
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
          "transition-all duration-400 ease-in-out"
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
            onClose={() => setIsCreatingItem(false)}
            onCreated={() => {
              setIsCreatingItem(false);
              loadItems(asset.id);
            }}
          />
        ) : selectedItem ? (
          <AssetItemEditPanel
            key={selectedItem.id}
            item={selectedItem}
            assetId={asset.id}
            assetType={asset.type}
            projectId={asset.projectId}
            onClose={() => setSelectedItem(null)}
            onUpdated={() => loadItems(asset.id)}
            onDeleted={() => {
              setSelectedItem(null);
              loadItems(asset.id);
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
