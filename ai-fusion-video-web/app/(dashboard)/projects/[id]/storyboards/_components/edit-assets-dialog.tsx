"use client";

import { useState, useEffect, useMemo } from "react";
import { X, Users, MapPin, Package, Check, Loader2, ZoomIn } from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { AssetWithItems, AssetItem } from "@/lib/api/asset";
import type { StoryboardItem } from "@/lib/api/storyboard";
import { SafeImage } from "@/components/ui/safe-image";

interface EditItemAssetsDialogProps {
  open: boolean;
  onClose: () => void;
  item: StoryboardItem | null;
  assetsList: AssetWithItems[];
  onConfirm: (data: {
    characterIds: number[];
    sceneAssetItemId: number | null;
    propIds: number[];
  }) => Promise<void> | void;
}

/** 安全解析 ID 数组 */
function parseIds(raw: number[] | string | null | undefined): number[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

export function EditItemAssetsDialog({
  open,
  onClose,
  item,
  assetsList,
  onConfirm,
 }: EditItemAssetsDialogProps) {
  // 解析已关联的资产
  const initialCharacterIds = useMemo(() => new Set(parseIds(item?.characterIds)), [item?.characterIds]);
  const initialSceneAssetItemId = item?.sceneAssetItemId || null;
  const initialPropIds = useMemo(() => new Set(parseIds(item?.propIds)), [item?.propIds]);

  const [selectedCharacterIds, setSelectedCharacterIds] = useState<Set<number>>(new Set());
  const [selectedSceneAssetItemId, setSelectedSceneAssetItemId] = useState<number | null>(null);
  const [selectedPropIds, setSelectedPropIds] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [previewImage, setPreviewImage] = useState<{ url: string; title: string } | null>(null);

  // 当弹窗打开或绑定的镜头改变时重置状态
  useEffect(() => {
    if (open && item) {
      setSelectedCharacterIds(new Set(initialCharacterIds));
      setSelectedSceneAssetItemId(initialSceneAssetItemId);
      setSelectedPropIds(new Set(initialPropIds));
    }
  }, [open, item?.id, initialCharacterIds, initialSceneAssetItemId, initialPropIds]);
  const characterAssets = useMemo(() => assetsList.filter((a) => a.type === "character"), [assetsList]);
  const sceneAssets = useMemo(() => assetsList.filter((a) => a.type === "scene"), [assetsList]);
  const propAssets = useMemo(() => assetsList.filter((a) => a.type === "prop"), [assetsList]);

  // 展开所有子资产变体为扁平列表
  const characterItems = useMemo(() => {
    return characterAssets.flatMap((asset) =>
      (asset.items || []).map((subItem) => ({
        ...subItem,
        asset,
      }))
    );
  }, [characterAssets]);

  const sceneItems = useMemo(() => {
    return sceneAssets.flatMap((asset) =>
      (asset.items || []).map((subItem) => ({
        ...subItem,
        asset,
      }))
    );
  }, [sceneAssets]);

  const propItems = useMemo(() => {
    return propAssets.flatMap((asset) =>
      (asset.items || []).map((subItem) => ({
        ...subItem,
        asset,
      }))
    );
  }, [propAssets]);

  if (!open || !item) return null;

  const toggleCharacter = (itemId: number) => {
    setSelectedCharacterIds((prev) => {
      const next = new Set(prev);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  };

  const toggleScene = (itemId: number) => {
    setSelectedSceneAssetItemId((prev) => (prev === itemId ? null : itemId));
  };

  const toggleProp = (itemId: number) => {
    setSelectedPropIds((prev) => {
      const next = new Set(prev);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      const charIdsArr = Array.from(selectedCharacterIds);
      const propIdsArr = Array.from(selectedPropIds);

      await onConfirm({
        characterIds: charIdsArr,
        sceneAssetItemId: selectedSceneAssetItemId,
        propIds: propIdsArr,
      });
      onClose();
    } catch (err) {
      console.error("保存关联资产失败:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog Container */}
      <div className="relative bg-card border border-border/30 rounded-2xl shadow-2xl w-[600px] max-w-[95vw] max-h-[85vh] flex flex-col overflow-hidden z-10">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/20 shrink-0">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-linear-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
              <Users className="h-4 w-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">关联镜头资产</h3>
              <p className="text-[10px] text-muted-foreground">
                设置分镜镜头 #{item.shotNumber || item.autoShotNumber || "?"} 的关联资产
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content Container (Independent Scroll) */}
        <div className="flex-1 min-h-0 overflow-y-auto p-5 space-y-5">
          {/* Characters Section */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-blue-400 uppercase tracking-wider flex items-center gap-1.5 pl-0.5">
              <Users className="h-3.5 w-3.5" /> 角色关联 (多选)
            </h4>
            {characterItems.length === 0 ? (
              <p className="text-[11px] text-muted-foreground/60 italic pl-0.5">暂无项目角色资产，请先在资产库中添加</p>
            ) : (
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2.5">
                {characterItems.map((item) => {
                  const isChecked = selectedCharacterIds.has(item.id);
                  const showSubName = item.name && item.name !== item.asset.name && item.name !== "默认变体" && item.name !== "默认";
                  return (
                    <button
                      key={item.id}
                      onClick={() => toggleCharacter(item.id)}
                      className={cn(
                        "group relative flex flex-col rounded-xl border overflow-hidden transition-all text-left bg-muted/5",
                        isChecked
                          ? "border-blue-500 ring-2 ring-blue-500/20 bg-blue-500/[0.02]"
                          : "border-border/30 hover:border-border hover:bg-muted/15"
                      )}
                    >
                      {/* Checkbox Overlay */}
                      <div
                        className={cn(
                          "absolute top-1.5 right-1.5 z-10 h-4.5 w-4.5 rounded-md border flex items-center justify-center backdrop-blur-sm transition-all",
                          isChecked
                            ? "bg-blue-500 border-blue-500 text-white shadow-sm"
                            : "bg-black/30 border-white/20"
                        )}
                      >
                        {isChecked && <Check className="h-3 w-3 stroke-[3]" />}
                      </div>

                      {/* Zoom Overlay */}
                      {item.imageUrl && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setPreviewImage({
                              url: item.imageUrl!,
                              title: `${item.asset.name} - ${showSubName ? item.name : "初始设定"}`
                            });
                          }}
                          className="absolute top-1.5 left-1.5 z-10 h-4.5 w-4.5 rounded-md border bg-black/45 hover:bg-black/75 border-white/20 flex items-center justify-center backdrop-blur-sm transition-all opacity-0 group-hover:opacity-100"
                        >
                          <ZoomIn className="h-2.5 w-2.5 text-white" />
                        </button>
                      )}

                      {/* Image Thumbnail */}
                      <div className="w-full aspect-[4/3] bg-muted/20 overflow-hidden relative flex items-center justify-center border-b border-border/10">
                        {item.imageUrl ? (
                          <SafeImage
                            src={resolveMediaUrl(item.imageUrl)}
                            alt={item.name || item.asset.name}
                            fallbackType="avatar"
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                          />
                        ) : (
                          <Users className="h-5 w-5 text-muted-foreground/30" />
                        )}
                      </div>

                      {/* Text Info */}
                      <div className="p-2 min-w-0 flex-1 flex flex-col justify-center">
                        <p className="text-[11px] font-semibold truncate text-foreground/90">{item.asset.name}</p>
                        <p className="text-[9px] text-muted-foreground truncate">
                          {showSubName ? item.name : "初始设定"}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Scenes Section */}
          <div className="space-y-3 pt-2.5 border-t border-border/10">
            <h4 className="text-xs font-semibold text-green-400 uppercase tracking-wider flex items-center gap-1.5 pl-0.5">
              <MapPin className="h-3.5 w-3.5" /> 场景关联 (单选)
            </h4>
            {sceneItems.length === 0 ? (
              <p className="text-[11px] text-muted-foreground/60 italic pl-0.5">暂无项目场景资产，请先在资产库中添加</p>
            ) : (
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2.5">
                {sceneItems.map((item) => {
                  const isChecked = selectedSceneAssetItemId === item.id;
                  const showSubName = item.name && item.name !== item.asset.name && item.name !== "默认变体" && item.name !== "默认";
                  return (
                    <button
                      key={item.id}
                      onClick={() => toggleScene(item.id)}
                      className={cn(
                        "group relative flex flex-col rounded-xl border overflow-hidden transition-all text-left bg-muted/5",
                        isChecked
                          ? "border-green-500 ring-2 ring-green-500/20 bg-green-500/[0.02]"
                          : "border-border/30 hover:border-border hover:bg-muted/15"
                      )}
                    >
                      {/* Radio Overlay */}
                      <div
                        className={cn(
                          "absolute top-1.5 right-1.5 z-10 h-4.5 w-4.5 rounded-full border flex items-center justify-center backdrop-blur-sm transition-all",
                          isChecked
                            ? "bg-green-500 border-green-500 text-white shadow-sm"
                            : "bg-black/30 border-white/20"
                        )}
                      >
                        {isChecked && <Check className="h-3 w-3 stroke-[3]" />}
                      </div>

                      {/* Zoom Overlay */}
                      {item.imageUrl && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setPreviewImage({
                              url: item.imageUrl!,
                              title: `${item.asset.name} - ${showSubName ? item.name : "初始设定"}`
                            });
                          }}
                          className="absolute top-1.5 left-1.5 z-10 h-4.5 w-4.5 rounded-md border bg-black/45 hover:bg-black/75 border-white/20 flex items-center justify-center backdrop-blur-sm transition-all opacity-0 group-hover:opacity-100"
                        >
                          <ZoomIn className="h-2.5 w-2.5 text-white" />
                        </button>
                      )}

                      {/* Image Thumbnail */}
                      <div className="w-full aspect-[4/3] bg-muted/20 overflow-hidden relative flex items-center justify-center border-b border-border/10">
                        {item.imageUrl ? (
                          <SafeImage
                            src={resolveMediaUrl(item.imageUrl)}
                            alt={item.name || item.asset.name}
                            fallbackType="scene"
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                          />
                        ) : (
                          <MapPin className="h-5 w-5 text-muted-foreground/30" />
                        )}
                      </div>

                      {/* Text Info */}
                      <div className="p-2 min-w-0 flex-1 flex flex-col justify-center">
                        <p className="text-[11px] font-semibold truncate text-foreground/90">{item.asset.name}</p>
                        <p className="text-[9px] text-muted-foreground truncate">
                          {showSubName ? item.name : "初始设定"}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Props Section */}
          <div className="space-y-3 pt-2.5 border-t border-border/10">
            <h4 className="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center gap-1.5 pl-0.5">
              <Package className="h-3.5 w-3.5" /> 道具关联 (多选)
            </h4>
            {propItems.length === 0 ? (
              <p className="text-[11px] text-muted-foreground/60 italic pl-0.5">暂无项目道具资产，请先在资产库中添加</p>
            ) : (
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2.5">
                {propItems.map((item) => {
                  const isChecked = selectedPropIds.has(item.id);
                  const showSubName = item.name && item.name !== item.asset.name && item.name !== "默认变体" && item.name !== "默认";
                  return (
                    <button
                      key={item.id}
                      onClick={() => toggleProp(item.id)}
                      className={cn(
                        "group relative flex flex-col rounded-xl border overflow-hidden transition-all text-left bg-muted/5",
                        isChecked
                          ? "border-amber-500 ring-2 ring-amber-500/20 bg-amber-500/[0.02]"
                          : "border-border/30 hover:border-border hover:bg-muted/15"
                      )}
                    >
                      {/* Checkbox Overlay */}
                      <div
                        className={cn(
                          "absolute top-1.5 right-1.5 z-10 h-4.5 w-4.5 rounded-md border flex items-center justify-center backdrop-blur-sm transition-all",
                          isChecked
                            ? "bg-amber-500 border-amber-500 text-white shadow-sm"
                            : "bg-black/30 border-white/20"
                        )}
                      >
                        {isChecked && <Check className="h-3 w-3 stroke-[3]" />}
                      </div>

                      {/* Zoom Overlay */}
                      {item.imageUrl && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setPreviewImage({
                              url: item.imageUrl!,
                              title: `${item.asset.name} - ${showSubName ? item.name : "初始设定"}`
                            });
                          }}
                          className="absolute top-1.5 left-1.5 z-10 h-4.5 w-4.5 rounded-md border bg-black/45 hover:bg-black/75 border-white/20 flex items-center justify-center backdrop-blur-sm transition-all opacity-0 group-hover:opacity-100"
                        >
                          <ZoomIn className="h-2.5 w-2.5 text-white" />
                        </button>
                      )}

                      {/* Image Thumbnail */}
                      <div className="w-full aspect-[4/3] bg-muted/20 overflow-hidden relative flex items-center justify-center border-b border-border/10">
                        {item.imageUrl ? (
                          <SafeImage
                            src={resolveMediaUrl(item.imageUrl)}
                            alt={item.name || item.asset.name}
                            fallbackType="prop"
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                          />
                        ) : (
                          <Package className="h-5 w-5 text-muted-foreground/30" />
                        )}
                      </div>

                      {/* Text Info */}
                      <div className="p-2 min-w-0 flex-1 flex flex-col justify-center">
                        <p className="text-[11px] font-semibold truncate text-foreground/90">{item.asset.name}</p>
                        <p className="text-[9px] text-muted-foreground truncate">
                          {showSubName ? item.name : "初始设定"}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3.5 border-t border-border/20 flex items-center justify-end gap-2 shrink-0 bg-background/50">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={loading}
            className={cn(
              "flex items-center gap-1.5 px-5 py-2 rounded-xl text-xs font-medium transition-all",
              "bg-primary text-primary-foreground",
              "hover:opacity-90 active:scale-[0.98]",
              "disabled:opacity-50 disabled:pointer-events-none"
            )}
          >
            {loading && <Loader2 className="h-3 w-3 animate-spin" />}
            保存
          </button>
        </div>
      </div>

      {/* 图片预览 Modal */}
      {previewImage && (
        <div className="modal-overlay fixed inset-0 z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="absolute inset-0" onClick={() => setPreviewImage(null)} />
          <div className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-3 z-10">
            <button
              onClick={() => setPreviewImage(null)}
              className="absolute -top-10 right-0 rounded-full border border-border/40 bg-background/80 p-1.5 text-foreground shadow-xl backdrop-blur-xl transition-colors hover:bg-background"
            >
              <X className="h-5 w-5" />
            </button>
            <SafeImage
              src={resolveMediaUrl(previewImage.url)}
              alt={previewImage.title}
              fallbackType="image"
              className="max-w-full max-h-[80vh] rounded-lg object-contain shadow-2xl border border-border/40"
            />
            <p className="rounded-full border border-border/40 bg-background/80 px-3 py-1.5 text-xs font-medium text-foreground shadow-xl backdrop-blur-xl">
              {previewImage.title}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
