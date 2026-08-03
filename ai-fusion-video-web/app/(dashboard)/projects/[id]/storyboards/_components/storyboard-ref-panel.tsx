"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  Info,
  Camera,
  Clock,
  Film,
  Hash,
  Image as ImageIcon,
  MessageSquare,
  Move3d,
  Music,
  Volume2,
  FileText,
  Sparkles,
  Loader2,
  Users,
  MapPin,
  Package,
  ExternalLink,
  Video,
  X,
  ZoomIn,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import { assetApi } from "@/lib/api/asset";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { SafeImage } from "@/components/ui/safe-image";

import type { Asset, AssetItem } from "@/lib/api/asset";
import type { Project } from "@/lib/api/project";
import type { StoryboardItem, Storyboard, StoryboardFrameType, StoryboardScene } from "@/lib/api/storyboard";
import { BatchGenDialog } from "./batch-gen-dialog";
import type { AssetItemWithInfo, SelectedAssetItem } from "./batch-gen-dialog";
import { VideoGenDialog } from "./video-gen-dialog";
import { FrameReferenceSection } from "./storyboard-frame-reference-dialog";
import { usePipelineStore } from "@/lib/store/pipeline-store";

// ========== 类型 ==========

interface SceneWithItems {
  scene: StoryboardScene;
  items: StoryboardItem[];
}

/** 带主资产名称和类型的子资产 */
interface AssetItemWithParent extends AssetItem {
  parentName: string;
  parentType: string;
}

/** 资产分组 */
interface GroupedAssets {
  characters: AssetItemWithParent[];
  scenes: AssetItemWithParent[];
  props: AssetItemWithParent[];
}

export interface BatchFrameGeneratePayload {
  episodeId: number;
  sceneId: number;
  firstItemIds: number[];
  lastItemIds: number[];
  overwriteExisting: boolean;
}

type BatchFrameGenerateHandler = (
  payload: BatchFrameGeneratePayload
) => Promise<void> | void;

// ========== 常量 ==========

const typeConfig = {
  character: {
    label: "角色",
    icon: Users,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
  },
  scene: {
    label: "场景",
    icon: MapPin,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
  },
  prop: {
    label: "道具",
    icon: Package,
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
  },
} as const;

/** 安全解析 ID 数组（兼容后端返回的 JSON 字符串或原生数组） */
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

const storyboardItemCollator = new Intl.Collator("zh-CN", {
  numeric: true,
  sensitivity: "base",
});

function compareStoryboardItemsAsc(a: StoryboardItem, b: StoryboardItem) {
  const aSortOrder = a.sortOrder ?? Number.MAX_SAFE_INTEGER;
  const bSortOrder = b.sortOrder ?? Number.MAX_SAFE_INTEGER;
  if (aSortOrder !== bSortOrder) return aSortOrder - bSortOrder;

  const aShotNumber = a.shotNumber || a.autoShotNumber || "";
  const bShotNumber = b.shotNumber || b.autoShotNumber || "";
  const shotCompare = storyboardItemCollator.compare(aShotNumber, bShotNumber);
  if (shotCompare !== 0) return shotCompare;

  return a.id - b.id;
}

function BatchFrameGenDialog({
  open,
  items,
  episodeId,
  sceneId,
  onClose,
  onConfirm,
}: {
  open: boolean;
  items: StoryboardItem[];
  episodeId: number;
  sceneId: number;
  onClose: () => void;
  onConfirm: BatchFrameGenerateHandler;
}) {
  const [overwriteExisting, setOverwriteExisting] = useState(false);
  const [includeFirstFrame, setIncludeFirstFrame] = useState(true);
  const [includeLastFrame, setIncludeLastFrame] = useState(true);
  const [selectedOverride, setSelectedOverride] = useState<Set<number> | null>(null);

  const hasFrameTypeSelected = includeFirstFrame || includeLastFrame;
  const sortedItems = [...items].sort(compareStoryboardItemsAsc);
  const selectableIds = new Set(sortedItems.map((item) => item.id));
  const defaultSelected = new Set(sortedItems.map((item) => item.id));
  const selected = selectedOverride
    ? new Set(Array.from(selectedOverride).filter((id) => selectableIds.has(id)))
    : defaultSelected;
  const selectedItems = sortedItems.filter((item) => selected.has(item.id));
  const firstItemIds = includeFirstFrame
    ? selectedItems
      .filter((item) => overwriteExisting || !item.firstFrameImageUrl)
      .map((item) => item.id)
    : [];
  const lastItemIds = includeLastFrame
    ? selectedItems
      .filter((item) => overwriteExisting || !item.lastFrameImageUrl)
      .map((item) => item.id)
    : [];
  const generateCount = firstItemIds.length + lastItemIds.length;
  const canSubmit = hasFrameTypeSelected && generateCount > 0;

  if (!open) return null;

  const resetSelection = () => {
    setSelectedOverride(null);
  };

  const toggleItem = (id: number) => {
    setSelectedOverride((prev) => {
      const next = new Set(prev ?? selected);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === sortedItems.length) {
      setSelectedOverride(new Set());
    } else {
      setSelectedOverride(new Set(sortedItems.map((item) => item.id)));
    }
  };

  const getPreviewUrl = (item: StoryboardItem) =>
    item.firstFrameImageUrl ||
    item.generatedImageUrl ||
    item.imageUrl ||
    item.referenceImageUrl ||
    item.lastFrameImageUrl ||
    null;

  const shotTypeLabels: Record<string, string> = {
    远景: "远景",
    全景: "全景",
    中景: "中景",
    近景: "近景",
    特写: "特写",
  };

  const handleConfirm = async () => {
    if (!canSubmit) return;
    await onConfirm({
      episodeId,
      sceneId,
      firstItemIds,
      lastItemIds,
      overwriteExisting,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative bg-card border border-border/30 rounded-xl shadow-2xl w-[520px] max-w-[92vw] max-h-[82vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border/20">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-primary" />
            </div>
            <div>
              <h3 className="text-sm font-semibold">批量生成首尾帧</h3>
              <p className="text-[10px] text-muted-foreground">
                选择当前场次中需要生成首尾帧的分镜镜头
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground transition-colors"
            title="关闭"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 border-b border-border/10">
          <div className="rounded-lg border border-border/30 bg-muted/10 px-3 py-3">
            <p className="text-xs font-medium mb-2">生成类型</p>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-2 rounded-lg bg-background/40 px-3 py-2 text-xs">
                <input
                  type="checkbox"
                  checked={includeFirstFrame}
                  onChange={(event) => {
                    setIncludeFirstFrame(event.target.checked);
                    resetSelection();
                  }}
                  className="h-4 w-4 rounded border-border accent-primary"
                />
                <span>首帧</span>
              </label>
              <label className="flex items-center gap-2 rounded-lg bg-background/40 px-3 py-2 text-xs">
                <input
                  type="checkbox"
                  checked={includeLastFrame}
                  onChange={(event) => {
                    setIncludeLastFrame(event.target.checked);
                    resetSelection();
                  }}
                  className="h-4 w-4 rounded border-border accent-primary"
                />
                <span>尾帧</span>
              </label>
            </div>
          </div>

          <label className="flex items-start gap-3 rounded-lg border border-border/30 bg-muted/10 px-3 py-3 text-left">
            <input
              type="checkbox"
              checked={overwriteExisting}
              onChange={(event) => {
                setOverwriteExisting(event.target.checked);
                resetSelection();
              }}
              className="mt-0.5 h-4 w-4 rounded border-border accent-primary"
            />
            <span className="min-w-0">
              <span className="block text-xs font-medium">覆盖已有首尾帧</span>
              <span className="mt-1 block text-[10px] leading-relaxed text-muted-foreground">
                默认关闭时只补齐所选镜头中缺失的首帧或尾帧；开启后会重新生成所选镜头的首帧和尾帧。
              </span>
            </span>
          </label>

          {!hasFrameTypeSelected ? (
            <div className="rounded-lg bg-amber-500/10 border border-amber-500/20 px-3 py-2.5 text-xs text-amber-600">
              请至少选择首帧或尾帧中的一种。
            </div>
          ) : canSubmit ? (
            <div className="rounded-lg bg-primary/5 border border-primary/15 px-3 py-2.5 text-xs text-primary">
              将提交 {firstItemIds.length} 个首帧和 {lastItemIds.length} 个尾帧生成任务。
            </div>
          ) : (
            <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-2.5 text-xs text-emerald-600">
              当前场次首尾帧已完整，无需生成。
            </div>
          )}
        </div>

        {sortedItems.length > 0 && (
          <div className="px-5 py-2.5 border-b border-border/10 flex items-center justify-between">
            <button
              type="button"
              onClick={toggleAll}
              className="text-xs text-primary hover:text-primary/80 font-medium transition-colors"
            >
              {selected.size === sortedItems.length ? "取消全选" : "全选"}
            </button>
            <span className="text-[10px] text-muted-foreground">
              已选 {selected.size} / {sortedItems.length} 个镜头
            </span>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-3 space-y-1.5 min-h-0">
          {sortedItems.length === 0 ? (
            <div className="text-center py-8">
              <Film className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground">
                当前场次暂无镜头
              </p>
            </div>
          ) : (
            sortedItems.map((item) => {
              const checked = selected.has(item.id);
              const imgUrl = getPreviewUrl(item);
              const willGenerateFirst =
                checked && includeFirstFrame && (overwriteExisting || !item.firstFrameImageUrl);
              const willGenerateLast =
                checked && includeLastFrame && (overwriteExisting || !item.lastFrameImageUrl);

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => toggleItem(item.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left",
                    checked
                      ? "bg-primary/8 ring-1 ring-primary/20"
                      : "hover:bg-muted/30"
                  )}
                >
                  <div
                    className={cn(
                      "h-5 w-5 rounded-md border-2 flex items-center justify-center shrink-0 transition-all",
                      checked
                        ? "bg-primary border-primary text-primary-foreground"
                        : "border-border/50"
                    )}
                  >
                    {checked && <Check className="h-3 w-3" />}
                  </div>

                  <div className="h-12 w-20 rounded-lg bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center">
                    {imgUrl ? (
                      <SafeImage
                        src={resolveMediaUrl(imgUrl) || undefined}
                        alt={`镜头 ${item.shotNumber || item.autoShotNumber || ""}`}
                        fallbackType="image"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <ImageIcon className="h-4 w-4 text-muted-foreground/30" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <p className="text-xs font-medium">
                        #{item.shotNumber || item.autoShotNumber || "?"}
                      </p>
                      {item.shotType && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground">
                          {shotTypeLabels[item.shotType] || item.shotType}
                        </span>
                      )}
                      {item.cameraMovement && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">
                          {item.cameraMovement}
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-muted-foreground truncate mt-0.5">
                      {item.content || "（无画面描述）"}
                    </p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {willGenerateFirst && (
                        <span className="text-[10px] text-cyan-400/70">
                          生成首帧
                        </span>
                      )}
                      {willGenerateLast && (
                        <span className="text-[10px] text-emerald-400/70">
                          生成尾帧
                        </span>
                      )}
                      {item.firstFrameImageUrl && (
                        <span className="text-[10px] text-muted-foreground/60">
                          已有首帧
                        </span>
                      )}
                      {item.lastFrameImageUrl && (
                        <span className="text-[10px] text-muted-foreground/60">
                          已有尾帧
                        </span>
                      )}
                    </div>
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="px-5 py-3.5 border-t border-border/20 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={!canSubmit}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all",
              "bg-primary text-primary-foreground hover:bg-primary/90",
              "disabled:opacity-40 disabled:pointer-events-none"
            )}
          >
            <Sparkles className="h-3.5 w-3.5" />
            开始生成 ({generateCount})
          </button>
        </div>
      </div>
    </div>
  );
}

function BatchFrameGenerateControl({
  items,
  loading,
  currentEpisodeId,
  currentSceneId,
  onConfirm,
}: {
  items: StoryboardItem[];
  loading?: boolean;
  currentEpisodeId?: number | null;
  currentSceneId?: number | null;
  onConfirm?: BatchFrameGenerateHandler;
}) {
  const [open, setOpen] = useState(false);
  const disabledReason =
    currentSceneId == null
      ? "请先选择场次"
      : currentEpisodeId == null
        ? "缺少分镜集上下文"
        : loading
          ? "正在加载当前场次镜头"
          : items.length === 0
            ? "当前场次暂无镜头"
            : !onConfirm
              ? "当前页面暂不支持批量生成"
              : null;
  const disabled = !!disabledReason;

  return (
    <div className="space-y-1.5">
      <button
        type="button"
        onClick={() => {
          if (!disabled) {
            setOpen(true);
          }
        }}
        disabled={disabled}
        title={disabledReason || "批量生成当前场次全部镜头的首尾帧"}
        className={cn(
          "w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium transition-all",
          disabled
            ? "bg-muted/30 text-muted-foreground cursor-not-allowed"
            : "bg-linear-to-r from-emerald-600 to-teal-600 text-white hover:shadow-lg hover:shadow-emerald-500/20 hover:scale-[1.02] active:scale-[0.98]"
        )}
      >
        <ImageIcon className="h-3.5 w-3.5" />
        批量生成首尾帧
      </button>
      {disabledReason && (
        <p className="text-[10px] text-muted-foreground/70 text-center">
          {disabledReason}
        </p>
      )}
      {onConfirm && currentEpisodeId != null && currentSceneId != null && (
        <BatchFrameGenDialog
          key={open ? "batch-frame-open" : "batch-frame-closed"}
          open={open}
          items={items}
          episodeId={currentEpisodeId}
          sceneId={currentSceneId}
          onClose={() => setOpen(false)}
          onConfirm={onConfirm}
        />
      )}
    </div>
  );
}

// ========== 场次资产面板 ==========

function SceneAssetPanel({
  sceneGroup,
  projectId,
  storyboard,
  onBatchGenerateFrames,
  onPreviewImage,
}: {
  sceneGroup: SceneWithItems;
  projectId: number;
  storyboard: Storyboard;
  onBatchGenerateFrames?: BatchFrameGenerateHandler;
  onPreviewImage?: (url: string, title: string) => void;
}) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [groupedAssets, setGroupedAssets] = useState<GroupedAssets>({
    characters: [],
    scenes: [],
    props: [],
  });
  const [showBatchGen, setShowBatchGen] = useState(false);
  const [showVideoGen, setShowVideoGen] = useState(false);

  // 直接从分镜 items 聚合子资产 ID（characterIds / sceneAssetItemId / propIds）
  // 批量查子资产详情，附带主资产名称做辅助展示
  const loadAssets = useCallback(async () => {
    setLoading(true);
    try {
      const characterItemIds = new Set<number>();
      const sceneItemIds = new Set<number>();
      const propItemIds = new Set<number>();

      for (const item of sceneGroup.items) {
        const charIds = parseIds(item.characterIds);
        charIds.forEach((id) => characterItemIds.add(id));
        if (item.sceneAssetItemId) sceneItemIds.add(item.sceneAssetItemId);
        const pIds = parseIds(item.propIds);
        pIds.forEach((id) => propItemIds.add(id));
      }

      const allItemIds = [
        ...characterItemIds,
        ...sceneItemIds,
        ...propItemIds,
      ];

      if (allItemIds.length === 0) {
        setGroupedAssets({ characters: [], scenes: [], props: [] });
        return;
      }

      // 批量获取子资产详情
      const results = await Promise.all(
        allItemIds.map((id) => assetApi.getItem(id).catch((error) => {
          toastApiError(error, "加载资产项失败");
          return null;
        }))
      );

      // 收集主资产ID用于查名称
      const parentAssetIds = new Set<number>();
      const itemMap = new Map<number, AssetItem>();
      for (const r of results) {
        if (!r) continue;
        itemMap.set(r.id, r);
        if (r.assetId) parentAssetIds.add(r.assetId);
      }

      // 批量获取主资产（用于获取名称和类型做辅助标注）
      const parentAssets = await Promise.all(
        Array.from(parentAssetIds).map((id) => assetApi.get(id).catch((error) => {
          toastApiError(error, "加载父级资产失败");
          return null;
        }))
      );
      const parentInfoMap = new Map<number, { name: string; type: string }>();
      for (const a of parentAssets) {
        if (a) parentInfoMap.set(a.id, { name: a.name, type: a.type });
      }

      // 直接展示子资产，附带主资产名称和类型
      const toItemsWithParent = (ids: Set<number>): AssetItemWithParent[] => {
        const result: AssetItemWithParent[] = [];
        for (const itemId of ids) {
          const item = itemMap.get(itemId);
          if (!item) continue;
          const info = parentInfoMap.get(item.assetId);
          result.push({
            ...item,
            parentName: info?.name || "未知资产",
            parentType: info?.type || "unknown",
          });
        }
        return result;
      };

      setGroupedAssets({
        characters: toItemsWithParent(characterItemIds),
        scenes: toItemsWithParent(sceneItemIds),
        props: toItemsWithParent(propItemIds),
      });
    } catch (err) {
      console.error("加载场次资产失败:", err);
      toastApiError(err, "加载场次资产失败");
    } finally {
      setLoading(false);
    }
  }, [sceneGroup]);

  useEffect(() => {
    loadAssets();
  }, [loadAssets]);

  const allItems = [
    ...groupedAssets.characters,
    ...groupedAssets.scenes,
    ...groupedAssets.props,
  ];
  const hasAssets = allItems.length > 0;

  // 点击子资产时跳转到其主资产
  const handleItemClick = (item: AssetItemWithParent) => {
    router.push(`/projects/${projectId}/assets?highlight=${item.assetId}`);
  };

  const addPipeline = usePipelineStore((s) => s.addPipeline);
  const setNotificationOpen = usePipelineStore((s) => s.setNotificationOpen);

  // 构建传给 BatchGenDialog 的子资产列表
  const batchGenItems: AssetItemWithInfo[] = allItems.map((ai) => ({
    item: ai,
    parentName: ai.parentName,
    parentType: ai.parentType,
    assetId: ai.assetId,
  }));

  const handleBatchGenConfirm = (selectedItems: SelectedAssetItem[]) => {
    // 提取去重的主资产ID和选中的子资产ID
    const selectedAssetIds = [...new Set(selectedItems.map((s) => s.assetId))];
    const selectedAssetItemIds = selectedItems.map((s) => s.itemId);

    // 触发 Agent Pipeline
    addPipeline({
      label: `批量生图 (${selectedItems.length} 个子资产)`,
      projectId,
      request: {
        agentType: "asset_image_gen",
        toolExecutionMode: "FULL_ACCESS",
        projectId,
        context: {
          selectedAssetIds,
          selectedAssetItemIds,
        },
      },
      onComplete: () => {
        loadAssets();
      },
    });

    // 打开通知面板让用户看到进度
    setNotificationOpen(true);
  };

  /** 批量生视频确认 */
  const handleVideoGenConfirm = (selectedItemIds: number[], promptOnly?: boolean) => {
    addPipeline({
      label: promptOnly
        ? `批量生成视频提示词 (${selectedItemIds.length} 个镜头)`
        : `批量生视频 (${selectedItemIds.length} 个镜头)`,
      projectId,
      request: {
        agentType: "storyboard_video_gen",
        toolExecutionMode: "FULL_ACCESS",
        projectId,
        context: {
          selectedStoryboardItemIds: selectedItemIds,
          storyboardId: storyboard.id,
          promptOnly: promptOnly || false,
        },
      },
      onComplete: () => {
        // 视频生成完成后可能需要刷新分镜数据
      },
    });
    setNotificationOpen(true);
  };

  return (
    <div className="p-4 space-y-4">
      {/* 标题 */}
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1.5">
          <Camera className="h-3 w-3" /> 场次详情
        </h4>
        <p className="text-sm font-semibold">
          {sceneGroup.scene.sceneHeading ||
            `场次 ${sceneGroup.scene.sceneNumber || sceneGroup.scene.id}`}
        </p>
        {sceneGroup.scene.location && (
          <p className="text-[10px] text-muted-foreground mt-0.5">
            {sceneGroup.scene.intExt && `${sceneGroup.scene.intExt} `}
            {sceneGroup.scene.location}
            {sceneGroup.scene.timeOfDay && ` · ${sceneGroup.scene.timeOfDay}`}
          </p>
        )}
        <p className="text-[10px] text-muted-foreground/60 mt-0.5">
          {sceneGroup.items.length} 个镜头
        </p>
      </div>

      {/* 批量生图按钮 */}
      {hasAssets && (
        <button
          onClick={() => setShowBatchGen(true)}
          className={cn(
            "w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium transition-all",
            "bg-linear-to-r from-cyan-600 to-blue-600 text-white",
            "hover:shadow-lg hover:shadow-cyan-500/20 hover:scale-[1.02]",
            "active:scale-[0.98]"
          )}
        >
          <Sparkles className="h-3.5 w-3.5" />
          批量生图
        </button>
      )}

      <BatchFrameGenerateControl
        items={sceneGroup.items}
        currentEpisodeId={sceneGroup.scene.episodeId}
        currentSceneId={sceneGroup.scene.id}
        onConfirm={onBatchGenerateFrames}
      />

      {/* 批量生视频按钮 */}
      <button
        onClick={() => setShowVideoGen(true)}
        className={cn(
          "w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium transition-all",
          "bg-linear-to-r from-purple-600 to-pink-600 text-white",
          "hover:shadow-lg hover:shadow-purple-500/20 hover:scale-[1.02]",
          "active:scale-[0.98]"
        )}
      >
        <Video className="h-3.5 w-3.5" />
        批量生视频
      </button>

      {/* 加载中 */}
      {loading && (
        <div className="flex items-center justify-center py-6">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
      )}

      {/* 资产列表 */}
      {!loading && (
        <>
          {!hasAssets && (
            <div className="text-center py-6 border-t border-border/20 pt-4">
              <Package className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground">
                该场次暂无关联资产
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                请先在剧本场次中设置角色、场景和道具
              </p>
            </div>
          )}

          {/* 按类型分组展示 */}
          {(
            [
              ["character", groupedAssets.characters],
              ["scene", groupedAssets.scenes],
              ["prop", groupedAssets.props],
            ] as [keyof typeof typeConfig, AssetItemWithParent[]][]
          ).map(
            ([type, items]) =>
              items.length > 0 && (
                <AssetItemGroup
                  key={type}
                  type={type}
                  items={items}
                  onItemClick={handleItemClick}
                  onPreviewImage={onPreviewImage}
                />
              )
          )}
        </>
      )}

      {/* 批量生图弹窗 — 传入子资产列表 */}
      <BatchGenDialog
        key={showBatchGen ? "batch-gen-open" : "batch-gen-closed"}
        open={showBatchGen}
        onClose={() => setShowBatchGen(false)}
        assetItems={batchGenItems}
        onConfirm={handleBatchGenConfirm}
      />

      {/* 批量生视频弹窗 */}
      <VideoGenDialog
        key={showVideoGen ? "video-gen-open" : "video-gen-closed"}
        open={showVideoGen}
        onClose={() => setShowVideoGen(false)}
        items={sceneGroup.items}
        onConfirm={handleVideoGenConfirm}
      />
    </div>
  );
}

/** 子资产分组展示 */
function AssetItemGroup({
  type,
  items,
  onItemClick,
  onPreviewImage,
}: {
  type: keyof typeof typeConfig;
  items: AssetItemWithParent[];
  onItemClick: (item: AssetItemWithParent) => void;
  onPreviewImage?: (url: string, title: string) => void;
}) {
  const config = typeConfig[type];
  const Icon = config.icon;

  return (
    <div className="border-t border-border/20 pt-3">
      <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2.5 flex items-center gap-1.5">
        <Icon className={cn("h-3 w-3", config.color)} />
        {config.label}
        <span className="text-[10px] font-normal text-muted-foreground/60 ml-auto">
          {items.length}
        </span>
      </h4>
      <div className="space-y-1.5">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onItemClick(item)}
            className={cn(
              "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition-all text-left group",
              "hover:bg-muted/30"
            )}
          >
            {/* 缩略图 */}
            <div
              onClick={(e) => {
                if (item.imageUrl && onPreviewImage) {
                  e.stopPropagation();
                  onPreviewImage(item.imageUrl, `${item.parentName}: ${item.name || "初始设定"}`);
                }
              }}
              className={cn(
                "h-10 w-10 bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center relative group/img",
                type === "character" ? "rounded-full" : "rounded-lg",
                item.imageUrl && "cursor-zoom-in hover:border-primary/40 transition-colors"
              )}
            >
              {item.imageUrl ? (
                <>
                  <SafeImage
                    src={resolveMediaUrl(item.imageUrl)}
                    alt={item.name || item.parentName}
                    fallbackType={type === "character" ? "avatar" : type === "scene" ? "scene" : "prop"}
                    className="w-full h-full object-cover transition-transform group-hover/img:scale-105"
                  />
                  <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/25 flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-all">
                    <ZoomIn className="h-3.5 w-3.5 text-white/90" />
                  </div>
                </>
              ) : (
                <Icon
                  className={cn("h-4 w-4 text-muted-foreground/30")}
                />
              )}
            </div>
            {/* 信息 */}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">
                {item.name || item.parentName}
              </p>
              <p className="text-[10px] text-muted-foreground/60 truncate mt-0.5">
                {item.parentName}
              </p>
            </div>
            {/* 跳转图标 */}
            <ExternalLink className="h-3 w-3 text-muted-foreground/30 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}

// ========== 镜头详情（保留原有） ==========

function ItemDetail({
  item,
  projectId,
  project,
  assetLookup,
  onEditAssets,
  onUpdateFrame,
  onGenerateFrame,
  onPreviewImage,
}: {
  item: StoryboardItem;
  projectId: number;
  project?: Project | null;
  assetLookup?: Record<number, { item: AssetItem; asset: Asset }>;
  onEditAssets?: () => void;
  onUpdateFrame?: (itemId: number, frameType: StoryboardFrameType, imageUrl: string | null) => Promise<void> | void;
  onGenerateFrame?: (item: StoryboardItem, frameType: StoryboardFrameType, prompt: string) => Promise<void> | void;
  onPreviewImage?: (url: string, title: string) => void;
}) {
  const router = useRouter();
  const detailRows: {
    icon: typeof Info;
    label: string;
    value: string | null;
  }[] = [
    { icon: Hash, label: "镜号", value: item.shotNumber },
    { icon: Camera, label: "景别", value: item.shotType },
    {
      icon: Clock,
      label: "时长",
      value: item.duration ? `${item.duration}s` : null,
    },
    { icon: Move3d, label: "镜头运动", value: item.cameraMovement },
    { icon: Camera, label: "机位角度", value: item.cameraAngle },
    { icon: Camera, label: "焦距", value: item.focalLength },
    { icon: Film, label: "转场", value: item.transition },
  ];

  // ===== 加载镜头关联资产 =====
  const [linkedAssets, setLinkedAssets] = useState<{
    characters: (Asset & { items: AssetItem[] })[];
    scenes: (Asset & { items: AssetItem[] })[];
    props: (Asset & { items: AssetItem[] })[];
  }>({ characters: [], scenes: [], props: [] });
  const [assetsLoading, setAssetsLoading] = useState(false);

  const loadLinkedAssets = useCallback(async () => {
    // 解析各类子资产 ID
    const charItemIds = parseIds(item.characterIds);
    const sceneItemId = item.sceneAssetItemId && item.sceneAssetItemId > 0 ? item.sceneAssetItemId : null;
    const propItemIds = parseIds(item.propIds);

    const allItemIds = [...charItemIds, ...propItemIds];
    if (sceneItemId) allItemIds.push(sceneItemId);

    if (allItemIds.length === 0) {
      setLinkedAssets({ characters: [], scenes: [], props: [] });
      return;
    }

    setAssetsLoading(true);
    try {
      // 批量获取子资产详情
      const items = await Promise.all(
        allItemIds.map((id) => assetApi.getItem(id).catch((error) => {
          toastApiError(error, "加载镜头关联资产项失败");
          return null;
        }))
      );

      // 收集主资产 ID（去重）
      const parentIds = new Set<number>();
      const itemMap = new Map<number, AssetItem>();
      for (const r of items) {
        if (!r) continue;
        itemMap.set(r.id, r);
        if (r.assetId) parentIds.add(r.assetId);
      }

      // 批量获取主资产 + 其子资产列表
      const parentResults = await Promise.all(
        Array.from(parentIds).map(async (id) => {
          try {
            const [asset, subItems] = await Promise.all([
              assetApi.get(id),
              assetApi.listItems(id),
            ]);
            return { ...asset, items: subItems || [] };
          } catch {
            return null;
          }
        })
      );

      const valid = parentResults.filter(
        (r): r is Asset & { items: AssetItem[] } => r !== null
      );

      // 按子资产ID → 主资产分类
      const charParentIds = new Set(charItemIds.map((id) => itemMap.get(id)?.assetId).filter((id): id is number => id != null));
      const sceneParentId = sceneItemId ? itemMap.get(sceneItemId)?.assetId : null;
      const propParentIds = new Set(propItemIds.map((id) => itemMap.get(id)?.assetId).filter((id): id is number => id != null));

      setLinkedAssets({
        characters: valid.filter((a) => charParentIds.has(a.id)),
        scenes: valid.filter((a) => a.id === sceneParentId),
        props: valid.filter((a) => propParentIds.has(a.id)),
      });
    } catch (err) {
      console.error("加载镜头关联资产失败:", err);
      toastApiError(err, "加载镜头关联资产失败");
    } finally {
      setAssetsLoading(false);
    }
  }, [item.characterIds, item.sceneAssetItemId, item.propIds]);

  useEffect(() => {
    if (assetLookup && Object.keys(assetLookup).length > 0) {
      const charItemIds = parseIds(item.characterIds);
      const sceneItemId = item.sceneAssetItemId && item.sceneAssetItemId > 0 ? item.sceneAssetItemId : null;
      const propItemIds = parseIds(item.propIds);

      const charsMap = new Map<number, Asset & { items: AssetItem[] }>();
      charItemIds.forEach(id => {
        const entry = assetLookup[id];
        if (entry) {
          const { item: subItem, asset } = entry;
          if (!charsMap.has(asset.id)) {
            charsMap.set(asset.id, { ...asset, items: [] });
          }
          if (!charsMap.get(asset.id)!.items.some(x => x.id === subItem.id)) {
            charsMap.get(asset.id)!.items.push(subItem);
          }
        }
      });

      const scenesList: (Asset & { items: AssetItem[] })[] = [];
      if (sceneItemId) {
        const entry = assetLookup[sceneItemId];
        if (entry) {
          const { item: subItem, asset } = entry;
          scenesList.push({ ...asset, items: [subItem] });
        }
      }

      const propsMap = new Map<number, Asset & { items: AssetItem[] }>();
      propItemIds.forEach(id => {
        const entry = assetLookup[id];
        if (entry) {
          const { item: subItem, asset } = entry;
          if (!propsMap.has(asset.id)) {
            propsMap.set(asset.id, { ...asset, items: [] });
          }
          if (!propsMap.get(asset.id)!.items.some(x => x.id === subItem.id)) {
            propsMap.get(asset.id)!.items.push(subItem);
          }
        }
      });

      setLinkedAssets({
        characters: Array.from(charsMap.values()),
        scenes: scenesList,
        props: Array.from(propsMap.values()),
      });
      setAssetsLoading(false);
    } else {
      loadLinkedAssets();
    }
  }, [item.characterIds, item.sceneAssetItemId, item.propIds, assetLookup, loadLinkedAssets]);

  const hasLinkedAssets =
    linkedAssets.characters.length > 0 ||
    linkedAssets.scenes.length > 0 ||
    linkedAssets.props.length > 0;

  return (
    <div className="p-4 space-y-5">
      {/* 标题 */}
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Info className="h-3 w-3" /> 镜头详情
        </h4>
      </div>

      <FrameReferenceSection
        item={item}
        project={project}
        frameType="first"
        imageUrl={item.firstFrameImageUrl}
        prompt={item.firstFramePrompt}
        onUpdateFrame={onUpdateFrame}
        onGenerateFrame={onGenerateFrame}
        onPreviewImage={onPreviewImage}
      />

      <FrameReferenceSection
        item={item}
        project={project}
        frameType="last"
        imageUrl={item.lastFrameImageUrl}
        prompt={item.lastFramePrompt}
        onUpdateFrame={onUpdateFrame}
        onGenerateFrame={onGenerateFrame}
        onPreviewImage={onPreviewImage}
      />

      {/* 预览图 */}
      {(item.imageUrl ||
        item.referenceImageUrl ||
        item.generatedImageUrl) && (
        <div
          onClick={() => {
            const rawUrl = item.generatedImageUrl || item.imageUrl || item.referenceImageUrl;
            if (rawUrl && onPreviewImage) {
              onPreviewImage(rawUrl, `镜头 #${item.shotNumber || item.autoShotNumber || ""} 画面`);
            }
          }}
          className={cn(
            "rounded-lg overflow-hidden border border-border/20 relative group/preview cursor-zoom-in hover:border-primary/40 transition-colors"
          )}
        >
          <SafeImage
            src={
              resolveMediaUrl(item.generatedImageUrl ||
                item.imageUrl ||
                item.referenceImageUrl)
            }
            alt="镜头画面"
            fallbackType="image"
            className="w-full aspect-video object-cover transition-transform group-hover/preview:scale-102"
          />
          <div className="absolute inset-0 bg-black/0 group-hover/preview:bg-black/25 flex items-center justify-center opacity-0 group-hover/preview:opacity-100 transition-all">
            <ZoomIn className="h-5 w-5 text-white/90" />
          </div>
        </div>
      )}

      {/* 基础属性 */}
      <div className="space-y-2">
        {detailRows.map(
          ({ icon: AttrIcon, label, value }) =>
            value && (
              <div key={label} className="flex items-center gap-2 text-xs">
                <AttrIcon className="h-3 w-3 text-muted-foreground shrink-0" />
                <span className="text-muted-foreground">{label}</span>
                <span className="font-medium ml-auto truncate max-w-[140px]">
                  {value}
                </span>
              </div>
            )
        )}
      </div>

      {/* 画面内容 */}
      {item.content && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <ImageIcon className="h-3 w-3" /> 画面内容
          </h4>
          <p className="text-xs text-foreground/80 leading-relaxed">
            {item.content}
          </p>
        </div>
      )}

      {/* 场景预期 */}
      {item.sceneExpectation && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            场景预期
          </h4>
          <p className="text-xs text-muted-foreground leading-relaxed italic">
            {item.sceneExpectation}
          </p>
        </div>
      )}

      {/* 对白 / 旁白 */}
      {item.dialogue && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <MessageSquare className="h-3 w-3" /> 对白 / 旁白
          </h4>
          <p className="text-xs text-foreground/80 leading-relaxed italic">
            「{item.dialogue}」
          </p>
        </div>
      )}

      {/* 音效 & 音乐 */}
      {(item.soundEffect || item.music || item.sound) && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Volume2 className="h-3 w-3" /> 声音
          </h4>
          <div className="space-y-1.5">
            {item.sound && (
              <div className="flex items-center gap-1.5 text-xs">
                <span className="text-muted-foreground">声音:</span>
                <span>{item.sound}</span>
              </div>
            )}
            {item.soundEffect && (
              <div className="flex items-center gap-1.5 text-xs">
                <span className="text-muted-foreground">音效:</span>
                <span>{item.soundEffect}</span>
              </div>
            )}
            {item.music && (
              <div className="flex items-center gap-1.5 text-xs">
                <Music className="h-3 w-3 text-muted-foreground shrink-0" />
                <span>{item.music}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 参考图 */}
      {item.referenceImageUrl && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <ImageIcon className="h-3 w-3" /> 参考图
          </h4>
          <div className="rounded-lg overflow-hidden border border-border/20">
            <SafeImage
              src={resolveMediaUrl(item.referenceImageUrl)}
              alt="参考图"
              fallbackType="image"
              className="w-full aspect-video object-cover"
            />
          </div>
        </div>
      )}

      {/* 备注 */}
      {item.remark && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <FileText className="h-3 w-3" /> 备注
          </h4>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {item.remark}
          </p>
        </div>
      )}

      {/* ===== 关联资产 ===== */}
      {assetsLoading && (
        <div className="border-t border-border/20 pt-4 flex items-center justify-center py-4">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
      )}

      {!assetsLoading && (
        <div className="border-t border-border/20 pt-4 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
              <Users className="h-3 w-3" /> 关联资产
            </h4>
            {onEditAssets && (
              <button
                onClick={onEditAssets}
                className="text-[10px] text-primary hover:underline font-medium"
              >
                编辑关联
              </button>
            )}
          </div>

          {!hasLinkedAssets ? (
            <p className="text-[11px] text-muted-foreground/50 italic pl-1">
              暂无镜头关联资产，请点击右上角编辑关联
            </p>
          ) : (
            (
              [
                ["character", linkedAssets.characters],
                ["scene", linkedAssets.scenes],
                ["prop", linkedAssets.props],
              ] as [keyof typeof typeConfig, (Asset & { items: AssetItem[] })[]][]
            ).map(
              ([type, assets]) =>
                assets.length > 0 && (
                  <LinkedAssetGroup
                    key={type}
                    type={type}
                    assets={assets}
                    onAssetClick={(id) =>
                      router.push(
                        `/projects/${projectId}/assets?highlight=${id}`
                      )
                    }
                    onPreviewImage={onPreviewImage}
                  />
                )
            )
          )}
        </div>
      )}
    </div>
  );
}

/** 关联资产分组展示（和表格页一致的上下布局，左右排列，无边框） */
function LinkedAssetGroup({
  type,
  assets,
  onAssetClick,
  onPreviewImage,
}: {
  type: keyof typeof typeConfig;
  assets: (Asset & { items: AssetItem[] })[];
  onAssetClick: (id: number) => void;
  onPreviewImage?: (url: string, title: string) => void;
}) {
  const config = typeConfig[type];
  const Icon = config.icon;

  // 将主资产与其关联的子资产打平，展示为独立的项，和表格页完全一致
  const itemsToShow = assets.flatMap((asset) => {
    if (asset.items.length === 0) {
      return [{ asset, sub: null as AssetItem | null }];
    }
    return asset.items.map((sub) => ({ asset, sub: sub as AssetItem | null }));
  });

  return (
    <div className="border-t border-border/20 pt-3">
      <h5 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
        <Icon className={cn("h-3 w-3", config.color)} />
        {config.label}
        <span className="text-[10px] font-normal text-muted-foreground/60 ml-auto">
          {itemsToShow.length}
        </span>
      </h5>
      
      <div className="flex flex-row flex-wrap gap-x-4 gap-y-3.5 px-1 py-1">
        {itemsToShow.map(({ asset, sub }, index) => {
          const displayName = !sub || !sub.name || sub.name === "初始设定" || sub.name === asset.name
            ? asset.name
            : sub.name;
          
          const imageUrl = sub?.imageUrl || asset.coverUrl;
          const letter = displayName.charAt(0);
          
          // 根据类型定制颜色
          const themeClass = type === "character"
            ? "bg-blue-500/10 text-blue-500 border-blue-500/20 text-blue-500 dark:text-blue-400 group-hover/asset:text-blue-600"
            : type === "scene"
            ? "bg-green-500/10 text-green-500 border-green-500/20 text-green-500 dark:text-green-400 group-hover/asset:text-green-600"
            : "bg-amber-500/10 text-amber-500 border-amber-500/20 text-amber-500 dark:text-amber-400 group-hover/asset:text-amber-600";

          return (
            <Tooltip key={`${asset.id}-${sub?.id || index}`}>
              <TooltipTrigger
                render={
                  <div
                    onClick={() => onAssetClick(asset.id)}
                    className="flex flex-col items-center group/asset cursor-pointer select-none shrink-0"
                  >
                    <div className="relative">
                      {imageUrl ? (
                        <div
                          onClick={(e) => {
                            if (onPreviewImage) {
                              e.stopPropagation();
                              onPreviewImage(imageUrl, `${config.label}: ${displayName}`);
                            }
                          }}
                          className={cn(
                            "h-11 w-11 bg-muted/30 border border-border/10 overflow-hidden shrink-0 flex items-center justify-center relative group/coverimg cursor-zoom-in",
                            type === "character" ? "rounded-full" : "rounded-lg"
                          )}
                        >
                          <SafeImage
                            src={resolveMediaUrl(imageUrl)}
                            alt={displayName}
                            fallbackType={type === "character" ? "avatar" : type === "scene" ? "scene" : "prop"}
                            className="w-full h-full object-cover transition-transform duration-200 group-hover/asset:scale-110"
                          />
                          <div className="absolute inset-0 bg-black/0 group-hover/coverimg:bg-black/25 flex items-center justify-center opacity-0 group-hover/coverimg:opacity-100 transition-all">
                            <ZoomIn className="h-3 w-3 text-white/90" />
                          </div>
                        </div>
                      ) : (
                        <div
                          className={cn(
                            "h-11 w-11 border flex items-center justify-center font-semibold text-xs shrink-0 transition-colors duration-200",
                            type === "character" ? "rounded-full" : "rounded-lg",
                            themeClass.split(" ").slice(0, 3).join(" ") // 只提取 bg, text, border 相关类
                          )}
                        >
                          {letter}
                        </div>
                      )}
                    </div>
                    <span
                      className={cn(
                        "text-[10px] font-semibold truncate max-w-[56px] leading-tight text-center mt-1.5 transition-colors duration-200",
                        themeClass.split(" ").slice(3).join(" ") // 提取文字颜色类
                      )}
                    >
                      {displayName}
                    </span>
                  </div>
                }
              />
              <TooltipContent className="max-w-xs flex flex-col gap-1 text-left items-start px-3.5 py-2.5 rounded-xl text-xs leading-normal bg-white/85 dark:bg-zinc-900/85 backdrop-blur-md border border-zinc-200/50 dark:border-zinc-800/50 text-zinc-900 dark:text-zinc-50 shadow-lg [&_.bg-foreground]:bg-white/85 [&_.fill-foreground]:fill-white/85 dark:[&_.bg-foreground]:bg-zinc-900/85 dark:[&_.fill-foreground]:fill-zinc-900/85">
                <span className="font-semibold text-xs">{config.label}: {displayName}</span>
                {asset.description && (
                  <span className="text-[10px] opacity-80 leading-normal max-w-[200px] break-words mt-0.5">
                    {asset.description}
                  </span>
                )}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
}

function StoryboardOverview({
  storyboard,
  items,
  projectName,
  onBatchGenerateFrames,
}: {
  storyboard: Storyboard;
  items: StoryboardItem[];
  projectName?: string;
  onBatchGenerateFrames?: BatchFrameGenerateHandler;
}) {
  const totalDuration = items.reduce(
    (sum, item) => sum + (item.duration || 0),
    0
  );
  const withImage = items.filter(
    (i) => i.firstFrameImageUrl || i.lastFrameImageUrl || i.imageUrl || i.generatedImageUrl || i.referenceImageUrl
  ).length;

  return (
    <div className="p-4 space-y-5">
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Info className="h-3 w-3" /> 分镜概览
        </h4>
        <p className="text-sm font-semibold mb-1">
          {projectName || "未命名项目"}
        </p>
        {storyboard.description && (
          <p className="text-xs text-muted-foreground leading-relaxed">
            {storyboard.description}
          </p>
        )}
      </div>

      <BatchFrameGenerateControl
        items={[]}
        currentEpisodeId={null}
        currentSceneId={null}
        onConfirm={onBatchGenerateFrames}
      />

      {/* 统计 */}
      <div className="border-t border-border/20 pt-4">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          统计
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-primary">{items.length}</p>
            <p className="text-[10px] text-muted-foreground">总镜头数</p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-cyan-400">
              {totalDuration > 0 ? `${totalDuration}s` : "—"}
            </p>
            <p className="text-[10px] text-muted-foreground">总时长</p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-amber-400">{withImage}</p>
            <p className="text-[10px] text-muted-foreground">有画面</p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-violet-400">
              {items.length - withImage}
            </p>
            <p className="text-[10px] text-muted-foreground">无画面</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ========== 主面板 ==========

export function StoryboardRefPanel({
  storyboard,
  items,
  selectedItem,
  activeSceneGroup,
  projectId,
  project,
  assetLookup,
  onUpdateFrame,
  onGenerateFrame,
  onBatchGenerateFrames,
  onEditAssets,
}: {
  storyboard: Storyboard;
  items: StoryboardItem[];
  selectedItem: StoryboardItem | null;
  activeSceneGroup?: SceneWithItems | null;
  projectId: number;
  project?: Project | null;
  assetLookup?: Record<number, { item: AssetItem; asset: Asset }>;
  onUpdateFrame?: (itemId: number, frameType: StoryboardFrameType, imageUrl: string | null) => Promise<void> | void;
  onGenerateFrame?: (item: StoryboardItem, frameType: StoryboardFrameType, prompt: string) => Promise<void> | void;
  onBatchGenerateFrames?: BatchFrameGenerateHandler;
  onEditAssets?: (item: StoryboardItem) => void;
}) {
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [previewImageTitle, setPreviewImageTitle] = useState<string>("");

  const handlePreviewImage = useCallback((url: string, title: string) => {
    setPreviewImageUrl(url);
    setPreviewImageTitle(title);
  }, []);

  const showShot = selectedItem;

  return (
    <div className="w-full lg:w-72 border-l border-border/20 flex flex-col shrink-0 bg-card/20 overflow-y-auto h-full relative">
      {showShot ? (
        <>
          <ItemDetail
            item={selectedItem}
            projectId={projectId}
            project={project}
            assetLookup={assetLookup}
            onUpdateFrame={onUpdateFrame}
            onGenerateFrame={onGenerateFrame}
            onEditAssets={() => onEditAssets?.(selectedItem)}
            onPreviewImage={handlePreviewImage}
          />
          {activeSceneGroup && (
            <>
              <div className="mx-4 border-t border-border/30" />
              <SceneAssetPanel
                sceneGroup={activeSceneGroup}
                projectId={projectId}
                storyboard={storyboard}
                onBatchGenerateFrames={onBatchGenerateFrames}
                onPreviewImage={handlePreviewImage}
              />
            </>
          )}
        </>
      ) : activeSceneGroup ? (
        <SceneAssetPanel
          sceneGroup={activeSceneGroup}
          projectId={projectId}
          storyboard={storyboard}
          onBatchGenerateFrames={onBatchGenerateFrames}
          onPreviewImage={handlePreviewImage}
        />
      ) : (
        <StoryboardOverview
          storyboard={storyboard}
          items={items}
          projectName={project?.name}
          onBatchGenerateFrames={onBatchGenerateFrames}
        />
      )}

      {/* 图片大图预览灯箱 */}
      {previewImageUrl && (
        <div 
          className="modal-overlay fixed inset-0 z-[9999] flex items-center justify-center p-4 animate-in fade-in duration-200"
          onClick={() => setPreviewImageUrl(null)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-3" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setPreviewImageUrl(null)}
              className="absolute -top-12 right-0 rounded-full border border-border/40 bg-background/80 p-1.5 text-foreground shadow-xl backdrop-blur-xl transition-colors hover:bg-background"
              type="button"
            >
              <X className="h-5 w-5" />
            </button>
            <SafeImage
              src={resolveMediaUrl(previewImageUrl)}
              alt={previewImageTitle}
              fallbackType="image"
              className="max-w-full max-h-[80vh] rounded-lg object-contain shadow-2xl border border-border/40 select-none pointer-events-none"
            />
            <p className="rounded-full border border-border/40 bg-background/80 px-3 py-1.5 text-xs font-medium text-foreground shadow-xl backdrop-blur-xl">
              {previewImageTitle}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
