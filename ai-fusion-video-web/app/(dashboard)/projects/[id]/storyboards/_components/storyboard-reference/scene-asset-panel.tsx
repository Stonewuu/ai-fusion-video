"use client";

import type { AssetItem } from "@/lib/api/asset";
import { assetApi } from "@/lib/api/asset";
import type { Storyboard } from "@/lib/api/storyboard";
import { toastApiError } from "@/lib/api/toast-api-error";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";
import { Camera, Loader2, Package, Sparkles, Video } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import type { AssetItemWithInfo, SelectedAssetItem } from "../batch-gen-dialog";
import { BatchGenDialog } from "../batch-gen-dialog";
import { VideoGenDialog } from "../video-gen-dialog";
import { AssetItemGroup } from "./asset-item-group";
import { BatchFrameGenerateControl } from "./batch-frame-generate";
import {
  AssetItemWithParent,
  BatchFrameGenerateHandler,
  GroupedAssets,
  parseIds,
  SceneWithItems,
  typeConfig,
} from "./reference-shared";

export // ========== 场次资产面板 ==========

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

      const allItemIds = [...characterItemIds, ...sceneItemIds, ...propItemIds];

      if (allItemIds.length === 0) {
        setGroupedAssets({ characters: [], scenes: [], props: [] });
        return;
      }

      // 批量获取子资产详情
      const results = await Promise.all(
        allItemIds.map((id) =>
          assetApi.getItem(id).catch((error) => {
            toastApiError(error, "加载资产项失败");
            return null;
          }),
        ),
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
        Array.from(parentAssetIds).map((id) =>
          assetApi.get(id).catch((error) => {
            toastApiError(error, "加载父级资产失败");
            return null;
          }),
        ),
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
  const handleVideoGenConfirm = (
    selectedItemIds: number[],
    promptOnly?: boolean,
  ) => {
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
            "active:scale-[0.98]",
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
          "active:scale-[0.98]",
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
              ),
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
