"use client";

import type { AssetItem } from "@/lib/api/asset";
import type { StoryboardItem, StoryboardScene } from "@/lib/api/storyboard";
import { MapPin, Package, Users } from "lucide-react";

export // ========== 类型 ==========

interface SceneWithItems {
  scene: StoryboardScene;
  items: StoryboardItem[];
}

export /** 带主资产名称和类型的子资产 */
interface AssetItemWithParent extends AssetItem {
  parentName: string;
  parentType: string;
}

export /** 资产分组 */
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

export type BatchFrameGenerateHandler = (
  payload: BatchFrameGeneratePayload,
) => Promise<void> | void;

export // ========== 常量 ==========

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

export /** 安全解析 ID 数组（兼容后端返回的 JSON 字符串或原生数组） */
function parseIds(raw: number[] | null | undefined): number[] {
  return raw ?? [];
}

export const storyboardItemCollator = new Intl.Collator("zh-CN", {
  numeric: true,
  sensitivity: "base",
});

export function compareStoryboardItemsAsc(
  a: StoryboardItem,
  b: StoryboardItem,
) {
  const aSortOrder = a.sortOrder ?? Number.MAX_SAFE_INTEGER;
  const bSortOrder = b.sortOrder ?? Number.MAX_SAFE_INTEGER;
  if (aSortOrder !== bSortOrder) return aSortOrder - bSortOrder;

  const aShotNumber = a.shotNumber || a.autoShotNumber || "";
  const bShotNumber = b.shotNumber || b.autoShotNumber || "";
  const shotCompare = storyboardItemCollator.compare(aShotNumber, bShotNumber);
  if (shotCompare !== 0) return shotCompare;

  return a.id - b.id;
}
