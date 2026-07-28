"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Info,
  MapPin,
  Clock,
  Users,
  Package,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { SceneItem, DialogueElement } from "@/lib/api/script";
import { assetApi } from "@/lib/api/asset";
import { toastApiError } from "@/lib/api/toast-api-error";
import type { Asset, AssetItem } from "@/lib/api/asset";
import { parseDialogues, parseCharacters } from "./utils";
import { SafeImage } from "@/components/ui/safe-image";

// ========== 资产类型配置 ==========

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

type AssetWithItems = Asset & { items: AssetItem[] };

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

export function SceneDetail({
  scene,
  projectId,
}: {
  scene: SceneItem;
  projectId: number;
}) {
  const router = useRouter();
  const dialogues: DialogueElement[] = parseDialogues(scene);
  const chars: string[] = parseCharacters(scene);

  // ===== 加载关联资产 =====
  const [linkedAssets, setLinkedAssets] = useState<{
    characters: AssetWithItems[];
    scenes: AssetWithItems[];
    props: AssetWithItems[];
  }>({ characters: [], scenes: [], props: [] });
  const [assetsLoading, setAssetsLoading] = useState(false);

  const loadLinkedAssets = useCallback(async () => {
    const charIds = parseIds(scene.characterAssetIds);
    const sceneAssetId = scene.sceneAssetId;
    const propIds = parseIds(scene.propAssetIds);

    const allIds = [
      ...charIds,
      ...(sceneAssetId ? [sceneAssetId] : []),
      ...propIds,
    ];

    if (allIds.length === 0) {
      setLinkedAssets({ characters: [], scenes: [], props: [] });
      return;
    }

    setAssetsLoading(true);
    try {
      const results = await Promise.all(
        allIds.map(async (id) => {
          try {
            const [asset, items] = await Promise.all([
              assetApi.get(id),
              assetApi.listItems(id),
            ]);
            return { ...asset, items: items || [] };
          } catch {
            return null;
          }
        })
      );

      const valid = results.filter((r): r is AssetWithItems => r !== null);
      const charIdSet = new Set(charIds);
      const propIdSet = new Set(propIds);

      setLinkedAssets({
        characters: valid.filter((a) => charIdSet.has(a.id)),
        scenes: valid.filter((a) => a.id === sceneAssetId),
        props: valid.filter((a) => propIdSet.has(a.id)),
      });
    } catch (err) {
      console.error("加载场次关联资产失败:", err);
      toastApiError(err, "加载场次关联资产失败");
    } finally {
      setAssetsLoading(false);
    }
  }, [scene.characterAssetIds, scene.sceneAssetId, scene.propAssetIds]);

  useEffect(() => {
    loadLinkedAssets();
  }, [loadLinkedAssets]);

  const hasLinkedAssets =
    linkedAssets.characters.length > 0 ||
    linkedAssets.scenes.length > 0 ||
    linkedAssets.props.length > 0;

  return (
    <div className="p-4 space-y-5">
      {/* 场景头部信息 */}
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Info className="h-3 w-3" /> 场景详情
        </h4>

        {/* 场号和内外景 */}
        <div className="flex items-center gap-2 mb-3">
          <span className="text-sm font-bold">{scene.sceneNumber}</span>
          {scene.intExt && (
            <span
              className={cn(
                "text-[11px] px-2 py-0.5 rounded-md font-medium",
                scene.intExt === "内"
                  ? "bg-violet-500/10 text-violet-400 border border-violet-500/20"
                  : "bg-orange-500/10 text-orange-400 border border-orange-500/20"
              )}
            >
              {scene.intExt}
            </span>
          )}
        </div>

        {/* 场景标题 */}
        <p className="text-sm font-semibold mb-3">
          {scene.sceneHeading || "未命名场次"}
        </p>

        {/* 属性列表 */}
        <div className="space-y-2">
          {[
            { icon: MapPin, label: "位置", value: scene.location },
            { icon: Clock, label: "时间", value: scene.timeOfDay },
          ].map(({ icon: AttrIcon, label, value }) => (
            <div key={label} className="flex items-center gap-2 text-xs">
              <AttrIcon className="h-3 w-3 text-muted-foreground shrink-0" />
              <span className="text-muted-foreground">{label}</span>
              <span className="font-medium ml-auto truncate max-w-[140px]">
                {value || "—"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* 出场角色 */}
      {chars.length > 0 && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Users className="h-3 w-3" /> 出场角色
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {chars.map((char, i) => (
              <span
                key={i}
                className="text-[11px] px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20"
              >
                {char}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 场景描述 */}
      {scene.sceneDescription && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            场景描述
          </h4>
          <p className="text-xs text-muted-foreground leading-relaxed italic">
            {scene.sceneDescription}
          </p>
        </div>
      )}

      {/* 统计 */}
      <div className="border-t border-border/20 pt-4">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          统计
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="text-center p-2 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-primary">
              {dialogues.filter((d) => d.type === 1).length}
            </p>
            <p className="text-[10px] text-muted-foreground">对白数</p>
          </div>
          <div className="text-center p-2 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-amber-400">{chars.length}</p>
            <p className="text-[10px] text-muted-foreground">角色数</p>
          </div>
        </div>
      </div>

      {/* ===== 关联资产 ===== */}
      {assetsLoading && (
        <div className="border-t border-border/20 pt-4 flex items-center justify-center py-4">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
      )}

      {!assetsLoading && (
        <div className="border-t border-border/20 pt-4 space-y-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <Package className="h-3 w-3" /> 关联资产
          </h4>

          {(
            [
              ["character", linkedAssets.characters],
              ["scene", linkedAssets.scenes],
              ["prop", linkedAssets.props],
            ] as [keyof typeof typeConfig, AssetWithItems[]][]
          ).map(([type, assets]) => (
            <SceneAssetGroup
              key={type}
              type={type}
              assets={assets}
              onAssetClick={(id) =>
                router.push(
                  `/projects/${projectId}/assets?highlight=${id}`
                )
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** 场次关联资产分组卡片 */
function SceneAssetGroup({
  type,
  assets,
  onAssetClick,
}: {
  type: keyof typeof typeConfig;
  assets: AssetWithItems[];
  onAssetClick: (id: number) => void;
}) {
  const config = typeConfig[type];
  const Icon = config.icon;

  return (
    <div>
      <h5 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
        <Icon className={cn("h-3 w-3", config.color)} />
        {config.label}
        <span className="text-[10px] font-normal text-muted-foreground/60 ml-auto">
          {assets.length}
        </span>
      </h5>
      <div className="space-y-3">
        {assets.length === 0 ? (
          <p className="text-xs text-muted-foreground/50 px-2.5 py-1">无</p>
        ) : (
          assets.map((asset) => (
          <div key={asset.id} className="space-y-2">
            <button
              onClick={() => onAssetClick(asset.id)}
              className={cn(
                "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition-all text-left group",
                "hover:bg-muted/30"
              )}
            >
              <SafeImage
                src={resolveMediaUrl(asset.coverUrl) || undefined}
                alt={asset.name}
                className="h-9 w-9 rounded-lg object-cover"
                fallbackType={type === "character" ? "avatar" : type === "scene" ? "scene" : type === "prop" ? "prop" : "image"}
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium truncate">{asset.name}</p>
                {asset.description && (
                  <p className="text-[10px] text-muted-foreground/60 truncate mt-0.5">
                    {asset.description}
                  </p>
                )}
              </div>
              <ExternalLink className="h-3 w-3 text-muted-foreground/30 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
            </button>

            {/* 子资产图片网格 */}
            {asset.items.length > 0 && (
              <div className="grid grid-cols-3 gap-1.5 px-1">
                {asset.items
                  .filter((sub) => sub.imageUrl)
                  .slice(0, 6)
                  .map((sub) => (
                    <div
                      key={sub.id}
                      className="aspect-square rounded-lg overflow-hidden border border-border/10 bg-muted/20"
                      title={sub.name || undefined}
                    >
                      <SafeImage
                        src={resolveMediaUrl(sub.imageUrl) || undefined}
                        alt={sub.name || "子资产"}
                        className="w-full h-full object-cover"
                        fallbackType={type === "character" ? "avatar" : type === "scene" ? "scene" : type === "prop" ? "prop" : "image"}
                      />
                    </div>
                  ))}
              </div>
            )}
          </div>
        ))
        )}
      </div>
    </div>
  );
}
