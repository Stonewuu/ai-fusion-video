"use client";

import { SafeImage } from "@/components/media/managed-image";
import type { Asset, AssetItem } from "@/lib/api/asset";
import { resolveMediaUrl } from "@/lib/api/client";
import type { Project } from "@/lib/api/project";
import type {
  Storyboard,
  StoryboardFrameType,
  StoryboardItem,
} from "@/lib/api/storyboard";
import { X } from "lucide-react";
import { useCallback, useState } from "react";
import { ItemDetail } from "./storyboard-reference/item-detail";
import {
  BatchFrameGenerateHandler,
  SceneWithItems,
} from "./storyboard-reference/reference-shared";
import { SceneAssetPanel } from "./storyboard-reference/scene-asset-panel";
import { StoryboardOverview } from "./storyboard-reference/storyboard-overview";

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
  onUpdateFrame?: (
    item: StoryboardItem,
    frameType: StoryboardFrameType,
    imageUrl: string | null,
  ) => Promise<void> | void;
  onGenerateFrame?: (
    item: StoryboardItem,
    frameType: StoryboardFrameType,
    prompt: string,
  ) => Promise<void> | void;
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
          <div
            className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-3"
            onClick={(e) => e.stopPropagation()}
          >
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

export { BatchFrameGenerateControl } from "./storyboard-reference/batch-frame-generate";
export type {
  BatchFrameGenerateHandler,
  BatchFrameGeneratePayload,
  SceneWithItems,
} from "./storyboard-reference/reference-shared";
