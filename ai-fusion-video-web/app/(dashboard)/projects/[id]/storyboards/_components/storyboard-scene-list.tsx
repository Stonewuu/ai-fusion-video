"use client";
import { Camera, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { StoryboardTableView } from "./storyboard-table-view";
import { StoryboardCardView } from "./storyboard-card-view";
import type { useStoryboardWorkspace } from "../_hooks/use-storyboard-workspace";

export function StoryboardSceneList({ workspace }: { workspace: ReturnType<typeof useStoryboardWorkspace> }) {
  const {
    assetLookup,
    setEditAssetsOpen,
    setEditingItem,
    viewMode,
    selectedItemId,
    sceneGroups,
    loadingScenes,
    sceneRefs,
    scrollContainerRef,
    activeSceneId,
    setActiveSceneId,
    handleSelectItem,
    handleAddItem,
    handleDeleteItem,
    handleUpdateItemField,
    handleOpenFrameDialog,
    handleReorderItems,
    handleVideoGen,
  } = workspace;
  return (

        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-y-auto px-6 py-5 space-y-8"
        >
          {loadingScenes ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : sceneGroups.length === 0 ? (
            <div className="text-center py-16">
              <Camera className="h-10 w-10 text-muted-foreground/20 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">
                暂无分镜内容，请在左侧创建分集和场次，或使用顶部 AI 生成
              </p>
            </div>
          ) : (
            sceneGroups.map(({ scene, items }) => (
              <div
                key={scene.id}
                data-scene-id={scene.id}
                ref={(el) => {
                  sceneRefs.current[scene.id] = el;
                }}
                className={cn(
                  "scroll-mt-4 p-5 rounded-xl border transition-colors duration-200 motion-reduce:transition-none",
                  activeSceneId === scene.id
                    ? "bg-primary/5 border-primary/30"
                    : "border-transparent bg-transparent"
                )}
                onClick={() => setActiveSceneId(scene.id)}
              >
                {/* 场次标题：点击时亦可切换激活场次 */}
                <div 
                  className="flex items-center gap-2 mb-3 cursor-pointer group/title"
                  onClick={() => setActiveSceneId(scene.id)}
                >
                  <Camera className={cn(
                    "h-3.5 w-3.5 transition-colors",
                    activeSceneId === scene.id ? "text-primary" : "text-primary/60 group-hover/title:text-primary"
                  )} />
                  <h3 className={cn(
                    "text-sm font-semibold transition-colors",
                    activeSceneId === scene.id ? "text-primary" : "group-hover/title:text-primary"
                  )}>
                    {scene.sceneHeading ||
                      `场次 ${scene.sceneNumber || scene.id}`}
                  </h3>
                  {scene.location && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-muted/30 text-muted-foreground">
                      {scene.intExt && `${scene.intExt} `}
                      {scene.location}
                      {scene.timeOfDay && ` ${scene.timeOfDay}`}
                    </span>
                  )}
                  <span className="text-[10px] text-muted-foreground/50 ml-auto">
                    {items.length} 镜
                  </span>
                </div>

                {/* 场次内的镜头列表 */}
                {viewMode === "table" ? (
                  <StoryboardTableView
                    items={items}
                    selectedItemId={selectedItemId}
                    onSelectItem={handleSelectItem}
                    onUpdateItemField={handleUpdateItemField}
                    onAddItem={() =>
                      handleAddItem(scene.id)
                    }
                    onDeleteItem={handleDeleteItem}
                    onReorderItems={(reordered) =>
                      handleReorderItems(scene.id, reordered)
                    }
                    onVideoGen={handleVideoGen}
                    onOpenFrameDialog={handleOpenFrameDialog}
                    assetLookup={assetLookup}
                    onEditAssets={(item) => {
                      setEditingItem(item);
                      setEditAssetsOpen(true);
                    }}
                  />
                ) : (
                  <StoryboardCardView
                    items={items}
                    selectedItemId={selectedItemId}
                    onSelectItem={handleSelectItem}
                    onAddItem={() =>
                      handleAddItem(scene.id)
                    }
                    onDeleteItem={handleDeleteItem}
                    onReorderItems={(reordered) =>
                      handleReorderItems(scene.id, reordered)
                    }
                    onVideoGen={handleVideoGen}
                    onOpenFrameDialog={handleOpenFrameDialog}
                  />
                )}
              </div>
            ))
          )}
        </div>

  );
}