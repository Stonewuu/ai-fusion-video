"use client";

import { useState } from "react";
import {
  Clapperboard,
  Loader2,
  Plus,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ScriptEpisode, SceneItem } from "@/lib/api/script";
import { SceneCard } from "./scene-card";
import { EpisodeDetailsEditor } from "./episode-details-editor";

/** 场次间插入按钮 – absolute 定位在卡片上方间隙居中 */
function SceneInsertIndicator({
  visible,
  onClick,
}: {
  visible: boolean;
  onClick: () => void;
}) {
  return (
    <div
      className={cn(
        "absolute left-1/2 -translate-x-1/2 -translate-y-1/2 -top-[10px] z-10 transition-opacity duration-150",
        visible ? "opacity-100" : "opacity-0 pointer-events-none"
      )}
    >
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        className="flex items-center gap-1 px-3 py-1 rounded-full bg-primary/80 text-primary-foreground hover:bg-primary hover:scale-105 active:scale-95 transition-all shadow-sm cursor-pointer text-[11px] font-medium whitespace-nowrap"
        title="插入场次"
      >
        <Plus className="h-3 w-3" />
        新增场次
      </button>
    </div>
  );
}

export function SceneList({
  activeEpisode,
  activeScenes,
  isLoading,
  selectedSceneId,
  onSelectScene,
  onSceneUpdated,
  onAddScene,
  onInsertScene,
  onDeleteScene,
  onEpisodeUpdated,
}: {
  activeEpisode: ScriptEpisode | undefined;
  activeScenes: SceneItem[];
  isLoading: boolean;
  selectedSceneId: number | null;
  onSelectScene: (id: number) => void;
  onSceneUpdated: (episodeId: number, updated: SceneItem) => void;
  onAddScene: (episodeId: number) => void;
  onInsertScene: (episodeId: number, atIndex: number) => void;
  onDeleteScene: (sceneId: number, episodeId: number) => void;
  onEpisodeUpdated?: (updated: ScriptEpisode) => void;
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!activeEpisode) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <Layers className="h-12 w-12 text-muted-foreground/20 mb-4" />
        <p className="text-muted-foreground text-sm">从左侧选择一个分集查看场次内容</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-5">
      {activeEpisode && <EpisodeDetailsEditor episode={activeEpisode} scenesCount={activeScenes.length} onSaved={onEpisodeUpdated} />}

      {/* 场次卡片列表 */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : activeScenes.length === 0 ? (
        <div className="text-center py-16">
          <Clapperboard className="h-10 w-10 text-muted-foreground/20 mx-auto mb-3" />
          <p className="text-sm text-muted-foreground mb-3">本集暂无场次</p>
          <button
            onClick={() => onAddScene(activeEpisode.id)}
            className={cn(
              "inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium",
              "bg-primary text-primary-foreground hover:opacity-90 transition-all"
            )}
          >
            <Plus className="h-3 w-3" />
            添加第一个场次
          </button>
        </div>
      ) : (
        <div className="space-y-5">
          {activeScenes.map((scene, i) => (
            <div
              key={scene.id}
              className="relative"
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {/* 当前卡片上方的插入按钮（hover 当前或上一个卡片时显示） */}
              <SceneInsertIndicator
                visible={hoveredIdx === i || hoveredIdx === i - 1}
                onClick={() => onInsertScene(activeEpisode.id, i)}
              />
              <SceneCard
                scene={scene}
                isSelected={selectedSceneId === scene.id}
                onSelect={() => onSelectScene(scene.id)}
                onSceneUpdated={(updated) =>
                  onSceneUpdated(activeEpisode.id, updated)
                }
                onDelete={() => onDeleteScene(scene.id, activeEpisode.id)}
              />
            </div>
          ))}
          {/* 末尾插入按钮 */}
          <div
            className="relative h-3"
            onMouseEnter={() => setHoveredIdx(activeScenes.length)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            <SceneInsertIndicator
              visible={hoveredIdx === activeScenes.length || hoveredIdx === activeScenes.length - 1}
              onClick={() => onInsertScene(activeEpisode.id, activeScenes.length)}
            />
          </div>

          {/* 底部添加场次 */}
          <button
            onClick={() => onAddScene(activeEpisode.id)}
            className={cn(
              "w-full flex items-center justify-center gap-2 py-4 rounded-xl",
              "border-2 border-dashed border-border/30 hover:border-primary/30",
              "text-muted-foreground hover:text-primary transition-all",
              "text-xs font-medium"
            )}
          >
            <Plus className="h-3.5 w-3.5" />
            添加场次
          </button>
        </div>
      )}
    </div>
  );
}

