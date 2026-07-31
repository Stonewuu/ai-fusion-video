"use client";

import {
  ChevronRight,
  Film,
  Clapperboard,
  Loader2,
  Plus,
  Trash2,
  GripVertical,
  Sparkles,
  PanelLeftClose,
  PanelLeftOpen,
  RotateCw,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import type { ScriptEpisode, SceneItem } from "@/lib/api/script";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import type { DragEndEvent } from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

// ========== 可拖拽场次项 ==========

function SortableSceneItem({
  scene,
  isSelected,
  onSelect,
  onDelete,
}: {
  scene: SceneItem;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: scene.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "flex items-center group/scene rounded-md transition-colors overflow-hidden",
        isDragging && "opacity-50 z-10",
        isSelected ? "bg-primary/10" : "hover:bg-primary/5"
      )}
    >
      <span
        {...attributes}
        {...listeners}
        className="px-1.5 self-stretch flex items-center opacity-0 group-hover/scene:opacity-40 hover:opacity-100! text-muted-foreground cursor-grab active:cursor-grabbing transition-opacity shrink-0"
        onPointerDown={(e) => {
          // 阻止 click 冒泡到 onSelect
          e.stopPropagation();
          listeners?.onPointerDown?.(e);
        }}
      >
        <GripVertical className="h-3 w-3" />
      </span>
      <button
        onClick={onSelect}
        className={cn(
          "flex-1 flex items-center gap-1.5 py-1.5 pr-2 min-w-0 outline-none text-left text-[11px]",
          isSelected
            ? "text-primary font-medium"
            : "text-muted-foreground hover:text-foreground"
        )}
      >
        <Clapperboard className={cn("h-3 w-3 shrink-0", isSelected && "text-primary")} />
        <span className="truncate">
          {scene.sceneNumber} {scene.sceneHeading || `场次`}
        </span>
      </button>
      <button
        onClick={onDelete}
        className="p-1.5 mr-1 flex items-center shrink-0 rounded-full opacity-0 group-hover/scene:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
        title="删除场次"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>
  );
}

// ========== 主组件 ==========

export function EpisodeTree({
  episodes,
  expandedEpisodes,
  activeEpisodeId,
  selectedSceneId,
  loadingEpisodes,
  episodeScenes,
  collapsed = false,
  onRefresh,
  isRefreshing = false,
  onToggleEpisode,
  onSelectEpisode,
  onSelectScene,
  onCollapsedChange,
  onAddEpisode,
  onDeleteEpisode,
  onAddScene,
  onDeleteScene,
  onReorderScenes,
  onParseEpisode,
  onGenerateStoryboard,
}: {
  episodes: ScriptEpisode[];
  expandedEpisodes: Set<number>;
  activeEpisodeId: number | null;
  selectedSceneId: number | null;
  loadingEpisodes: Set<number>;
  episodeScenes: Record<number, SceneItem[]>;
  collapsed?: boolean;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  onToggleEpisode: (id: number) => void;
  onSelectEpisode: (id: number) => void;
  onSelectScene: (sceneId: number, episodeId: number) => void;
  onCollapsedChange?: (collapsed: boolean) => void;
  onAddEpisode: () => void;
  onDeleteEpisode: (id: number) => void;
  onAddScene: (episodeId: number) => void;
  onDeleteScene: (sceneId: number, episodeId: number) => void;
  onReorderScenes?: (episodeId: number, oldIndex: number, newIndex: number) => void;
  onParseEpisode?: (episodeId: number) => void;
  onGenerateStoryboard?: (episodeId: number) => void;
}) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  const handleDragEnd = (episodeId: number) => (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const scenes = episodeScenes[episodeId] || [];
    const oldIndex = scenes.findIndex((s) => s.id === active.id);
    const newIndex = scenes.findIndex((s) => s.id === over.id);
    if (oldIndex !== -1 && newIndex !== -1) {
      onReorderScenes?.(episodeId, oldIndex, newIndex);
    }
  };

  if (collapsed) {
    return (
      <div className="w-12 border-r border-border/20 flex flex-col shrink-0 bg-card/20 h-full transition-[width] duration-200">
        <div className="px-2 py-3 border-b border-primary/8 flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => onCollapsedChange?.(false)}
            className="h-8 w-8 rounded-lg flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-primary/8 transition-colors"
            title="展开剧本目录"
            aria-label="展开剧本目录"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
          <Film className="h-4 w-4 text-primary/70" />
        </div>
      </div>
    );
  }

  return (
    <div className="w-full lg:w-64 border-r border-border/20 flex flex-col shrink-0 bg-card/20 h-full transition-[width] duration-200">
      <div className="px-4 py-3 border-b border-primary/8 flex items-center justify-between">
        <h3 className="text-xs font-semibold text-primary/80 uppercase tracking-wider">
          剧本目录
        </h3>
        <div className="flex items-center gap-1">
          {onRefresh && (
            <button
              type="button"
              onClick={onRefresh}
              disabled={isRefreshing}
              className="h-6 w-6 rounded-md flex items-center justify-center text-muted-foreground/70 hover:text-primary hover:bg-primary/8 transition-colors disabled:opacity-50"
              title="刷新剧本目录"
              aria-label="刷新剧本目录"
            >
              <RotateCw className={cn("h-3.5 w-3.5", isRefreshing && "animate-spin")} />
            </button>
          )}
          <span className="text-[10px] text-primary/50 bg-primary/6 px-1.5 py-0.5 rounded-full tabular-nums font-medium">
            {episodes.length} 集
          </span>
          {onCollapsedChange && (
            <button
              type="button"
              onClick={() => onCollapsedChange(true)}
              className="h-6 w-6 rounded-md flex items-center justify-center text-muted-foreground/70 hover:text-primary hover:bg-primary/8 transition-colors"
              title="收起剧本目录"
              aria-label="收起剧本目录"
            >
              <PanelLeftClose className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto py-1.5 px-1.5">
        {episodes.map((ep) => {
          const expanded = expandedEpisodes.has(ep.id);
          const scenes = episodeScenes[ep.id] || [];
          const isLoading = loadingEpisodes.has(ep.id);
          const isActive = ep.id === activeEpisodeId;

          return (
            <div key={ep.id} className="group/ep mb-0.5">
              <div
                className={cn(
                  "flex items-center rounded-lg transition-colors overflow-hidden",
                  isActive ? "bg-primary/8" : "hover:bg-primary/5"
                )}
              >
                <button
                  onClick={() => onToggleEpisode(ep.id)}
                  className={cn(
                    "p-2 shrink-0 transition-colors",
                    isActive ? "text-primary/60 hover:text-primary" : "text-muted-foreground/50 hover:text-foreground"
                  )}
                >
                  <motion.div
                    animate={{ rotate: expanded ? 90 : 0 }}
                    transition={{ duration: 0.15, ease: "easeOut" }}
                  >
                    <ChevronRight className="h-3 w-3" />
                  </motion.div>
                </button>
                <button
                  onClick={() => {
                    if (!expanded) onToggleEpisode(ep.id);
                    onSelectEpisode(ep.id);
                  }}
                  className={cn(
                    "flex-1 flex items-center gap-1.5 py-2 pr-2 min-w-0 text-left text-sm",
                    isActive ? "text-primary font-medium" : "text-foreground/80 hover:text-foreground"
                  )}
                >
                  <Film className={cn("h-3.5 w-3.5 shrink-0", isActive ? "text-primary" : "text-muted-foreground")} />
                  <span className={cn("font-medium truncate text-xs", isActive && "text-primary")}>
                    {ep.title || `第 ${ep.episodeNumber} 集`}
                  </span>
                </button>
                {onParseEpisode && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onParseEpisode(ep.id); }}
                    className="p-1.5 shrink-0 rounded-full opacity-0 group-hover/ep:opacity-100 text-purple-400 hover:text-purple-300 hover:bg-purple-500/10 transition-all"
                    title="AI 解析该集"
                  >
                    <Sparkles className="h-3 w-3" />
                  </button>
                )}
                {onGenerateStoryboard && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onGenerateStoryboard(ep.id); }}
                    className="p-1.5 shrink-0 rounded-full opacity-0 group-hover/ep:opacity-100 text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 transition-all"
                    title="AI 生成该集分镜"
                  >
                    <Film className="h-3 w-3" />
                  </button>
                )}
                <button
                  onClick={() => onDeleteEpisode(ep.id)}
                  className="p-1.5 mr-1 shrink-0 rounded-full opacity-0 group-hover/ep:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
                  title="删除分集"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>

              <AnimatePresence initial={false}>
                {expanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: "easeInOut" }}
                    className="overflow-hidden"
                  >
                    <div className="ml-3 pl-3 border-l border-primary/15 py-0.5">
                      {isLoading ? (
                        <div className="flex items-center gap-2 px-4 py-2">
                          <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                          <span className="text-[10px] text-muted-foreground">加载中...</span>
                        </div>
                      ) : scenes.length === 0 ? (
                        <p className="text-[10px] text-muted-foreground/50 px-4 py-2 italic">暂无场次</p>
                      ) : (
                        <DndContext
                          sensors={sensors}
                          collisionDetection={closestCenter}
                          onDragEnd={handleDragEnd(ep.id)}
                        >
                          <SortableContext
                            items={scenes.map((s) => s.id)}
                            strategy={verticalListSortingStrategy}
                          >
                            {scenes.map((scene) => (
                              <SortableSceneItem
                                key={scene.id}
                                scene={scene}
                                isSelected={selectedSceneId === scene.id}
                                onSelect={() => onSelectScene(scene.id, ep.id)}
                                onDelete={() => onDeleteScene(scene.id, ep.id)}
                              />
                            ))}
                          </SortableContext>
                        </DndContext>
                      )}
                      <button
                        onClick={() => onAddScene(ep.id)}
                        className={cn(
                          "w-full flex items-center gap-1.5 pl-6 pr-3 py-1.5 text-[10px] rounded-md",
                          "text-muted-foreground/50 hover:text-primary hover:bg-primary/5 transition-colors"
                        )}
                      >
                        <Plus className="h-2.5 w-2.5" />
                        添加场次
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
      <div className="px-3 py-2 border-t border-primary/8">
        <button
          onClick={onAddEpisode}
          className={cn(
            "w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium",
            "text-primary/60 bg-primary/5 transition-colors",
            "border border-dashed border-primary/20 hover:text-primary hover:bg-primary/10 hover:border-primary/40"
          )}
        >
          <Plus className="h-3 w-3" />
          添加分集
        </button>
      </div>
    </div>
  );
}
