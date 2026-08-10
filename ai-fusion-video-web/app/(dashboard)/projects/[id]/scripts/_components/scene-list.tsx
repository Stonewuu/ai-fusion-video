"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Film,
  Clapperboard,
  Loader2,
  Plus,
  Layers,
  Pencil,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ScriptEpisode, SceneItem } from "@/lib/api/script";
import { SceneCard } from "./scene-card";
import { toast } from "sonner";

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

/** 可编辑的分集标题 */
function EditableTitle({
  value,
  episodeNumber,
  scenesCount,
  onSave,
}: {
  value: string;
  episodeNumber: number;
  scenesCount: number;
  onSave: (newTitle: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const handleSave = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== value) {
      onSave(trimmed);
    } else {
      setDraft(value);
    }
    setEditing(false);
  };

  if (editing) {
    return (
      <div className="mb-5">
        <div className="flex items-center gap-2">
          <Film className="h-4 w-4 text-primary shrink-0" />
          <span className="text-base font-bold shrink-0">分集：</span>
          <input
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={handleSave}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSave();
              if (e.key === "Escape") { setDraft(value); setEditing(false); }
            }}
            className="flex-1 text-base font-bold bg-transparent border-b-2 border-primary/40 focus:border-primary outline-none py-0.5 min-w-0"
          />
          <span className="text-xs font-normal text-muted-foreground ml-2 shrink-0">
            共 {scenesCount} 场
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-5 group/title">
      <h2
        className="text-base font-bold flex items-center gap-2 cursor-pointer"
        onClick={() => setEditing(true)}
        title="点击编辑标题"
      >
        <Film className="h-4 w-4 text-primary" />
        分集：{value || "未命名"}
        <Pencil className="h-3 w-3 text-muted-foreground/0 group-hover/title:text-muted-foreground/50 transition-colors" />
        <span className="text-xs font-normal text-muted-foreground ml-2">
          共 {scenesCount} 场
        </span>
      </h2>
    </div>
  );
}

/** 可编辑的分集概览 */
function EditableSynopsis({
  value,
  onSave,
}: {
  value: string;
  onSave: (newSynopsis: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      // 自适应高度
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [editing]);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const handleSave = () => {
    const trimmed = draft.trim();
    if (trimmed !== value) {
      onSave(trimmed);
    }
    setEditing(false);
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDraft(e.target.value);
    // 自适应高度
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  };

  if (editing) {
    return (
      <textarea
        ref={textareaRef}
        value={draft}
        onChange={handleInput}
        onBlur={handleSave}
        onKeyDown={(e) => {
          if (e.key === "Escape") { setDraft(value); setEditing(false); }
        }}
        className="w-full text-xs text-muted-foreground leading-relaxed bg-transparent border border-primary/20 focus:border-primary/40 rounded-md px-2 py-1.5 outline-none resize-none -mt-0.5 mb-4"
        rows={2}
      />
    );
  }

  if (!value) {
    return (
      <p
        className="text-xs text-muted-foreground/40 mt-1 mb-4 leading-relaxed cursor-pointer hover:text-muted-foreground/60 italic transition-colors"
        onClick={() => { setDraft(""); setEditing(true); }}
      >
        点击添加分集概览...
      </p>
    );
  }

  return (
    <p
      className="text-xs text-muted-foreground mt-1 mb-4 leading-relaxed cursor-pointer hover:bg-muted/30 rounded-md px-1 -mx-1 py-0.5 transition-colors"
      onClick={() => setEditing(true)}
      title="点击编辑概览"
    >
      {value}
    </p>
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
      {/* 分集标题 - 可编辑 */}
      <EditableTitle
        value={activeEpisode.title || ""}
        episodeNumber={activeEpisode.episodeNumber}
        scenesCount={activeScenes.length}
        onSave={async (newTitle) => {
          try {
            const { scriptApi } = await import("@/lib/api/script");
            const updated = await scriptApi.updateEpisode({ id: activeEpisode.id, title: newTitle, version: activeEpisode.version });
            onEpisodeUpdated?.(updated);
          } catch (err: unknown) {
            toast.error(err instanceof Error ? err.message : "更新分集标题失败");
          }
        }}
      />

      {/* 分集概览 - 可编辑 */}
      <EditableSynopsis
        value={activeEpisode.synopsis || ""}
        onSave={async (newSynopsis) => {
          try {
            const { scriptApi } = await import("@/lib/api/script");
            const updated = await scriptApi.updateEpisode({ id: activeEpisode.id, synopsis: newSynopsis, version: activeEpisode.version });
            onEpisodeUpdated?.(updated);
          } catch (err: unknown) {
            toast.error(err instanceof Error ? err.message : "更新分集概览失败");
          }
        }}
      />

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

