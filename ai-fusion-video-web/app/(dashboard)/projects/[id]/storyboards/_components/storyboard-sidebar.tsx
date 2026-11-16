"use client";
import { Button } from "@/components/ui/button";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  useCallback,
  useRef,
} from "react";
import { useStoryboardInvalidation } from "../_hooks/use-storyboard-invalidation";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  ChevronRight,
  Plus,
  Clapperboard,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  RotateCw,
} from "lucide-react";
import {
  DndContext,
  DragOverlay,
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
} from "@dnd-kit/sortable";
import { createPortal } from "react-dom";
import { VirtualDirectory } from "@/components/shared/virtual-directory";
import {
  DirectoryTooltipScope,
  DirectoryTooltipTrigger,
} from "@/components/shared/directory-tooltip";
import { SortableSceneItem } from "./storyboard-sidebar-scene";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  storyboardApi,
  type StoryboardEpisode,
  StoryboardScene,
} from "@/lib/api/storyboard";
import type { ScriptEpisode } from "@/lib/api/script";
import {
  ScriptEpisodeBinding,
  ScriptEpisodeBindingReminder,
} from "./script-episode-binding";
// ========== 可拖拽场次项 ==========
interface SidebarSelection {
  type: "all" | "episode" | "scene";
  episodeId?: number;
  sceneId?: number;
}
interface StoryboardSidebarProps {
  projectId: number;
  storyboardId: number;
  storyboardVersion: number;
  selection: SidebarSelection;
  activeSceneId?: number | null;
  /** 是否以桌面端窄栏形态收起目录 */
  collapsed?: boolean;
  contained?: boolean;
  onSelect: (selection: SidebarSelection) => void;
  /** 切换桌面端目录收起状态 */
  onCollapsedChange?: (collapsed: boolean) => void;
  onRefresh?: () => Promise<void> | void;
  isRefreshing?: boolean;
  /** 初始加载完成时调用，传递第一集 episodeId（避免通过 onSelect 触发 page 的重复加载） */
  onInitialLoad?: (firstEpisodeId: number) => void;
  onDeleteEpisode?: (
    episode: StoryboardEpisode,
  ) => Promise<boolean | void> | boolean | void;
  onDeleteScene?: (
    scene: StoryboardScene,
  ) => Promise<boolean | void> | boolean | void;
  onReorderScenes?: (
    episode: StoryboardEpisode,
    sortedScenes: StoryboardScene[],
  ) => Promise<void>;
  scriptEpisodes?: ScriptEpisode[];
  onBindScriptEpisode?: (
    episode: StoryboardEpisode,
    scriptEpisodeId: number,
  ) => Promise<StoryboardEpisode>;
  onGenerateEpisodeStoryboard?: (
    episode: StoryboardEpisode,
  ) => Promise<void> | void;
}
export function StoryboardSidebar(props: StoryboardSidebarProps) {
  const requests = useCommandRequestKey();
  const {
    projectId,
    storyboardId,
    selection,
    activeSceneId,
    collapsed = false,
    onSelect,
    onCollapsedChange,
    onRefresh,
    isRefreshing,
    onDeleteEpisode,
    onDeleteScene,
    scriptEpisodes = [],
    onBindScriptEpisode,
    onGenerateEpisodeStoryboard,
  } = props;
  const [episodes, setEpisodes] = useState<StoryboardEpisode[]>([]);
  const [scenesMap, setScenesMap] = useState<Record<number, StoryboardScene[]>>(
    {},
  );
  const [expandedEpisodes, setExpandedEpisodes] = useState<Set<number>>(
    new Set(),
  );
  const [loading, setLoading] = useState(true);
  const [bindingEpisodeIds, setBindingEpisodeIds] = useState<Set<number>>(
    new Set(),
  );
  const [openBindingEpisodeId, setOpenBindingEpisodeId] = useState<
    number | null
  >(null);
  const isInitialLoadedRef = useRef(false);
  const shouldReduceMotion = useReducedMotion();
  const expandedRef = useRef(expandedEpisodes);
  const scenesRef = useRef(scenesMap);
  const selectionRef = useRef(selection);
  const pendingScenes = useRef(new Set<number>());
  const loadVersion = useRef(0);
  const [loadingEpisodeIds, setLoadingEpisodeIds] = useState<Set<number>>(
    new Set(),
  );
  const [failedEpisodeIds, setFailedEpisodeIds] = useState<Set<number>>(
    new Set(),
  );
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [draggedScene, setDraggedScene] = useState<StoryboardScene | null>(
    null,
  );
  useLayoutEffect(() => {
    expandedRef.current = expandedEpisodes;
    scenesRef.current = scenesMap;
    selectionRef.current = selection;
  }, [expandedEpisodes, scenesMap, selection]);
  const boundScriptEpisodeIds = useMemo(
    () =>
      new Set(
        episodes.flatMap((episode) =>
          episode.scriptEpisodeId == null ? [] : [episode.scriptEpisodeId],
        ),
      ),
    [episodes],
  );
  const availableScriptEpisodeIds = useMemo(
    () => new Set(scriptEpisodes.map((episode) => episode.id)),
    [scriptEpisodes],
  );
  const scriptByNumber = useMemo(
    () =>
      new Map(
        scriptEpisodes.map((episode) => [episode.episodeNumber, episode]),
      ),
    [scriptEpisodes],
  );
  const estimateEpisodeSize = useCallback(
    (index: number) => {
      const episode = episodes[index];
      return expandedEpisodes.has(episode.id)
        ? 76 + Math.max(1, scenesMap[episode.id]?.length ?? 1) * 32
        : 42;
    },
    [episodes, expandedEpisodes, scenesMap],
  );
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );
  const handleDragEnd = (episodeId: number) => (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) {
      return;
    }
    const scenes = scenesMap[episodeId] || [];
    const oldIndex = scenes.findIndex((s) => s.id === active.id);
    const newIndex = scenes.findIndex((s) => s.id === over.id);
    if (oldIndex !== -1 && newIndex !== -1 && props.onReorderScenes) {
      const prevScenes = scenes;
      const sorted = [...prevScenes];
      const [moved] = sorted.splice(oldIndex, 1);
      sorted.splice(newIndex, 0, moved);
      setScenesMap((prev) => ({ ...prev, [episodeId]: sorted }));
      const episode = episodes.find((item) => item.id === episodeId);
      if (!episode) {
        return;
      }
      props
        .onReorderScenes(episode, sorted)
        .then(() => loadEpisodes())
        .catch((error) => {
          toastApiError(error, "更新场次排序失败");
          setScenesMap((prev) => ({ ...prev, [episodeId]: prevScenes }));
        });
    }
  };
  // 目录首次只读取当前集，刷新时只读取已展开的分集。
  const loadEpisodes = useCallback(async () => {
    const version = ++loadVersion.current;
    try {
      const eps = await storyboardApi.listEpisodes(storyboardId);
      if (version !== loadVersion.current) {
        return;
      }
      setEpisodes(eps);
      const initial = !isInitialLoadedRef.current;
      const selected = selectionRef.current.episodeId;
      const first =
        eps.find((episode) => episode.id === selected)?.id ?? eps[0]?.id;
      const wanted = initial
        ? new Set(first === undefined ? [] : [first])
        : new Set(expandedRef.current);
      if (selected !== undefined) {
        wanted.add(selected);
      }
      if (initial) {
        setExpandedEpisodes(wanted);
      }
      const newScenesMap: Record<number, StoryboardScene[]> = {};
      const failed = new Set<number>();
      await Promise.all(
        eps
          .filter((episode) => wanted.has(episode.id))
          .map(async (episode) => {
            try {
              newScenesMap[episode.id] =
                await storyboardApi.listScenesByEpisode(episode.id);
            } catch (error) {
              failed.add(episode.id);
              toastApiError(error, "加载分集场次失败");
            }
          }),
      );
      if (version !== loadVersion.current) {
        return;
      }
      setScenesMap(newScenesMap);
      setFailedEpisodeIds(failed);
      if (initial && first !== undefined) {
        isInitialLoadedRef.current = true;
        props.onInitialLoad?.(first);
      }
    } catch (error) {
      toastApiError(error, "加载分镜目录失败");
    } finally {
      if (version === loadVersion.current) {
        setLoading(false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storyboardId]);
  useEffect(() => {
    loadEpisodes();
  }, [loadEpisodes, props.storyboardVersion]);
  useStoryboardInvalidation(projectId, loadEpisodes);
  const loadScenes = useCallback(async (episodeId: number) => {
    if (scenesRef.current[episodeId] || pendingScenes.current.has(episodeId)) {
      return;
    }
    pendingScenes.current.add(episodeId);
    setLoadingEpisodeIds((prev) => new Set(prev).add(episodeId));
    setFailedEpisodeIds((prev) => {
      const next = new Set(prev);
      next.delete(episodeId);
      return next;
    });
    try {
      const scenes = await storyboardApi.listScenesByEpisode(episodeId);
      setScenesMap((prev) => ({ ...prev, [episodeId]: scenes }));
    } catch (error) {
      setFailedEpisodeIds((prev) => new Set(prev).add(episodeId));
      toastApiError(error, "加载场次失败");
    } finally {
      pendingScenes.current.delete(episodeId);
      setLoadingEpisodeIds((prev) => {
        const next = new Set(prev);
        next.delete(episodeId);
        return next;
      });
    }
  }, []);
  const toggleEpisode = (id: number) => {
    const expanding = !expandedEpisodes.has(id);
    setExpandedEpisodes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
    if (expanding) {
      void loadScenes(id);
    }
  };
  useEffect(() => {
    const id = selection.episodeId;
    if (id !== undefined && !loading) {
      setExpandedEpisodes((prev) =>
        prev.has(id) ? prev : new Set(prev).add(id),
      );
      void loadScenes(id);
    }
  }, [selection.episodeId, loading, loadScenes]);
  // 添加分镜集
  const handleAddEpisode = async () => {
    try {
      const number =
        Math.max(0, ...episodes.map((episode) => episode.episodeNumber ?? 0)) +
        1;
      const payload = {
        storyboardId,
        parentVersion: props.storyboardVersion,
        episodeNumber: number,
        title: `第 ${number} 集`,
        sortOrder: episodes.length,
      };
      const ep = await storyboardApi.createEpisode(
        payload,
        requests.forPayload(payload),
      );
      requests.clear();
      await onRefresh?.();
      await loadEpisodes();
      setExpandedEpisodes((prev) => new Set(prev).add(ep.id));
      onSelect({ type: "episode", episodeId: ep.id });
    } catch (err) {
      console.error("创建分镜集失败:", err);
      toastApiError(err, "创建分镜集失败");
    }
  };
  // 添加分镜场次
  const handleAddScene = async (episodeId: number) => {
    const episode = episodes.find((item) => item.id === episodeId);
    if (!episode) {
      return;
    }
    try {
      const payload = { episodeId, parentVersion: episode.version };
      const scene = await storyboardApi.createScene(
        payload,
        requests.forPayload(payload),
      );
      requests.clear();
      await onRefresh?.();
      await loadEpisodes();
      onSelect({ type: "scene", episodeId, sceneId: scene.id });
    } catch (err) {
      console.error("创建场次失败:", err);
      toastApiError(err, "创建场次失败");
    }
  };
  const handleBindScriptEpisode = async (
    storyboardEpisodeId: number,
    value: string,
  ) => {
    if (!onBindScriptEpisode || !value) {
      return;
    }
    const scriptEpisodeId = Number(value);
    if (!Number.isFinite(scriptEpisodeId)) {
      return;
    }
    const episode = episodes.find((item) => item.id === storyboardEpisodeId);
    if (!episode) {
      return;
    }
    setBindingEpisodeIds((prev) => new Set(prev).add(storyboardEpisodeId));
    try {
      const updated = await onBindScriptEpisode(episode, scriptEpisodeId);
      setEpisodes((prev) =>
        prev.map((episode) => (episode.id === updated.id ? updated : episode)),
      );
    } catch (err) {
      console.error("绑定剧本集失败:", err);
      toastApiError(err, "绑定剧本集失败，请重试");
    } finally {
      setBindingEpisodeIds((prev) => {
        const next = new Set(prev);
        next.delete(storyboardEpisodeId);
        return next;
      });
    }
  };
  return (
    <DirectoryTooltipScope>
      <div
        data-sidebar-navigation="storyboards"
        className={cn(
          "relative min-h-0 shrink-0 overflow-hidden border-r border-border/20 bg-card/20 h-full transition-[width] duration-200 ease-out motion-reduce:transition-none",
          props.contained ? "w-full" : collapsed ? "w-12" : "w-64",
        )}
      >
        {onCollapsedChange && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-1.5 top-3 z-10"
            aria-label={collapsed ? "展开分镜目录" : "收起分镜目录"}
            title={collapsed ? "展开分镜目录" : "收起分镜目录"}
            aria-expanded={!collapsed}
            onClick={() => onCollapsedChange(!collapsed)}
          >
            {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
          </Button>
        )}
        <div
          inert={collapsed}
          aria-hidden={collapsed}
          className={cn(
            "flex h-full min-h-0 flex-col transition-opacity duration-150 motion-reduce:transition-none",
            props.contained ? "w-full" : "w-64",
            collapsed ? "pointer-events-none opacity-0" : "opacity-100",
          )}
        >
          {/* 标题栏 */}
          <div
            className={cn(
              "px-3 py-3 border-b border-primary/8 flex items-center justify-between",
              onCollapsedChange && "pr-12",
            )}
          >
            <h3 className="text-xs font-semibold text-primary/80 uppercase tracking-wider">
              分镜目录
            </h3>
            <div className="flex items-center gap-1">
              {onRefresh && (
                <button
                  type="button"
                  onClick={() => void onRefresh()}
                  disabled={isRefreshing}
                  className="h-6 w-6 rounded-md flex items-center justify-center text-muted-foreground/70 hover:text-primary hover:bg-primary/8 transition-colors disabled:opacity-50"
                  title="刷新分镜目录"
                  aria-label="刷新分镜目录"
                >
                  <RotateCw
                    className={cn(
                      "h-3.5 w-3.5",
                      isRefreshing && "animate-spin",
                    )}
                  />
                </button>
              )}
              <span className="text-[10px] text-primary/50 bg-primary/6 px-1.5 py-0.5 rounded-full tabular-nums font-medium">
                {episodes.length} 集
              </span>
            </div>
          </div>

          <div className="flex min-h-0 flex-1 flex-col">
            {/* 分镜集列表 */}
            {loading ? (
              <div className="px-3 py-4 text-[10px] text-muted-foreground/50 text-center">
                加载中…
              </div>
            ) : episodes.length === 0 ? (
              <div className="px-3 py-4 text-[10px] text-muted-foreground/50 text-center">
                暂无分镜集，点击 + 创建
              </div>
            ) : (
              <VirtualDirectory
                items={episodes}
                activeId={selection.episodeId ?? null}
                expandedIds={expandedEpisodes}
                draggingId={draggingId}
                estimateSize={estimateEpisodeSize}
                onToggle={toggleEpisode}
                label="分镜分集列表"
              >
                {(ep) => {
                  const isExpanded = expandedEpisodes.has(ep.id);
                  const isEpisodeSelected =
                    (selection.type === "episode" ||
                      selection.type === "scene") &&
                    selection.episodeId === ep.id;
                  const scenes = scenesMap[ep.id] || [];
                  const suggested = scriptByNumber.get(ep.episodeNumber ?? 0);
                  const suggestedScriptEpisode =
                    suggested && !boundScriptEpisodeIds.has(suggested.id)
                      ? suggested
                      : undefined;
                  const sourceMissing =
                    ep.scriptEpisodeId != null &&
                    !availableScriptEpisodeIds.has(ep.scriptEpisodeId);
                  const needsBinding =
                    ep.scriptEpisodeId == null || sourceMissing;
                  const bindingPanelId = `script-episode-binding-${ep.id}`;
                  const isBindingPanelOpen = openBindingEpisodeId === ep.id;
                  return (
                    <div key={ep.id} className="group/ep mb-0.5">
                      {/* 集标题行 */}
                      <div
                        data-sidebar-row
                        className={cn(
                          "flex items-center rounded-lg",
                          isEpisodeSelected
                            ? "bg-primary/8"
                            : "hover:bg-primary/5",
                        )}
                      >
                        <button
                          aria-label={`${isExpanded ? "收起" : "展开"}第 ${ep.episodeNumber} 集场次`}
                          aria-expanded={isExpanded}
                          onClick={() => toggleEpisode(ep.id)}
                          className={cn(
                            "p-2 shrink-0 transition-colors",
                            isEpisodeSelected
                              ? "text-primary/60 hover:text-primary"
                              : "text-muted-foreground/50 hover:text-foreground",
                          )}
                        >
                          <motion.div
                            animate={{ rotate: isExpanded ? 90 : 0 }}
                            transition={{ duration: 0.15, ease: "easeOut" }}
                          >
                            <ChevronRight className="h-3 w-3" />
                          </motion.div>
                        </button>
                        <DirectoryTooltipTrigger
                          text={ep.title || `第 ${ep.episodeNumber} 集`}
                          render={
                            <button
                              data-directory-select={ep.id}
                              onClick={() => {
                                if (!isExpanded) {
                                  toggleEpisode(ep.id);
                                }
                                onSelect({ type: "episode", episodeId: ep.id });
                              }}
                              className={cn(
                                "flex-1 flex items-center gap-1.5 py-2 pr-2 min-w-0 text-left text-sm",
                                isEpisodeSelected
                                  ? "text-primary font-medium"
                                  : "text-foreground/80 hover:text-foreground",
                              )}
                            >
                              <Clapperboard
                                className={cn(
                                  "h-3.5 w-3.5 shrink-0",
                                  isEpisodeSelected
                                    ? "text-primary"
                                    : "text-muted-foreground",
                                )}
                              />
                              <span
                                className={cn(
                                  "font-medium truncate text-xs",
                                  isEpisodeSelected && "text-primary",
                                )}
                              >
                                {ep.title || `第 ${ep.episodeNumber || "?"} 集`}
                              </span>
                            </button>
                          }
                        />
                        {!needsBinding && onGenerateEpisodeStoryboard && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onGenerateEpisodeStoryboard(ep);
                            }}
                            className="p-1.5 shrink-0 rounded-full opacity-0 group-hover/ep:opacity-100 text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 transition-all"
                            title="AI 重新生成本集分镜"
                          >
                            <Sparkles className="h-3 w-3" />
                          </button>
                        )}
                        <button
                          onClick={async (e) => {
                            e.stopPropagation();
                            if (onDeleteEpisode) {
                              const success = await onDeleteEpisode(ep);
                              if (success) {
                                setEpisodes((prev) =>
                                  prev.filter((o) => o.id !== ep.id),
                                );
                                setScenesMap((prev) => {
                                  const next = { ...prev };
                                  delete next[ep.id];
                                  return next;
                                });
                              }
                            }
                          }}
                          className="p-1.5 mr-1 shrink-0 rounded-full opacity-0 group-hover/ep:opacity-100 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
                          title="删除分集"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                        <button
                          onClick={() => handleAddScene(ep.id)}
                          className="p-1.5 mr-1 shrink-0 rounded-full opacity-0 group-hover/ep:opacity-100 text-muted-foreground hover:text-primary hover:bg-primary/10 transition-all"
                          title="添加场次"
                        >
                          <Plus className="h-3 w-3" />
                        </button>
                        {needsBinding && onBindScriptEpisode && (
                          <ScriptEpisodeBindingReminder
                            sourceMissing={sourceMissing}
                            isOpen={isBindingPanelOpen}
                            isBinding={bindingEpisodeIds.has(ep.id)}
                            panelId={bindingPanelId}
                            onToggle={() =>
                              setOpenBindingEpisodeId((current) =>
                                current === ep.id ? null : ep.id,
                              )
                            }
                          />
                        )}
                      </div>

                      <AnimatePresence initial={false}>
                        {needsBinding &&
                          onBindScriptEpisode &&
                          isBindingPanelOpen && (
                            <motion.div
                              id={bindingPanelId}
                              initial={
                                shouldReduceMotion
                                  ? false
                                  : { height: 0, opacity: 0 }
                              }
                              animate={{
                                height: "auto",
                                opacity: 1,
                                transition: {
                                  duration: shouldReduceMotion ? 0 : 0.2,
                                  ease: "easeOut",
                                },
                              }}
                              exit={{
                                height: 0,
                                opacity: 0,
                                transition: {
                                  duration: shouldReduceMotion ? 0 : 0.15,
                                  ease: "easeIn",
                                },
                              }}
                              className="overflow-hidden"
                            >
                              <ScriptEpisodeBinding
                                sourceMissing={sourceMissing}
                                episodes={scriptEpisodes}
                                boundEpisodeIds={boundScriptEpisodeIds}
                                suggestedEpisodeId={suggestedScriptEpisode?.id}
                                isBinding={bindingEpisodeIds.has(ep.id)}
                                onBind={(scriptEpisodeId) =>
                                  handleBindScriptEpisode(
                                    ep.id,
                                    String(scriptEpisodeId),
                                  )
                                }
                              />
                            </motion.div>
                          )}
                      </AnimatePresence>

                      {/* 展开的场次列表 */}
                      <AnimatePresence initial={false}>
                        {isExpanded && (
                          <motion.div
                            key="scenes"
                            initial={
                              shouldReduceMotion
                                ? false
                                : { height: 0, opacity: 0 }
                            }
                            animate={{
                              height: "auto",
                              opacity: 1,
                              transition: {
                                duration: shouldReduceMotion ? 0 : 0.2,
                                ease: "easeOut",
                              },
                            }}
                            exit={{
                              height: 0,
                              opacity: 0,
                              transition: {
                                duration: shouldReduceMotion ? 0 : 0.15,
                                ease: "easeIn",
                              },
                            }}
                            className="overflow-hidden"
                          >
                            <div className="ml-3 pl-3 border-l border-primary/15 py-0.5">
                              {loadingEpisodeIds.has(ep.id) ? (
                                <p className="p-3 text-xs text-muted-foreground">
                                  加载中...
                                </p>
                              ) : failedEpisodeIds.has(ep.id) ? (
                                <button
                                  onClick={() => void loadScenes(ep.id)}
                                  className="p-3 text-xs text-destructive"
                                >
                                  加载失败，点击重试
                                </button>
                              ) : scenes.length === 0 ? (
                                <div className="flex flex-col gap-2">
                                  <p className="text-[10px] text-muted-foreground/50 px-4 py-2 italic">
                                    暂无场次
                                  </p>
                                  <button
                                    onClick={() => handleAddScene(ep.id)}
                                    className="w-full flex items-center gap-1.5 px-2 py-1.5 text-[10px] text-muted-foreground/50 hover:text-muted-foreground rounded-md hover:bg-muted/20 transition-colors"
                                  >
                                    <Plus className="h-2.5 w-2.5" />
                                    添加场次
                                  </button>
                                </div>
                              ) : (
                                <>
                                  <DndContext
                                    sensors={sensors}
                                    collisionDetection={closestCenter}
                                    onDragStart={({ active }) => {
                                      setDraggingId(ep.id);
                                      setDraggedScene(
                                        scenes.find(
                                          (scene) => scene.id === active.id,
                                        )!,
                                      );
                                    }}
                                    onDragCancel={() => {
                                      setDraggingId(null);
                                      setDraggedScene(null);
                                    }}
                                    onDragEnd={(event) => {
                                      setDraggingId(null);
                                      setDraggedScene(null);
                                      handleDragEnd(ep.id)(event);
                                    }}
                                  >
                                    <SortableContext
                                      items={scenes.map((s) => s.id)}
                                      strategy={verticalListSortingStrategy}
                                    >
                                      {scenes.map((scene) => (
                                        <SortableSceneItem
                                          key={scene.id}
                                          scene={scene}
                                          isActive={activeSceneId === scene.id}
                                          onSelect={() =>
                                            onSelect({
                                              type: "scene",
                                              episodeId: ep.id,
                                              sceneId: scene.id,
                                            })
                                          }
                                          onDelete={async (e) => {
                                            e.stopPropagation();
                                            if (onDeleteScene) {
                                              const success =
                                                await onDeleteScene(scene);
                                              if (success) {
                                                setScenesMap((prev) => ({
                                                  ...prev,
                                                  [ep.id]: (
                                                    prev[ep.id] || []
                                                  ).filter(
                                                    (s) => s.id !== scene.id,
                                                  ),
                                                }));
                                              }
                                            }
                                          }}
                                        />
                                      ))}
                                    </SortableContext>
                                    {typeof document !== "undefined" &&
                                      createPortal(
                                        <DragOverlay dropAnimation={null}>
                                          {draggedScene && (
                                            <div className="rounded-lg border border-border/40 bg-popover p-2 text-xs shadow-xl">
                                              {draggedScene.sceneHeading ||
                                                "场次"}
                                            </div>
                                          )}
                                        </DragOverlay>,
                                        document.body,
                                      )}
                                  </DndContext>
                                  <button
                                    onClick={() => handleAddScene(ep.id)}
                                    className="w-full flex items-center gap-1.5 px-2 py-1.5 mt-1 text-[10px] text-muted-foreground/40 hover:text-muted-foreground rounded-md hover:bg-muted/20 transition-colors"
                                  >
                                    <Plus className="h-2.5 w-2.5" />
                                    添加场次
                                  </button>
                                </>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                }}
              </VirtualDirectory>
            )}
          </div>
          <div className="px-3 py-2 border-t border-primary/8">
            <button
              onClick={handleAddEpisode}
              className={cn(
                "w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium",
                "text-primary/60 bg-primary/5 transition-colors",
                "border border-dashed border-primary/20 hover:text-primary hover:bg-primary/10 hover:border-primary/40",
              )}
            >
              <Plus className="h-3 w-3" />
              添加分镜集
            </button>
          </div>
        </div>
      </div>
    </DirectoryTooltipScope>
  );
}
