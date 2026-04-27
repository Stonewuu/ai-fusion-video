"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import {
  Film,
  Plus,
  Loader2,
  Sparkles,
  Table2,
  LayoutGrid,
  Camera,
  Menu,
  Info,
} from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  storyboardApi,
  type Storyboard,
  type StoryboardItem,
  type StoryboardScene,
  type StoryboardEpisode,
} from "@/lib/api/storyboard";
import { StoryboardSidebar } from "./_components/storyboard-sidebar";
import { StoryboardTableView } from "./_components/storyboard-table-view";
import { StoryboardCardView } from "./_components/storyboard-card-view";
import { StoryboardRefPanel } from "./_components/storyboard-ref-panel";
import { CreateStoryboardDialog } from "./_components/create-dialog";
import { useFullWidth } from "@/lib/hooks/use-layout";

type ViewMode = "table" | "card";

interface SidebarSelection {
  type: "all" | "episode" | "scene";
  episodeId?: number;
  sceneId?: number;
}

/** 场次及其条目 */
interface SceneWithItems {
  scene: StoryboardScene;
  items: StoryboardItem[];
}

export default function StoryboardTabPage() {
  const params = useParams();
  const projectId = Number(params.id);

  // 分镜页始终占满 layout 宽度
  useFullWidth(true);

  const [loading, setLoading] = useState(true);
  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // 视图状态
  const [viewMode, setViewMode] = useState<ViewMode>("table");

  // 加载本地用户偏好
  useEffect(() => {
    const savedMode = localStorage.getItem("fusion-storyboard-view-mode");
    if (savedMode === "table" || savedMode === "card") {
      setViewMode(savedMode);
    }
  }, []);

  const handleSetViewMode = useCallback((mode: ViewMode) => {
    setViewMode(mode);
    localStorage.setItem("fusion-storyboard-view-mode", mode);
  }, []);
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [sidebarSelection, setSidebarSelection] = useState<SidebarSelection>({
    type: "episode",
  });

  // 移动端侧边栏状态
  const [leftSheetOpen, setLeftSheetOpen] = useState(false);
  const [rightSheetOpen, setRightSheetOpen] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(min-width: 1024px)");
    const handleResize = (e: MediaQueryListEvent | MediaQueryList) => {
      if (e.matches) {
        setLeftSheetOpen(false);
        setRightSheetOpen(false);
      }
    };
    handleResize(mediaQuery);
    mediaQuery.addEventListener("change", handleResize);
    return () => mediaQuery.removeEventListener("change", handleResize);
  }, []);

  // 按场次分组数据
  const [sceneGroups, setSceneGroups] = useState<SceneWithItems[]>([]);
  const [loadingScenes, setLoadingScenes] = useState(false);

  // 滚动定位 refs
  const sceneRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // 滚动时当前可见的场次 ID（用于侧边栏高亮）
  const [activeSceneId, setActiveSceneId] = useState<number | null>(null);
  // 标记是否由用户点击触发的滚动（此时不要通过 observer 覆盖）
  const isUserScrollRef = useRef(false);

  // 加载分镜
  const loadStoryboard = useCallback(async () => {
    try {
      setLoading(true);
      const list = await storyboardApi.list(projectId);
      if (list.length > 0) {
        setStoryboard(list[0]);
      } else {
        setStoryboard(null);
        setSceneGroups([]);
      }
    } catch (err) {
      console.error("加载分镜失败:", err);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadStoryboard();
  }, [loadStoryboard]);

  // AI 工具执行后自动刷新
  const storyboardsInvalidation = usePipelineStore((s) => s.invalidation.storyboards);
  const storyboardsInvRef = useRef(storyboardsInvalidation);
  useEffect(() => {
    if (storyboardsInvRef.current !== storyboardsInvalidation) {
      storyboardsInvRef.current = storyboardsInvalidation;
      loadStoryboard();
    }
  }, [storyboardsInvalidation, loadStoryboard]);

  // 加载场次分组数据
  const loadSceneGroups = useCallback(
    async (episodeId?: number) => {
      if (!storyboard) return;
      setLoadingScenes(true);
      try {
        let scenes: StoryboardScene[];
        if (episodeId) {
          scenes = await storyboardApi.listScenesByEpisode(episodeId);
        } else {
          scenes = await storyboardApi.listScenesByStoryboard(storyboard.id);
        }

        // 并行加载每个场次的条目
        const groups = await Promise.all(
          scenes.map(async (scene) => {
            const items = await storyboardApi.listItemsByScene(scene.id);
            return { scene, items };
          })
        );

        setSceneGroups(groups);
      } catch (err) {
        console.error("加载场次失败:", err);
      } finally {
        setLoadingScenes(false);
      }
    },
    [storyboard]
  );

  // 追踪当前已加载的集ID，避免同集内切换场次重复加载
  const loadedEpisodeIdRef = useRef<number | null>(null);

  // sidebar 初始化完成后通知 page 第一集 episodeId
  const handleSidebarInitialLoad = useCallback((firstEpisodeId: number) => {
    setSidebarSelection({ type: "episode", episodeId: firstEpisodeId });
  }, []);

  // 当侧边栏选择变化时加载数据
  useEffect(() => {
    if (!storyboard) return;

    if (sidebarSelection.type === "all") {
      setActiveSceneId(null);
      loadedEpisodeIdRef.current = null;
      loadSceneGroups();
    } else if (sidebarSelection.type === "episode" && sidebarSelection.episodeId) {
      setActiveSceneId(null);
      loadedEpisodeIdRef.current = sidebarSelection.episodeId;
      loadSceneGroups(sidebarSelection.episodeId);
    } else if (
      sidebarSelection.type === "scene" &&
      sidebarSelection.sceneId
    ) {
      const sceneExists = sceneGroups.some(
        (g) => g.scene.id === sidebarSelection.sceneId
      );
      // 同一集且场次已存在于数据中：直接滚动
      if (
        sidebarSelection.episodeId &&
        loadedEpisodeIdRef.current === sidebarSelection.episodeId &&
        sceneExists
      ) {
        setTimeout(() => {
          scrollToScene(sidebarSelection.sceneId!);
        }, 50);
      } else if (sidebarSelection.episodeId) {
        // 不同集 / 首次加载 / 新添加的场次：重新加载数据再滚动
        loadedEpisodeIdRef.current = sidebarSelection.episodeId;
        loadSceneGroups(sidebarSelection.episodeId).then(() => {
          setTimeout(() => {
            scrollToScene(sidebarSelection.sceneId!);
          }, 100);
        });
      }
    }
  }, [sidebarSelection, storyboard, loadSceneGroups]);

  // 滚动到指定场次
  const scrollToScene = (sceneId: number) => {
    isUserScrollRef.current = true;
    setActiveSceneId(sceneId);
    const el = sceneRefs.current[sceneId];
    if (el && scrollContainerRef.current) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    // 滚动动画完成后恢复 observer
    setTimeout(() => {
      isUserScrollRef.current = false;
    }, 600);
  };

  // ========== 滚动监听：IntersectionObserver ==========
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || sceneGroups.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (isUserScrollRef.current) return;
        // 找到最靠近顶部且可见的场次
        let topEntry: IntersectionObserverEntry | null = null;
        for (const entry of entries) {
          if (entry.isIntersecting) {
            if (!topEntry || entry.boundingClientRect.top < topEntry.boundingClientRect.top) {
              topEntry = entry;
            }
          }
        }
        if (topEntry) {
          const id = Number((topEntry.target as HTMLElement).dataset.sceneId);
          if (id) setActiveSceneId(id);
        }
      },
      {
        root: container,
        rootMargin: "-10% 0px -70% 0px",
        threshold: 0,
      }
    );

    // 观察所有场次元素
    for (const [, el] of Object.entries(sceneRefs.current)) {
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, [sceneGroups]);

  // ========== 操作 ==========

  const handleAddItem = async (sceneId: number, episodeId?: number) => {
    if (!storyboard) return;
    const group = sceneGroups.find((g) => g.scene.id === sceneId);
    const currentItems = group?.items || [];
    try {
      const newItem = await storyboardApi.createItem({
        storyboardId: storyboard.id,
        storyboardSceneId: sceneId,
        storyboardEpisodeId: episodeId,
        sortOrder: currentItems.length,
        shotNumber: String(currentItems.length + 1),
      });
      setSceneGroups((prev) =>
        prev.map((g) =>
          g.scene.id === sceneId
            ? { ...g, items: [...g.items, newItem] }
            : g
        )
      );
      setSelectedItemId(newItem.id);
    } catch (err) {
      console.error("添加新分镜条目失败:", err);
    }
  };

  const handleDeleteEpisode = async (episodeId: number) => {
    if (!confirm("确定要删除该分镜集吗？相关的分镜内容也将被删除。")) return false;
    try {
      await storyboardApi.deleteEpisode(episodeId);
      if (
        sidebarSelection.episodeId === episodeId
      ) {
        setSidebarSelection({ type: "all" });
      }
      loadStoryboard();
      return true;
    } catch (err) {
      console.error("删除分集失败:", err);
      return false;
    }
  };

  const handleDeleteScene = async (sceneId: number, episodeId: number) => {
    if (!confirm("确定要删除该分镜场次吗？")) return false;
    try {
      await storyboardApi.deleteScene(sceneId);
      if (sidebarSelection.sceneId === sceneId) {
        setSidebarSelection({ type: "episode", episodeId });
      }
      loadSceneGroups(episodeId);
      return true;
    } catch (err) {
      console.error("删除分镜头报错", err);
      return false;
    }
  };

  const handleReorderScenes = async (episodeId: number, sortedScenes: import("@/lib/api/storyboard").StoryboardScene[]) => {
    try {
      await Promise.all(
        sortedScenes.map((scene, idx) =>
          storyboardApi.updateScene({ id: scene.id, sortOrder: idx })
        )
      );
      if (
        sidebarSelection.type === "all" ||
        sidebarSelection.episodeId === episodeId
      ) {
        loadSceneGroups(sidebarSelection.type === "all" ? undefined : episodeId);
      }
    } catch (err) {
      console.error("更新排序失败", err);
      throw err;
    }
  };

  const handleDeleteItem = async (itemId: number) => {
    if (!confirm("确定要删除该镜头吗？")) return;
    try {
      await storyboardApi.deleteItem(itemId);
      setSceneGroups((prev) =>
        prev.map((g) => ({
          ...g,
          items: g.items.filter((i) => i.id !== itemId),
        }))
      );
      if (selectedItemId === itemId) setSelectedItemId(null);
    } catch (err) {
      console.error("删除条目失败:", err);
    }
  };

  const handleUpdateItemField = async (
    itemId: number,
    field: string,
    value: string
  ) => {
    try {
      const updated = await storyboardApi.updateItem({
        id: itemId,
        [field]: field === "duration" ? (value ? Number(value) : null) : value,
      });
      setSceneGroups((prev) =>
        prev.map((g) => ({
          ...g,
          items: g.items.map((i) => (i.id === updated.id ? updated : i)),
        }))
      );
    } catch (err) {
      console.error("更新条目失败:", err);
    }
  };

  // 拖拽排序
  const handleReorderItems = async (
    sceneId: number,
    reorderedItems: import("@/lib/api/storyboard").StoryboardItem[]
  ) => {
    // 乐观更新本地状态
    setSceneGroups((prev) =>
      prev.map((g) =>
        g.scene.id === sceneId ? { ...g, items: reorderedItems } : g
      )
    );
    // 后台批量更新 sortOrder
    try {
      await storyboardApi.batchUpdateItemSort(
        reorderedItems.map((item) => item.id)
      );
    } catch (err) {
      console.error("更新排序失败:", err);
    }
  };

  /** 单个镜头生成视频 */
  const handleVideoGen = useCallback(
    (itemId: number) => {
      if (!storyboard) return;
      const addPipeline = usePipelineStore.getState().addPipeline;
      const setNotificationOpen =
        usePipelineStore.getState().setNotificationOpen;

      addPipeline({
        label: `生成视频 (镜头 #${itemId})`,
        projectId,
        request: {
          agentType: "storyboard_video_gen",
          projectId,
          context: {
            selectedStoryboardItemIds: [itemId],
            storyboardId: storyboard.id,
          },
        },
      });
      setNotificationOpen(true);
    },
    [projectId, storyboard]
  );

  // ========== 派生数据 ==========

  const allItems = sceneGroups.flatMap((g) => g.items);
  const selectedItem = selectedItemId
    ? allItems.find((i) => i.id === selectedItemId) || null
    : null;

  // 当前激活场次的分组（用于右侧面板展示场次资产）
  const activeSceneGroup = activeSceneId
    ? sceneGroups.find((g) => g.scene.id === activeSceneId) || null
    : null;

  // ========== 渲染 ==========

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // 空状态：没有分镜
  if (!storyboard) {
    return (
      <>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col items-center justify-center py-20"
        >
          <div className="h-20 w-20 rounded-2xl bg-linear-to-br from-cyan-500/10 via-blue-500/10 to-indigo-500/10 flex items-center justify-center mb-6 border border-cyan-500/10">
            <Film className="h-10 w-10 text-cyan-400/60" />
          </div>
          <h2 className="text-xl font-semibold mb-2">还没有分镜</h2>
          <p className="text-muted-foreground text-sm mb-6 max-w-md text-center">
            手动创建分镜表并逐条添加镜头，或使用 AI 根据剧本自动生成分镜
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowCreateDialog(true)}
              className={cn(
                "flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium",
                "bg-primary text-primary-foreground",
                "hover:opacity-90 hover:scale-[1.02]",
                "active:scale-[0.98] transition-all duration-200"
              )}
            >
              <Plus className="h-4 w-4" />
              手动创建
            </button>
            <button
              onClick={() => alert("AI 分镜功能开发中，敬请期待")}
              className={cn(
                "flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium",
                "bg-linear-to-r from-cyan-600 to-blue-600",
                "text-white shadow-lg shadow-cyan-500/20",
                "hover:shadow-cyan-500/30 hover:scale-[1.02]",
                "active:scale-[0.98] transition-all duration-200"
              )}
            >
              <Sparkles className="h-4 w-4" />
              AI 生成分镜
            </button>
          </div>
        </motion.div>
        <CreateStoryboardDialog
          open={showCreateDialog}
          projectId={projectId}
          onClose={() => setShowCreateDialog(false)}
          onCreated={loadStoryboard}
        />
      </>
    );
  }

  // 有分镜：三栏布局
  return (
    <motion.div
      variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } } }}
      initial="hidden"
      animate="visible"
      className="flex h-full rounded-xl border border-border/20 overflow-hidden bg-card/10"
    >
      {/* 左栏：分镜目录 */}
      <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } } }} className="shrink-0 hidden xl:block">
      <StoryboardSidebar
        storyboardId={storyboard.id}
        selection={sidebarSelection}
        activeSceneId={activeSceneId}
        onSelect={setSidebarSelection}
        onInitialLoad={handleSidebarInitialLoad}
        onDeleteEpisode={handleDeleteEpisode}
        onDeleteScene={handleDeleteScene}
        onReorderScenes={handleReorderScenes}
      />
      </motion.div>

      {/* 中栏：按场次分组的分镜内容 */}
      <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } } }} className="flex-1 flex flex-col min-w-0">
        {/* 工具栏 */}
        <div className="px-4 md:px-5 py-3 border-b border-border/20 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2 max-w-[60%]">
            <Sheet open={leftSheetOpen} onOpenChange={setLeftSheetOpen}>
              <SheetTrigger
                render={
                  <button className="xl:hidden p-1.5 -ml-1.5 rounded-md hover:bg-muted text-muted-foreground transition-colors shrink-0">
                    <Menu className="h-5 w-5" />
                  </button>
                }
              />
              <SheetContent side="left" className="w-[300px] p-0 border-r-0 flex flex-col pt-12">
                <StoryboardSidebar
                  storyboardId={storyboard.id}
                  selection={sidebarSelection}
                  activeSceneId={activeSceneId}
                  onSelect={setSidebarSelection}
                  onInitialLoad={handleSidebarInitialLoad}
                  onDeleteEpisode={handleDeleteEpisode}
                  onDeleteScene={handleDeleteScene}
                  onReorderScenes={handleReorderScenes}
                />
              </SheetContent>
            </Sheet>
            <h2 className="text-base font-semibold flex items-center gap-2 overflow-hidden">
              <Film className="h-4 w-4 text-primary shrink-0" />
              <span className="truncate">{storyboard.title || "分镜表"}</span>
              <span className="hidden sm:inline text-xs text-muted-foreground font-normal ml-1 shrink-0">
                · {sceneGroups.length} 场次 · {allItems.length} 镜头
              </span>
            </h2>
          </div>
          <div className="flex items-center gap-2">
            {/* 视图切换 */}
            <div className="flex items-center rounded-lg border border-border/30 bg-muted/20 p-0.5 shrink-0">
              <button
                onClick={() => handleSetViewMode("table")}
                className={cn(
                  "hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all",
                  viewMode === "table"
                    ? "bg-background shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                title="表格视图"
              >
                <Table2 className="h-3.5 w-3.5" />
                表格
              </button>
              <button
                onClick={() => handleSetViewMode("table")}
                className={cn(
                  "flex sm:hidden items-center justify-center w-8 h-8 rounded-md transition-all",
                  viewMode === "table"
                    ? "bg-background shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                title="表格视图"
              >
                <Table2 className="h-4 w-4" />
              </button>
              <button
                onClick={() => handleSetViewMode("card")}
                className={cn(
                  "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all hidden sm:flex",
                  viewMode === "card"
                    ? "bg-background shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                title="卡片视图"
              >
                <LayoutGrid className="h-3.5 w-3.5" />
                卡片
              </button>
              <button
                onClick={() => handleSetViewMode("card")}
                className={cn(
                  "flex sm:hidden items-center justify-center w-8 h-8 rounded-md transition-all",
                  viewMode === "card"
                    ? "bg-background shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
                title="卡片视图"
              >
                <LayoutGrid className="h-4 w-4" />
              </button>
            </div>
            {/* 右侧边栏触发器 */}
            <Sheet open={rightSheetOpen} onOpenChange={setRightSheetOpen}>
              <SheetTrigger
                render={
                  <button className="2xl:hidden p-1.5 -mr-1.5 rounded-md hover:bg-muted text-muted-foreground transition-colors shrink-0">
                    <Info className="h-5 w-5" />
                  </button>
                }
              />
              <SheetContent side="right" className="w-[300px] p-0 border-l-0 flex flex-col pt-12 overflow-y-auto">
                <StoryboardRefPanel
                  storyboard={storyboard}
                  items={allItems}
                  selectedItem={selectedItem}
                  activeSceneGroup={activeSceneGroup}
                  projectId={projectId}
                />
              </SheetContent>
            </Sheet>
          </div>
        </div>

        {/* 内容区域 - 按场次滚动 */}
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
                暂无场次，请在左侧目录创建分镜集和场次
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
                className="scroll-mt-4"
              >
                {/* 场次标题 */}
                <div className="flex items-center gap-2 mb-3">
                  <Camera className="h-3.5 w-3.5 text-primary/60" />
                  <h3 className="text-sm font-semibold">
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
                    onSelectItem={setSelectedItemId}
                    onUpdateItemField={handleUpdateItemField}
                    onAddItem={() =>
                      handleAddItem(scene.id, scene.episodeId)
                    }
                    onDeleteItem={handleDeleteItem}
                    onReorderItems={(reordered) =>
                      handleReorderItems(scene.id, reordered)
                    }
                    onVideoGen={handleVideoGen}
                  />
                ) : (
                  <StoryboardCardView
                    items={items}
                    selectedItemId={selectedItemId}
                    onSelectItem={setSelectedItemId}
                    onAddItem={() =>
                      handleAddItem(scene.id, scene.episodeId)
                    }
                    onReorderItems={(reordered) =>
                      handleReorderItems(scene.id, reordered)
                    }
                    onVideoGen={handleVideoGen}
                  />
                )}
              </div>
            ))
          )}
        </div>
      </motion.div>

      {/* 右栏：引用信息 */}
      <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } } }} className="shrink-0 hidden 2xl:block">
      <StoryboardRefPanel
        storyboard={storyboard}
        items={allItems}
        selectedItem={selectedItem}
        activeSceneGroup={activeSceneGroup}
        projectId={projectId}
      />
      </motion.div>
    </motion.div>
  );
}
