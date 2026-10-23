"use client";

import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { useEffect, useState, useCallback, useRef } from "react";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { toastApiError } from "@/lib/api/toast-api-error";
import { scriptApi, type ScriptEpisode } from "@/lib/api/script";
import { storyboardApi, type Storyboard, type StoryboardEpisode, type StoryboardFrameType, type StoryboardItem, type StoryboardScene } from "@/lib/api/storyboard";
import { assetApi } from "@/lib/api/asset";
import { useProject } from "../../project-context";
import { useStoryboardGeneration } from "./use-storyboard-generation";
import { useStoryboardInvalidation } from "./use-storyboard-invalidation";
import { reconcileStoryboardGroups } from "../storyboard-data";
import { SIDEBAR_COLLAPSED_STORAGE_KEY, type ViewMode, type SidebarSelection, type SceneWithItems } from "../storyboard-types";

export function useStoryboardWorkspace(projectId: number) {
  const composeRequests = useCommandRequestKey();
  const { project } = useProject();
  const requests = useCommandRequestKey();
  const { confirm } = useConfirm();
  const attachTaskStream = usePipelineStore((state) => state.attachTaskStream);
  const setNotificationOpen = usePipelineStore((state) => state.setNotificationOpen);

  const [loading, setLoading] = useState(true);
  const [isRefreshingStoryboard, setIsRefreshingStoryboard] = useState(false);
  const [showAiConfirmDialog, setShowAiConfirmDialog] = useState(false);
  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
  const [scriptEpisodes, setScriptEpisodes] = useState<ScriptEpisode[]>([]);

  // 关联资产状态
  const [assetsList, setAssetsList] = useState<import("@/lib/api/asset").AssetWithItems[]>([]);
  const [assetLookup, setAssetLookup] = useState<Record<number, { item: import("@/lib/api/asset").AssetItem; asset: import("@/lib/api/asset").Asset }>>({});
  const [editAssetsOpen, setEditAssetsOpen] = useState(false);
  const [editingItem, setEditingItem] = useState<StoryboardItem | null>(null);

  const loadProjectAssets = useCallback(async () => {
    try {
      const list = await assetApi.listWithItems(projectId);
      setAssetsList(list);

      const lookup: Record<number, { item: import("@/lib/api/asset").AssetItem; asset: import("@/lib/api/asset").Asset }> = {};
      list.forEach((asset) => {
        if (asset.items && Array.isArray(asset.items)) {
          asset.items.forEach((item) => {
            lookup[item.id] = { item, asset };
          });
        }
      });
      setAssetLookup(lookup);
    } catch (err) {
      console.error("加载资产失败:", err);
      toastApiError(err, "加载资产失败");
    }
  }, [projectId]);

  useEffect(() => {
    loadProjectAssets();
  }, [loadProjectAssets]);

  // 视图状态
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // 加载本地用户偏好
  useEffect(() => {
    const savedMode = localStorage.getItem("fusion-storyboard-view-mode");
    if (savedMode === "table" || savedMode === "card") {
      setViewMode(savedMode);
    }
    setIsSidebarCollapsed(
      localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === "true"
    );
  }, []);

  const handleSetViewMode = useCallback((mode: ViewMode) => {
    setViewMode(mode);
    localStorage.setItem("fusion-storyboard-view-mode", mode);
  }, []);
  const handleSetSidebarCollapsed = useCallback((collapsed: boolean) => {
    setIsSidebarCollapsed(collapsed);
    localStorage.setItem(
      SIDEBAR_COLLAPSED_STORAGE_KEY,
      collapsed ? "true" : "false"
    );
  }, []);
  const [selectedItemId, setSelectedItemId] = useState<number | null>(null);
  const [frameDialogItemId, setFrameDialogItemId] = useState<number | null>(null);
  const [frameDialogInitialType, setFrameDialogInitialType] =
    useState<StoryboardFrameType>("first");
  const [sidebarSelection, setSidebarSelection] = useState<SidebarSelection>({
    type: "episode",
  });
  const sidebarSelectionRef = useRef(sidebarSelection);
  useEffect(() => {
    sidebarSelectionRef.current = sidebarSelection;
  }, [sidebarSelection]);

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
  const sceneGroupsRef = useRef(sceneGroups);
  const [loadingScenes, setLoadingScenes] = useState(false);
  const sceneRequestRef = useRef(0);

  useEffect(() => {
    sceneGroupsRef.current = sceneGroups;
  }, [sceneGroups]);

  // 当前选中集的合成状态
  const [currentEpisode, setCurrentEpisode] = useState<StoryboardEpisode | null>(null);

  // 当前选中的 episodeId（episode 或 scene 选择都会有）
  const currentEpisodeId =
    sidebarSelection.type === "episode" || sidebarSelection.type === "scene"
      ? sidebarSelection.episodeId ?? null
      : null;

  const currentEpisodeIdRef = useRef(currentEpisodeId);
  useEffect(() => {
    currentEpisodeIdRef.current = currentEpisodeId;
  }, [currentEpisodeId]);

  // 拉取当前集详情（含合成状态）
  const refreshCurrentEpisode = useCallback(async () => {
    if (!currentEpisodeId) {
      setCurrentEpisode(null);
      return;
    }
    try {
      const ep = await storyboardApi.getEpisode(currentEpisodeId);
      if (currentEpisodeIdRef.current === currentEpisodeId) setCurrentEpisode(ep);
    } catch (err) {
      console.error("加载集详情失败:", err);
      toastApiError(err, "加载集详情失败");
    }
  }, [currentEpisodeId]);

  const [composedPreviewUrl, setComposedPreviewUrl] = useState<string | null>(null);
  const [submittingComposeEpisodeIds, setSubmittingComposeEpisodeIds] = useState<number[]>([]);

  // 滚动定位 refs
  const sceneRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // 滚动时当前可见的场次 ID（用于侧边栏高亮）
  const [activeSceneId, setActiveSceneId] = useState<number | null>(null);
  // 标记是否由用户点击触发的滚动（此时不要通过 observer 覆盖）
  const isUserScrollRef = useRef(false);

  // ========== 派生数据 ==========

  const allItems = sceneGroups.flatMap((g) => g.items);
  const selectedItem = selectedItemId
    ? allItems.find((i) => i.id === selectedItemId) || null
    : null;
  const frameDialogItem = frameDialogItemId
    ? allItems.find((i) => i.id === frameDialogItemId) || null
    : null;

  // 当前激活场次的分组（用于右侧面板展示场次资产）
  const activeSceneGroup = activeSceneId
    ? sceneGroups.find((g) => g.scene.id === activeSceneId) || null
    : null;

  // 处理镜头选择，同时静默同步定位该镜头所属的场次
  const handleSelectItem = useCallback((itemId: number | null) => {
    setSelectedItemId(itemId);
    if (itemId) {
      const group = sceneGroups.find((g) => g.items.some((item) => item.id === itemId));
      if (group) {
        setActiveSceneId(group.scene.id);
      }
    }
  }, [sceneGroups]);

  // 读取分镜和场次数据时保持当前工作区状态，避免后台任务完成后重置滚动位置。
  const refreshStoryboardData = useCallback(async () => {
    const request = ++sceneRequestRef.current;
    const requestedSelection = sidebarSelectionRef.current;

    try {
      void loadProjectAssets();
      void refreshCurrentEpisode();

      const loadedStoryboard = await storyboardApi.getByProject(projectId);
      if (!loadedStoryboard) {
        setStoryboard(null);
        setScriptEpisodes([]);
        setSceneGroups([]);
        return;
      }

      setStoryboard(loadedStoryboard);
      if (loadedStoryboard.scriptId) {
        const episodes = await scriptApi.listEpisodes(loadedStoryboard.scriptId);
        setScriptEpisodes(episodes);
      } else {
        setScriptEpisodes([]);
      }

      const selection = requestedSelection;
      let scenes: StoryboardScene[];
      if (selection.type === "episode" || selection.type === "scene") {
        scenes = selection.episodeId
          ? await storyboardApi.listScenesByEpisode(selection.episodeId)
          : [];
      } else {
        scenes = await storyboardApi.listScenesByStoryboard(loadedStoryboard.id);
      }

      const groups = await Promise.all(
        scenes.map(async (scene) => {
          const items = await storyboardApi.listItemsByScene(scene.id);
          return { scene, items };
        })
      );
      if (request !== sceneRequestRef.current
          || requestedSelection.episodeId !== sidebarSelectionRef.current.episodeId
          || (requestedSelection.type === "all") !== (sidebarSelectionRef.current.type === "all")) return;
      setSceneGroups((previous) => reconcileStoryboardGroups(previous, groups));

    } catch (err) {
      console.error("完整刷新分镜页数据失败:", err);
      toastApiError(err, "刷新分镜页数据失败");
    } finally {
      if (request === sceneRequestRef.current) setLoadingScenes(false);
    }
  }, [projectId, loadProjectAssets, refreshCurrentEpisode]);

  // 加载分镜
  const loadStoryboard = useCallback(async () => {
    try {
      setLoading(true);
      const activeStoryboard = await storyboardApi.getByProject(projectId);
      if (activeStoryboard) {
        setStoryboard(activeStoryboard);
        if (activeStoryboard.scriptId) {
          const episodes = await scriptApi.listEpisodes(activeStoryboard.scriptId);
          setScriptEpisodes(episodes);
        } else {
          setScriptEpisodes([]);
        }
      } else {
        setStoryboard(null);
        setScriptEpisodes([]);
        setSceneGroups([]);
      }
    } catch (err) {
      console.error("加载分镜失败:", err);
      toastApiError(err, "加载分镜失败");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadStoryboard();
  }, [loadStoryboard]);

  

  const handleManualRefreshStoryboard = useCallback(async () => {
    setIsRefreshingStoryboard(true);
    try {
      await refreshStoryboardData();
    } finally {
      setIsRefreshingStoryboard(false);
    }
  }, [refreshStoryboardData]);

  const handleBindScriptEpisode = useCallback(async (
    episode: StoryboardEpisode,
    scriptEpisodeId: number
  ) => {
    const payload = { id: episode.id, version: episode.version, scriptEpisodeId };
    const updated = await storyboardApi.bindScriptEpisode(episode.id, episode.version, scriptEpisodeId, requests.forPayload(payload));
    requests.clear();
    await refreshStoryboardData();
    return updated;
  }, [refreshStoryboardData, requests]);

  

  useStoryboardInvalidation(projectId, refreshStoryboardData);

  // 加载场次分组数据
  const storyboardId = storyboard?.id;
  const loadSceneGroups = useCallback(
    async (episodeId?: number) => {
      if (!storyboardId) return;
      const request = ++sceneRequestRef.current;
      setLoadingScenes(true);
      try {
        let scenes: StoryboardScene[];
        if (episodeId) {
          scenes = await storyboardApi.listScenesByEpisode(episodeId);
        } else {
          scenes = await storyboardApi.listScenesByStoryboard(storyboardId);
        }

        // 并行加载每个场次的条目
        const groups = await Promise.all(
          scenes.map(async (scene) => {
            const items = await storyboardApi.listItemsByScene(scene.id);
            return { scene, items };
          })
        );

        if (request !== sceneRequestRef.current) return;
        setSceneGroups((previous) => reconcileStoryboardGroups(previous, groups));
      } catch (err) {
        console.error("加载场次失败:", err);
        toastApiError(err, "加载场次失败");
      } finally {
        if (request === sceneRequestRef.current) setLoadingScenes(false);
      }
    },
    [storyboardId]
  );

  // 追踪当前已加载的集ID，避免同集内切换场次重复加载
  const loadedEpisodeIdRef = useRef<number | null>(null);

  // sidebar 初始化完成后通知 page 第一集 episodeId
  const handleSidebarInitialLoad = useCallback((firstEpisodeId: number) => {
    setSidebarSelection({ type: "episode", episodeId: firstEpisodeId });
  }, []);

  // 当侧边栏选择变化时加载数据
  useEffect(() => {
    if (!storyboardId) return;

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
      const sceneExists = sceneGroupsRef.current.some(
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
  }, [sidebarSelection, storyboardId, loadSceneGroups]);

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

  // ========== 滚动监听：更新当前可视场次 ==========
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || sceneGroups.length === 0) return;

    let ticking = false;

    const handleScroll = () => {
      if (isUserScrollRef.current) return;

      if (!ticking) {
        window.requestAnimationFrame(() => {
          const scrollTop = container.scrollTop;
          const scrollHeight = container.scrollHeight;
          const clientHeight = container.clientHeight;

          // 1. 如果已滚动到最顶部，直接激活第一个场次
          if (scrollTop === 0) {
            setActiveSceneId(sceneGroups[0].scene.id);
            ticking = false;
            return;
          }

          // 2. 如果已滚动到最底部（解决短场次或大屏幕下，最末尾场次无法卷到顶部触发激活线的问题）
          if (scrollTop + clientHeight >= scrollHeight - 15) {
            setActiveSceneId(sceneGroups[sceneGroups.length - 1].scene.id);
            ticking = false;
            return;
          }

          // 3. 普通滚动过程中，使用较为灵敏的激活线（容器高度的 35%，最大不超过 300px）
          const containerRect = container.getBoundingClientRect();
          let activeId: number | null = null;
          let minDiff = Infinity;
          const triggerY = Math.min(300, containerRect.height * 0.35);

          for (const { scene } of sceneGroups) {
            const el = sceneRefs.current[scene.id];
            if (!el) continue;
            const rect = el.getBoundingClientRect();
            const relativeTop = rect.top - containerRect.top;
            const relativeBottom = rect.bottom - containerRect.top;

            // 判断该场次是否跨越容器顶部的激活线
            if (relativeTop <= triggerY && relativeBottom > triggerY) {
              activeId = scene.id;
              break;
            }

            // 备选：如果没有跨越激活线的，记录离激活线最近的一个
            const diff = Math.abs(relativeTop - triggerY);
            if (diff < minDiff) {
              minDiff = diff;
              activeId = scene.id;
            }
          }

          if (activeId !== null) {
            setActiveSceneId(activeId);
          }
          ticking = false;
        });

        ticking = true;
      }
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    // 初始化执行一次
    handleScroll();

    return () => {
      container.removeEventListener("scroll", handleScroll);
    };
  }, [sceneGroups]);

  // ========== 操作 ==========

  const handleAddItem = async (sceneId: number) => {
    if (!storyboard) return;
    const group = sceneGroups.find((g) => g.scene.id === sceneId);
    if (!group) return;
    const currentItems = group.items;
    try {
      const payload = {
        storyboardSceneId: sceneId,
        parentVersion: group.scene.version,
        sortOrder: currentItems.length,
        shotNumber: String(currentItems.length + 1),
      };
      const newItem = await storyboardApi.createItem(payload, requests.forPayload(payload));
      requests.clear();
      await refreshStoryboardData();
      setSelectedItemId(newItem.id);
    } catch (err) {
      console.error("添加新分镜条目失败:", err);
      toastApiError(err, "添加新分镜条目失败");
    }
  };

  const handleDeleteEpisode = async (episode: StoryboardEpisode) => {
    const episodeId = episode.id;
    const ok = await confirm({ title: "删除分镜集", description: "确定要删除该分镜集吗？相关的分镜内容也将被一并删除，已有引用会显示失效，媒体文件会保留。", variant: "destructive", confirmText: "确定删除" }); if (!ok) return false;
    try {
      await storyboardApi.deleteEpisode(episodeId, episode.version, requests.forPayload({ deleteEpisode: episodeId, version: episode.version }));
      requests.clear();
      if (
        sidebarSelection.episodeId === episodeId
      ) {
        setSidebarSelection({ type: "all" });
      }
      await refreshStoryboardData();
      return true;
    } catch (err) {
      console.error("删除分集失败:", err);
      toastApiError(err, "删除分集失败");
      return false;
    }
  };

  const handleDeleteScene = async (scene: StoryboardScene) => {
    const sceneId = scene.id;
    const episodeId = scene.episodeId;
    const ok = await confirm({ title: "删除分镜场次", description: "确定要删除该分镜场次吗？场次内的所有镜头将被一并删除。", variant: "destructive", confirmText: "确定删除" }); if (!ok) return false;
    try {
      await storyboardApi.deleteScene(sceneId, scene.version, requests.forPayload({ deleteScene: sceneId, version: scene.version }));
      requests.clear();
      if (sidebarSelection.sceneId === sceneId) {
        setSidebarSelection({ type: "episode", episodeId });
      }
      await refreshStoryboardData();
      return true;
    } catch (err) {
      console.error("删除分镜头报错", err);
      toastApiError(err, "删除分镜头失败");
      return false;
    }
  };

  const handleReorderScenes = async (episode: StoryboardEpisode, sortedScenes: StoryboardScene[]) => {
    const ids = sortedScenes.map(scene => scene.id);
    const payload = { reorderScenes: episode.id, version: episode.version, ids };
    await storyboardApi.reorderScenes(episode.id, episode.version, ids, requests.forPayload(payload));
    requests.clear();
    await refreshStoryboardData();
  };

  const handleDeleteItem = async (itemId: number) => {
    const item = sceneGroups.flatMap(group => group.items).find(candidate => candidate.id === itemId);
    if (!item) return;
    const ok = await confirm({ title: "删除镜头", description: "确定要删除该镜头吗？已有引用会显示失效，媒体文件会保留。", variant: "destructive", confirmText: "确定删除" }); if (!ok) return;
    try {
      await storyboardApi.deleteItem(itemId, item.version, requests.forPayload({ deleteItem: itemId, version: item.version }));
      requests.clear();
      await refreshStoryboardData();
      setSceneGroups((prev) =>
        prev.map((g) => ({
          ...g,
          items: g.items.filter((i) => i.id !== itemId),
        }))
      );
      if (selectedItemId === itemId) setSelectedItemId(null);
      if (frameDialogItemId === itemId) setFrameDialogItemId(null);
    } catch (err) {
      console.error("删除条目失败:", err);
      toastApiError(err, "删除条目失败");
    }
  };

  const handleUpdateItemField = async (
    item: StoryboardItem,
    field: string,
    value: string | number | null
  ) => {
    try {
      const payload = {
        id: item.id,
        version: item.version,
        [field]: field === "duration" ? (value === "" || value === null ? null : Number(value)) : value,
      };
      const updated = await storyboardApi.updateItem(payload, requests.forPayload(payload));
      requests.clear();
      setSceneGroups((prev) =>
        prev.map((g) => ({
          ...g,
          items: g.items.map((i) => (i.id === updated.id ? updated : i)),
        }))
      );
      await refreshStoryboardData();
    } catch (err) {
      console.error("更新条目失败:", err);
      throw err;
    }
  };

  const updateItemInSceneGroups = useCallback((updated: StoryboardItem) => {
    setSceneGroups((prev) =>
      prev.map((g) => ({
        ...g,
        items: g.items.map((i) => (i.id === updated.id ? updated : i)),
      }))
    );
  }, []);

  const refreshStoryboardItems = useCallback(async (itemIds: number[]) => {
    if (itemIds.length === 0) return;
    try {
      const updatedItems = await Promise.all(
        itemIds.map((itemId) => storyboardApi.getItem(itemId))
      );
      const updates = new Map(updatedItems.map((item) => [item.id, item]));
      setSceneGroups((prev) =>
        prev.map((group) => {
          let changed = false;
          const items = group.items.map((item) => {
            const updated = updates.get(item.id);
            if (!updated) return item;
            changed = true;
            return updated;
          });
          return changed ? { ...group, items } : group;
        })
      );
    } catch (err) {
      console.error("更新生成后的分镜条目失败:", err);
      toastApiError(err, "更新生成后的分镜条目失败");
    }
  }, []);

  /** 手动更新镜头首尾帧 */
  const handleUpdateItemFrame = useCallback(
    async (item: StoryboardItem, frameType: StoryboardFrameType, imageUrl: string | null) => {
      try {
        const payload = {
          version: item.version,
          frameType,
          imageUrl,
          prompt: null,
        };
        const updated = await storyboardApi.updateFrame(item.id, payload, requests.forPayload({ itemId: item.id, ...payload }));
        requests.clear();
        updateItemInSceneGroups(updated);
        await refreshStoryboardData();
      } catch (err) {
        console.error("更新镜头首尾帧失败:", err);
        throw err;
      }
    },
    [updateItemInSceneGroups, refreshStoryboardData, requests]
  );

  /** 提交镜头首尾帧 AI 生成任务 */
  

  /** 批量提交当前场次的首尾帧 AI 生成任务 */
  

  /** 打开单个镜头首尾帧编辑弹窗 */
  const handleOpenFrameDialog = useCallback(
    (item: StoryboardItem, frameType: StoryboardFrameType) => {
      setSelectedItemId(item.id);
      setFrameDialogItemId(item.id);
      setFrameDialogInitialType(frameType);
    },
    []
  );

  /** 关闭单个镜头首尾帧编辑弹窗 */
  const handleCloseFrameDialog = useCallback(() => {
    setFrameDialogItemId(null);
  }, []);

  // 业务顺序一次原子提交；失败时保留已确认的界面顺序。
  const handleReorderItems = async (sceneId: number, reorderedItems: StoryboardItem[]) => {
    const group = sceneGroups.find(entry => entry.scene.id === sceneId);
    if (!group) return;
    const ids = reorderedItems.map(item => item.id);
    try {
      const result = await storyboardApi.reorderItems(sceneId, group.scene.version, ids,
        requests.forPayload({ reorderItems: sceneId, version: group.scene.version, ids }));
      requests.clear();
      setSceneGroups(current => current.map(entry => entry.scene.id === sceneId ? { scene: result.parent, items: result.children } : entry));
      await refreshStoryboardData();
    } catch (err) { toastApiError(err, "更新排序失败"); }
  };

  useEffect(() => {
    refreshCurrentEpisode();
  }, [refreshCurrentEpisode]);

  /** 提交本集合成视频任务 */
  const handleComposeEpisodeVideo = useCallback(async (includeBgm: boolean) => {
    if (!currentEpisodeId || !currentEpisode) return;
    if (submittingComposeEpisodeIds.includes(currentEpisodeId)) return;

    const epLabel = currentEpisode.title?.trim()
      || (currentEpisode.episodeNumber != null ? `第 ${currentEpisode.episodeNumber} 集` : `集 ${currentEpisode.id}`);
    setSubmittingComposeEpisodeIds((prev) =>
      prev.includes(currentEpisodeId) ? prev : [...prev, currentEpisodeId]
    );
    setNotificationOpen(true);
    try {
      const taskId = await storyboardApi.composeEpisodeVideo(currentEpisodeId, currentEpisode.version, composeRequests.forPayload({ episodeId: currentEpisodeId, version: currentEpisode.version, includeBgm }), includeBgm);
      composeRequests.clear();
      setSubmittingComposeEpisodeIds((prev) =>
        prev.filter((id) => id !== currentEpisodeId)
      );

      attachTaskStream({
        label: `合成本集视频：${epLabel}`,
        projectId,
        taskId,
        cancellable: false,
        onSettled: () => {
          void refreshCurrentEpisode();
        },
      });

      void refreshCurrentEpisode();
    } catch (err) {
      console.error("提交合成任务失败:", err);
      toastApiError(err, "提交合成任务失败");
      setSubmittingComposeEpisodeIds((prev) =>
        prev.filter((id) => id !== currentEpisodeId)
      );
    }
  }, [
    currentEpisodeId,
    currentEpisode,
    submittingComposeEpisodeIds,
    projectId,
    attachTaskStream,
    composeRequests,
    setNotificationOpen,
    refreshCurrentEpisode,
  ]);

  /** 单个镜头生成视频 */
  

  const { handleAiStoryboard, handleGenerateEpisodeStoryboard, handleGenerateItemFrame, handleBatchGenerateSceneFrames, handleVideoGen } = useStoryboardGeneration({
    projectId, projectName: project?.name, storyboard, scriptEpisodes, currentEpisode, sceneGroups,
    refreshStoryboardItems,
  });
  return {
    projectId,
    project,
    loading,
    isRefreshingStoryboard,
    showAiConfirmDialog,
    setShowAiConfirmDialog,
    storyboard,
    scriptEpisodes,
    assetsList,
    assetLookup,
    editAssetsOpen,
    setEditAssetsOpen,
    editingItem,
    setEditingItem,
    viewMode,
    isSidebarCollapsed,
    handleSetViewMode,
    handleSetSidebarCollapsed,
    selectedItemId,
    frameDialogItemId,
    frameDialogInitialType,
    sidebarSelection,
    setSidebarSelection,
    leftSheetOpen,
    setLeftSheetOpen,
    rightSheetOpen,
    setRightSheetOpen,
    sceneGroups,
    setSceneGroups,
    loadingScenes,
    currentEpisode,
    currentEpisodeId,
    composedPreviewUrl,
    setComposedPreviewUrl,
    runningComposeEpisodeIds: currentEpisode?.composeStatus === 1 ? [currentEpisode.id] : [],
    submittingComposeEpisodeIds,
    sceneRefs,
    scrollContainerRef,
    activeSceneId,
    setActiveSceneId,
    allItems,
    selectedItem,
    frameDialogItem,
    activeSceneGroup,
    handleSelectItem,
    handleManualRefreshStoryboard,
    handleBindScriptEpisode,
    handleSidebarInitialLoad,
    handleAddItem,
    handleDeleteEpisode,
    handleDeleteScene,
    handleReorderScenes,
    handleDeleteItem,
    handleUpdateItemField,
    handleUpdateItemFrame,
    handleOpenFrameDialog,
    handleCloseFrameDialog,
    handleReorderItems,
    handleComposeEpisodeVideo,
    handleAiStoryboard,
    handleGenerateEpisodeStoryboard,
    handleGenerateItemFrame,
    handleBatchGenerateSceneFrames,
    handleVideoGen
  };
}
