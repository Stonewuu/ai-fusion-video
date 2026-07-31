"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import { Loader2, Menu, Info } from "lucide-react";
import { motion } from "framer-motion";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  scriptApi,
  type Script,
  type ScriptEpisode,
  type SceneItem,
} from "@/lib/api/script";
import {
  storyboardApi,
  type Storyboard,
  type StoryboardEpisode,
} from "@/lib/api/storyboard";
import { EpisodeTree } from "./_components/episode-tree";
import { SceneList } from "./_components/scene-list";
import { SceneDetail } from "./_components/scene-detail";
import { ScriptOverview } from "./_components/script-overview";
import { EpisodeParseDialog } from "@/components/dashboard/episode-parse-dialog";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useProject } from "../project-context";

const SCRIPT_SIDEBAR_COLLAPSED_STORAGE_KEY = "fusion-script-sidebar-collapsed";

export default function ScriptTabPage() {
  const params = useParams();
  const projectId = Number(params.id);
  const { project } = useProject();

  const [loading, setLoading] = useState(true);
  const [isRefreshingScript, setIsRefreshingScript] = useState(false);
  const [script, setScript] = useState<Script | null>(null);
  const [episodes, setEpisodes] = useState<ScriptEpisode[]>([]);
  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);

  // 当前展示的分集
  const [activeEpisodeId, setActiveEpisodeId] = useState<number | null>(null);
  // 树结构：展开状态
  const [expandedEpisodes, setExpandedEpisodes] = useState<Set<number>>(new Set());
  // 各分集的场次缓存
  const [episodeScenes, setEpisodeScenes] = useState<Record<number, SceneItem[]>>({});
  const [loadingEpisodes, setLoadingEpisodes] = useState<Set<number>>(new Set());

  const activeEpisodeIdRef = useRef(activeEpisodeId);
  useEffect(() => {
    activeEpisodeIdRef.current = activeEpisodeId;
  }, [activeEpisodeId]);

  const expandedEpisodesRef = useRef(expandedEpisodes);
  useEffect(() => {
    expandedEpisodesRef.current = expandedEpisodes;
  }, [expandedEpisodes]);

  const episodeScenesRef = useRef(episodeScenes);
  useEffect(() => {
    episodeScenesRef.current = episodeScenes;
  }, [episodeScenes]);

  // 选中的场次（用于右栏显示详情）
  const [selectedSceneId, setSelectedSceneId] = useState<number | null>(null);

  // 移动端侧边栏状态
  const [leftSheetOpen, setLeftSheetOpen] = useState(false);
  const [rightSheetOpen, setRightSheetOpen] = useState(false);
  const [isScriptSidebarCollapsed, setIsScriptSidebarCollapsed] = useState(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem(SCRIPT_SIDEBAR_COLLAPSED_STORAGE_KEY) === "true";
  });

  // 分集解析弹窗
  const [showEpisodeParseDialog, setShowEpisodeParseDialog] = useState(false);
  const [episodeToParseId, setEpisodeToParseId] = useState<number | null>(null);

  // Pipeline store
  const { addPipeline, setPanelExpanded, setExpandedTaskId } = usePipelineStore();

  const handleSetScriptSidebarCollapsed = useCallback((collapsed: boolean) => {
    setIsScriptSidebarCollapsed(collapsed);
    localStorage.setItem(
      SCRIPT_SIDEBAR_COLLAPSED_STORAGE_KEY,
      collapsed ? "true" : "false"
    );
  }, []);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(min-width: 1280px)");
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

  // ========== 数据加载 ==========

  const loadEpisodeScenes = async (episodeId: number) => {
    if (episodeScenes[episodeId]) return;
    setLoadingEpisodes((prev) => new Set(prev).add(episodeId));
    try {
      const scenes = await scriptApi.listScenes(episodeId);
      setEpisodeScenes((prev) => ({ ...prev, [episodeId]: scenes }));
    } catch (err) {
      console.error("加载场次失败:", err);
      toastApiError(err, "加载场次失败");
    } finally {
      setLoadingEpisodes((prev) => {
        const s = new Set(prev);
        s.delete(episodeId);
        return s;
      });
    }
  };

  const loadStoryboardForScript = useCallback(async (scriptId: number) => {
    const currentStoryboard = await storyboardApi.getByProject(projectId);
    if (!currentStoryboard || currentStoryboard.scriptId !== scriptId) {
      setStoryboard(null);
      return { storyboard: null, episodes: [] as StoryboardEpisode[] };
    }

    setStoryboard(currentStoryboard);
    const currentEpisodes = await storyboardApi.listEpisodes(currentStoryboard.id);
    return { storyboard: currentStoryboard, episodes: currentEpisodes };
  }, [projectId]);

  const loadScript = useCallback(async () => {
    try {
      setLoading(true);
      const scr = await scriptApi.getByProject(projectId);
      if (scr) {
        setScript(scr);
        const eps = await scriptApi.listEpisodes(scr.id);
        setEpisodes(eps);
        await loadStoryboardForScript(scr.id);
        if (eps.length > 0) {
          setActiveEpisodeId(eps[0].id);
          // 默认展开所有分集，并并行加载场次
          setExpandedEpisodes(new Set(eps.map((e) => e.id)));
          await Promise.all(eps.map((e) => loadEpisodeScenes(e.id)));
        }
      } else {
        setScript(null);
        setEpisodes([]);
        setStoryboard(null);
      }
    } catch (err) {
      console.error("加载剧本工作区失败:", err);
      toastApiError(err, "加载剧本工作区失败");
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadStoryboardForScript, projectId]);

  useEffect(() => {
    loadScript();
  }, [loadScript]);

  const refreshScript = useCallback(async () => {
    try {
      const scr = await scriptApi.getByProject(projectId);
      if (scr) {
        setScript(scr);
        const eps = await scriptApi.listEpisodes(scr.id);
        setEpisodes(eps);
        await loadStoryboardForScript(scr.id);
        
        if (eps.length > 0) {
          const prevActiveId = activeEpisodeIdRef.current;
          const stillExists = eps.some((e) => e.id === prevActiveId);
          const nextActiveId = stillExists ? prevActiveId : eps[0].id;
          setActiveEpisodeId(nextActiveId);

          const currentExpanded = expandedEpisodesRef.current;
          const currentScenes = episodeScenesRef.current;
          
          const newEpisodeScenes: Record<number, SceneItem[]> = {};
          
          await Promise.all(
            eps.map(async (e) => {
              const shouldLoad = currentExpanded.has(e.id) || !!currentScenes[e.id] || e.id === nextActiveId;
              if (shouldLoad) {
                try {
                  const scenes = await scriptApi.listScenes(e.id);
                  newEpisodeScenes[e.id] = scenes;
                } catch (err) {
                  console.error(`刷新分集 ${e.id} 场次失败:`, err);
                  toastApiError(err, `刷新分集 ${e.id} 场次失败`);
                }
              }
            })
          );
          
          setEpisodeScenes((prev) => ({ ...prev, ...newEpisodeScenes }));
        }
      } else {
        setScript(null);
        setEpisodes([]);
        setEpisodeScenes({});
        setStoryboard(null);
      }
    } catch (err) {
      console.error("刷新剧本工作区失败:", err);
      toastApiError(err, "刷新剧本工作区失败");
    }
  }, [loadStoryboardForScript, projectId]);

  const handleManualRefreshScript = useCallback(async () => {
    setIsRefreshingScript(true);
    try {
      await refreshScript();
    } finally {
      setIsRefreshingScript(false);
    }
  }, [refreshScript]);

  // AI 工具执行后自动刷新
  const scriptsInvalidation = usePipelineStore((s) => s.invalidation.scripts);
  const scriptsInvRef = useRef(scriptsInvalidation);
  useEffect(() => {
    if (scriptsInvRef.current !== scriptsInvalidation) {
      scriptsInvRef.current = scriptsInvalidation;
      refreshScript();
    }
  }, [scriptsInvalidation, refreshScript]);

  const storyboardsInvalidation = usePipelineStore((s) => s.invalidation.storyboards);
  const storyboardsInvRef = useRef(storyboardsInvalidation);
  useEffect(() => {
    if (storyboardsInvRef.current !== storyboardsInvalidation && script) {
      storyboardsInvRef.current = storyboardsInvalidation;
      loadStoryboardForScript(script.id);
    }
  }, [loadStoryboardForScript, script, storyboardsInvalidation]);

  // ========== 导航操作 ==========

  const toggleEpisode = async (episodeId: number) => {
    if (expandedEpisodes.has(episodeId)) {
      setExpandedEpisodes((prev) => {
        const next = new Set(prev);
        next.delete(episodeId);
        return next;
      });
    } else {
      setExpandedEpisodes((prev) => new Set(prev).add(episodeId));
      await loadEpisodeScenes(episodeId);
    }
  };

  const selectEpisode = async (episodeId: number) => {
    setActiveEpisodeId(episodeId);
    setSelectedSceneId(null);
    if (!expandedEpisodes.has(episodeId)) {
      setExpandedEpisodes((prev) => new Set(prev).add(episodeId));
    }
    setLeftSheetOpen(false);
    await loadEpisodeScenes(episodeId);
  };

  // 追踪用户滚动状态，避免点击滚动时触发 observer 误判
  const isUserScrollRef = useRef(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const scrollToScene = (sceneId: number) => {
    isUserScrollRef.current = true;
    setSelectedSceneId(sceneId);
    const el = document.querySelector(`[data-scene-id="${sceneId}"]`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    // 滚动动画完成后恢复 observer
    setTimeout(() => {
      isUserScrollRef.current = false;
    }, 600);
  };

  const handleSelectScene = (sceneId: number, episodeId: number) => {
    selectEpisode(episodeId);
    setSelectedSceneId(sceneId);
    setLeftSheetOpen(false);
    setTimeout(() => scrollToScene(sceneId), 50);
  };

  // ========== 增删操作 ==========

  const handleAddEpisode = async () => {
    if (!script) return;
    try {
      const ep = await scriptApi.createEpisode({
        scriptId: script.id,
        episodeNumber: episodes.length + 1,
        title: `第 ${episodes.length + 1} 集`,
        sortOrder: episodes.length,
      });
      setEpisodes((prev) => [...prev, ep]);
      setExpandedEpisodes((prev) => new Set(prev).add(ep.id));
      setEpisodeScenes((prev) => ({ ...prev, [ep.id]: [] }));
      setActiveEpisodeId(ep.id);
    } catch (err) {
      console.error("添加分集失败:", err);
      toastApiError(err, "添加分集失败");
    }
  };

  const handleDeleteEpisode = async (episodeId: number) => {
    if (!confirm("确定要删除该分集及其所有场次吗？")) return;
    try {
      await scriptApi.deleteEpisode(episodeId);
      setEpisodes((prev) => prev.filter((ep) => ep.id !== episodeId));
      setExpandedEpisodes((prev) => {
        const next = new Set(prev);
        next.delete(episodeId);
        return next;
      });
      const newScenes = { ...episodeScenes };
      delete newScenes[episodeId];
      setEpisodeScenes(newScenes);
      if (activeEpisodeId === episodeId) {
        setActiveEpisodeId(episodes.length > 1 ? episodes[0].id : null);
      }
    } catch (err) {
      console.error("删除分集失败:", err);
      toastApiError(err, "删除分集失败");
    }
  };

  const handleAddScene = async (episodeId: number) => {
    if (!script) return;
    const scenes = episodeScenes[episodeId] || [];
    const ep = episodes.find((e) => e.id === episodeId);
    const epNum = ep?.episodeNumber || 1;
    try {
      const scene = await scriptApi.createScene({
        episodeId,
        scriptId: script.id,
        sceneNumber: `${epNum}-${scenes.length + 1}`,
        sceneHeading: `新场次`,
        sortOrder: scenes.length,
      });
      setEpisodeScenes((prev) => ({
        ...prev,
        [episodeId]: [...(prev[episodeId] || []), scene],
      }));
      setSelectedSceneId(scene.id);
      setTimeout(() => scrollToScene(scene.id), 100);
    } catch (err) {
      console.error("添加场次失败:", err);
      toastApiError(err, "添加场次失败");
    }
  };

  const handleInsertScene = async (episodeId: number, atIndex: number) => {
    if (!script) return;
    const scenes = episodeScenes[episodeId] || [];
    const ep = episodes.find((e) => e.id === episodeId);
    const epNum = ep?.episodeNumber || 1;
    try {
      const scene = await scriptApi.createScene({
        episodeId,
        scriptId: script.id,
        sceneNumber: `${epNum}-${scenes.length + 1}`,
        sceneHeading: `新场次`,
        sortOrder: atIndex,
      });
      // 将新场次插入本地数组指定位置
      const newScenes = [...scenes];
      newScenes.splice(atIndex, 0, scene);
      // 更新后续场次的 sortOrder
      const updated = newScenes.map((s, idx) => ({ ...s, sortOrder: idx }));
      setEpisodeScenes((prev) => ({ ...prev, [episodeId]: updated }));
      setSelectedSceneId(scene.id);
      setTimeout(() => scrollToScene(scene.id), 100);
      // 批量持久化 sortOrder
      await Promise.all(
        updated
          .filter((s) => s.id !== scene.id)
          .map((s) => scriptApi.updateScene({ id: s.id, sortOrder: s.sortOrder }))
      );
    } catch (err) {
      console.error("插入场次失败:", err);
      toastApiError(err, "插入场次失败");
    }
  };

  const handleDeleteScene = async (sceneId: number, episodeId: number) => {
    if (!confirm("确定要删除该场次吗？")) return;
    try {
      await scriptApi.deleteScene(sceneId);
      setEpisodeScenes((prev) => ({
        ...prev,
        [episodeId]: (prev[episodeId] || []).filter((s) => s.id !== sceneId),
      }));
      if (selectedSceneId === sceneId) {
        setSelectedSceneId(null);
      }
    } catch (err) {
      console.error("删除场次失败:", err);
      toastApiError(err, "删除场次失败");
    }
  };

  const handleSceneUpdated = (episodeId: number, updated: SceneItem) => {
    setEpisodeScenes((prev) => ({
      ...prev,
      [episodeId]: (prev[episodeId] || []).map((s) =>
        s.id === updated.id ? updated : s
      ),
    }));
  };

  const handleReorderScenes = async (episodeId: number, oldIndex: number, newIndex: number) => {
    // 乐观更新 UI
    const prevScenes = episodeScenes[episodeId] || [];
    const scenes = [...prevScenes];
    const [moved] = scenes.splice(oldIndex, 1);
    scenes.splice(newIndex, 0, moved);
    setEpisodeScenes((prev) => ({ ...prev, [episodeId]: scenes }));

    // 持久化到后端：为每个场次更新 sortOrder
    try {
      await Promise.all(
        scenes.map((scene, idx) =>
          scriptApi.updateScene({ id: scene.id, sortOrder: idx })
        )
      );
    } catch (err) {
      console.error("排序保存失败，已回滚:", err);
      toastApiError(err, "排序保存失败，已回滚");
      // 回滚
      setEpisodeScenes((prev) => ({ ...prev, [episodeId]: prevScenes }));
    }
  };

  // ========== 分集 AI 解析 ==========

  const handleOpenEpisodeParse = (episodeId: number) => {
    setEpisodeToParseId(episodeId);
    setShowEpisodeParseDialog(true);
  };

  const handleEpisodeParse = async (rawContent: string) => {
    if (!script || !episodeToParseId) return;
    const ep = episodes.find((e) => e.id === episodeToParseId);
    if (!ep) return;

    try {
      // 1. 先保存原文到分集记录
      await scriptApi.updateEpisode({ id: episodeToParseId, rawContent });

      // 2. 启动 script_episode_parse Pipeline
      const pipelineId = addPipeline({
        label: `AI 解析 · 第 ${ep.episodeNumber} 集`,
        projectId,
        request: {
          agentType: "script_episode_parse",
          toolExecutionMode: "FULL_ACCESS",
          category: "pipeline",
          title: `AI 解析 · 第 ${ep.episodeNumber} 集`,
          projectId,
          context: { episodeId: episodeToParseId, scriptId: script.id },
        },
        onComplete: () => {
          // Pipeline 完成后重新加载该集场次
          reloadEpisodeScenes(episodeToParseId);
        },
      });

      // 3. 打开任务面板
      setPanelExpanded(true);
      setExpandedTaskId(pipelineId);
    } catch (err) {
      console.error("启动分集解析失败:", err);
      toastApiError(err, "启动分集解析失败");
    }
  };

  const ensureStoryboardForScript = async () => {
    if (!script) {
      throw new Error("项目剧本工作区未初始化");
    }
    if (storyboard && storyboard.scriptId === script.id) {
      return storyboard;
    }

    const loaded = await loadStoryboardForScript(script.id);
    if (loaded.storyboard) {
      return loaded.storyboard;
    }

    throw new Error("项目分镜工作区未初始化或关联剧本不一致");
  };

  const handleGenerateEpisodeStoryboard = async (episodeId: number) => {
    if (!script) return;
    const ep = episodes.find((item) => item.id === episodeId);
    if (!ep) return;

    try {
      const targetStoryboard = await ensureStoryboardForScript();
      const currentStoryboardEpisodes = await storyboardApi.listEpisodes(targetStoryboard.id);

      const boundEpisode = currentStoryboardEpisodes.find(
        (item) => item.scriptEpisodeId === episodeId
      );
      const legacyUnboundEpisode = currentStoryboardEpisodes.find(
        (item) => item.scriptEpisodeId == null && item.episodeNumber === ep.episodeNumber
      );

      if (!boundEpisode && legacyUnboundEpisode) {
        alert(`分镜中存在未绑定的第 ${ep.episodeNumber} 集，请先到分镜页将旧分镜集绑定到当前剧本集后再重新生成。`);
        return;
      }

      if (boundEpisode) {
        const confirmed = confirm(`将覆盖第 ${ep.episodeNumber} 集已有分镜内容，不影响其它集。确定继续？`);
        if (!confirmed) return;
        await storyboardApi.clearEpisodeContent(boundEpisode.id);
      }

      const pipelineId = addPipeline({
        label: `AI 分镜 · 第 ${ep.episodeNumber} 集`,
        projectId,
        request: {
          agentType: "episode_storyboard_writer",
          toolExecutionMode: "FULL_ACCESS",
          category: "pipeline",
          title: `AI 分镜 · 第 ${ep.episodeNumber} 集`,
          message: `请为剧本分集（scriptEpisodeId: ${ep.id}）生成分镜，并保存到分镜脚本 ${targetStoryboard.id}。`,
          projectId,
          context: {
            scriptId: script.id,
            storyboardId: targetStoryboard.id,
            scriptEpisodeId: ep.id,
          },
        },
        onComplete: () => {
          loadStoryboardForScript(script.id);
        },
      });

      setPanelExpanded(true);
      setExpandedTaskId(pipelineId);
    } catch (err) {
      console.error("启动单集分镜生成失败:", err);
      toastApiError(err, "启动单集分镜生成失败，请重试");
    }
  };

  /** 重新加载指定分集的场次（强制刷新忽略缓存） */
  const reloadEpisodeScenes = async (episodeId: number) => {
    setLoadingEpisodes((prev) => new Set(prev).add(episodeId));
    try {
      const scenes = await scriptApi.listScenes(episodeId);
      setEpisodeScenes((prev) => ({ ...prev, [episodeId]: scenes }));
    } catch (err) {
      console.error("重新加载场次失败:", err);
      toastApiError(err, "重新加载场次失败");
    } finally {
      setLoadingEpisodes((prev) => {
        const s = new Set(prev);
        s.delete(episodeId);
        return s;
      });
    }
  };

  // ========== 派生数据 ==========

  const selectedScene = selectedSceneId
    ? Object.values(episodeScenes)
        .flat()
        .find((s) => s.id === selectedSceneId) || null
    : null;

  const activeEpisode = episodes.find((ep) => ep.id === activeEpisodeId);
  const activeScenes = activeEpisodeId
    ? episodeScenes[activeEpisodeId] || []
    : [];
  const isActiveLoading = activeEpisodeId
    ? loadingEpisodes.has(activeEpisodeId)
    : false;

  // ========== 滚动监听：IntersectionObserver ==========
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || activeScenes.length === 0) return;

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
          if (id) setSelectedSceneId(id);
        }
      },
      {
        root: container,
        rootMargin: "-10% 0px -70% 0px",
        threshold: 0,
      }
    );

    // 延迟一点绑定，确保 DOM 已经渲染完毕
    const timer = setTimeout(() => {
      const elements = container.querySelectorAll("[data-scene-id]");
      elements.forEach((el) => observer.observe(el));
    }, 100);

    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [activeScenes]);

  // ========== 渲染 ==========

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!script) {
    return (
      <div className="rounded-xl border border-destructive/30 bg-card p-6">
        <p className="font-medium">项目剧本工作区未初始化</p>
        <p className="mt-1 text-sm text-muted-foreground">请检查项目创建流程后重试。</p>
      </div>
    );
  }

  return (
    <motion.div
      variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.08, delayChildren: 0.1 } } }}
      initial="hidden"
      animate="visible"
      className="flex min-h-0 flex-1 rounded-xl border border-border/20 overflow-hidden bg-card/10"
    >
      {/* 左栏：树形导航 */}
      <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } } }} className="shrink-0 hidden xl:block">
      <EpisodeTree
        onRefresh={handleManualRefreshScript}
        isRefreshing={isRefreshingScript}
        episodes={episodes}
        expandedEpisodes={expandedEpisodes}
        activeEpisodeId={activeEpisodeId}
        selectedSceneId={selectedSceneId}
        loadingEpisodes={loadingEpisodes}
        episodeScenes={episodeScenes}
        collapsed={isScriptSidebarCollapsed}
        onToggleEpisode={toggleEpisode}
        onSelectEpisode={selectEpisode}
        onSelectScene={handleSelectScene}
        onCollapsedChange={handleSetScriptSidebarCollapsed}
        onAddEpisode={handleAddEpisode}
        onDeleteEpisode={handleDeleteEpisode}
        onAddScene={handleAddScene}
        onDeleteScene={handleDeleteScene}
        onReorderScenes={handleReorderScenes}
        onParseEpisode={handleOpenEpisodeParse}
        onGenerateStoryboard={handleGenerateEpisodeStoryboard}
      />
      </motion.div>

      {/* 中栏：场次卡片列表 */}
      <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } } }} className="flex-1 flex flex-col min-w-0">
        <div className="2xl:hidden px-4 py-3 border-b border-border/20 flex items-center justify-between shrink-0 bg-card/30 backdrop-blur-sm">
          <Sheet open={leftSheetOpen} onOpenChange={setLeftSheetOpen}>
            <SheetTrigger
              render={
                <button className="xl:hidden p-1.5 -ml-1.5 rounded-md hover:bg-muted text-muted-foreground transition-colors">
                  <Menu className="h-5 w-5" />
                </button>
              }
            />
            <SheetContent side="left" className="w-[300px] p-0 border-r-0 flex flex-col pt-12">
              <EpisodeTree
        onRefresh={handleManualRefreshScript}
        isRefreshing={isRefreshingScript}
        episodes={episodes}
                expandedEpisodes={expandedEpisodes}
                activeEpisodeId={activeEpisodeId}
                selectedSceneId={selectedSceneId}
                loadingEpisodes={loadingEpisodes}
                episodeScenes={episodeScenes}
                onToggleEpisode={toggleEpisode}
                onSelectEpisode={selectEpisode}
                onSelectScene={handleSelectScene}
                onAddEpisode={handleAddEpisode}
                onDeleteEpisode={handleDeleteEpisode}
                onAddScene={handleAddScene}
                onDeleteScene={handleDeleteScene}
                onReorderScenes={handleReorderScenes}
                onParseEpisode={handleOpenEpisodeParse}
                onGenerateStoryboard={handleGenerateEpisodeStoryboard}
              />
            </SheetContent>
          </Sheet>
          <span className="text-sm font-semibold text-foreground/80">剧本内容</span>
          <Sheet open={rightSheetOpen} onOpenChange={setRightSheetOpen}>
            <SheetTrigger
              render={
                <button className="p-1.5 -mr-1.5 rounded-md hover:bg-muted text-muted-foreground transition-colors">
                  <Info className="h-5 w-5" />
                </button>
              }
            />
            <SheetContent side="right" className="w-[300px] p-0 border-l-0 flex flex-col pt-12 overflow-y-auto">
              {selectedScene ? (
                <SceneDetail scene={selectedScene} projectId={projectId} />
              ) : (
                <ScriptOverview script={script} episodes={episodes} projectName={project?.name} />
              )}
            </SheetContent>
          </Sheet>
        </div>
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
          <SceneList
            activeEpisode={activeEpisode}
            activeScenes={activeScenes}
            isLoading={isActiveLoading}
            selectedSceneId={selectedSceneId}
            onSelectScene={setSelectedSceneId}
            onSceneUpdated={handleSceneUpdated}
            onAddScene={handleAddScene}
            onInsertScene={handleInsertScene}
            onDeleteScene={handleDeleteScene}
            onEpisodeUpdated={(updated) => {
              setEpisodes((prev) =>
                prev.map((ep) => (ep.id === updated.id ? updated : ep))
              );
            }}
          />
        </div>
      </motion.div>

      {/* 右栏：场景详情 / 剧本概览 */}
      <motion.div variants={{ hidden: { opacity: 0, y: 16 }, visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } } }} className="w-72 border-l border-border/20 flex-col shrink-0 bg-card/20 overflow-y-auto hidden 2xl:flex">
        {selectedScene ? (
          <SceneDetail scene={selectedScene} projectId={projectId} />
        ) : (
          <ScriptOverview script={script} episodes={episodes} projectName={project?.name} />
        )}
      </motion.div>

      {/* 分集解析弹窗 */}
      <EpisodeParseDialog
        open={showEpisodeParseDialog}
        episode={episodes.find((e) => e.id === episodeToParseId) ?? null}
        hasExistingScenes={(episodeToParseId && (episodeScenes[episodeToParseId]?.length ?? 0) > 0) || false}
        existingSceneCount={episodeToParseId ? (episodeScenes[episodeToParseId]?.length ?? 0) : 0}
        onClose={() => setShowEpisodeParseDialog(false)}
        onStartParse={handleEpisodeParse}
      />
    </motion.div>
  );
}
