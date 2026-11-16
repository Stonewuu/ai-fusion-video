"use client";

import { useResourceChangeStore } from "@/lib/store/resource-change-store";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { useParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
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
import { useScriptMutations } from "./_components/use-script-mutations";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { EpisodeTree } from "./_components/episode-tree";
import { ScriptWorkspaceActions } from "./_components/script-workspace-actions";
import { SceneList } from "./_components/scene-list";
import { useSceneNavigation } from "./_components/use-scene-navigation";
import { SceneDetail } from "./_components/scene-detail";
import { ScriptOverview } from "./_components/script-overview";
import {
  ScriptWorkspacePanel,
  ScriptPanelToggle,
} from "./_components/script-workspace-panel";
import {
  ScriptDirectorySheet,
  ScriptDetailsSheet,
} from "./_components/script-panel-sheets";
import { EpisodeParseDialog } from "@/components/dashboard/episode-parse-dialog";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useProject } from "../project-context";

const SCRIPT_SIDEBAR_COLLAPSED_STORAGE_KEY = "fusion-script-sidebar-collapsed";
const SCRIPT_DETAILS_COLLAPSED_STORAGE_KEY = "fusion-script-details-collapsed";

export default function ScriptTabPage() {
  const reduceMotion = useReducedMotion();
  const params = useParams();
  const projectId = Number(params.id);
  const { project } = useProject();
  const { confirm, alert } = useConfirm();

  const [loading, setLoading] = useState(true);
  const [isRefreshingScript, setIsRefreshingScript] = useState(false);
  const [script, setScript] = useState<Script | null>(null);
  const [episodes, setEpisodes] = useState<ScriptEpisode[]>([]);
  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);

  // 当前展示的分集
  const [activeEpisodeId, setActiveEpisodeId] = useState<number | null>(null);
  // 树结构：展开状态
  const [expandedEpisodes, setExpandedEpisodes] = useState<Set<number>>(
    new Set(),
  );
  // 各分集的场次缓存
  const [episodeScenes, setEpisodeScenes] = useState<
    Record<number, SceneItem[]>
  >({});
  const [loadingEpisodes, setLoadingEpisodes] = useState<Set<number>>(
    new Set(),
  );

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
  const [isDetailsCollapsed, setIsDetailsCollapsed] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    return (
      localStorage.getItem(SCRIPT_DETAILS_COLLAPSED_STORAGE_KEY) === "true"
    );
  });
  const [isScriptSidebarCollapsed, setIsScriptSidebarCollapsed] = useState(
    () => {
      if (typeof window === "undefined") {
        return false;
      }
      return (
        localStorage.getItem(SCRIPT_SIDEBAR_COLLAPSED_STORAGE_KEY) === "true"
      );
    },
  );

  // 分集解析弹窗
  const [showEpisodeParseDialog, setShowEpisodeParseDialog] = useState(false);
  const [episodeToParseId, setEpisodeToParseId] = useState<number | null>(null);

  // Pipeline store
  const { addPipeline, setPanelExpanded, setExpandedTaskId } =
    usePipelineStore();

  const handleSetScriptSidebarCollapsed = useCallback((collapsed: boolean) => {
    setIsScriptSidebarCollapsed(collapsed);
    localStorage.setItem(
      SCRIPT_SIDEBAR_COLLAPSED_STORAGE_KEY,
      collapsed ? "true" : "false",
    );
  }, []);

  const toggleDetails = () => {
    const next = !isDetailsCollapsed;
    setIsDetailsCollapsed(next);
    localStorage.setItem(SCRIPT_DETAILS_COLLAPSED_STORAGE_KEY, String(next));
  };

  useEffect(() => {
    const directoryMedia = window.matchMedia("(min-width: 1280px)");
    const detailsMedia = window.matchMedia("(min-width: 1536px)");
    const handleResize = () => {
      if (directoryMedia.matches) {
        setLeftSheetOpen(false);
      }
      if (detailsMedia.matches) {
        setRightSheetOpen(false);
      }
    };
    handleResize();
    directoryMedia.addEventListener("change", handleResize);
    detailsMedia.addEventListener("change", handleResize);
    return () => {
      directoryMedia.removeEventListener("change", handleResize);
      detailsMedia.removeEventListener("change", handleResize);
    };
  }, []);

  // ========== 数据加载 ==========

  const loadEpisodeScenes = async (episodeId: number) => {
    if (episodeScenes[episodeId]) {
      return;
    }
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

  const loadStoryboardForScript = useCallback(
    async (scriptId: number) => {
      const currentStoryboard = await storyboardApi.getByProject(projectId);
      if (!currentStoryboard || currentStoryboard.scriptId !== scriptId) {
        setStoryboard(null);
        return { storyboard: null, episodes: [] as StoryboardEpisode[] };
      }

      setStoryboard(currentStoryboard);
      const currentEpisodes = await storyboardApi.listEpisodes(
        currentStoryboard.id,
      );
      return { storyboard: currentStoryboard, episodes: currentEpisodes };
    },
    [projectId],
  );

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
          // 首次只加载当前集，其他分集在选中时读取。
          setExpandedEpisodes(new Set([eps[0].id]));
          await loadEpisodeScenes(eps[0].id);
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
              const shouldLoad =
                currentExpanded.has(e.id) ||
                !!currentScenes[e.id] ||
                e.id === nextActiveId;
              if (shouldLoad) {
                try {
                  const scenes = await scriptApi.listScenes(e.id);
                  newEpisodeScenes[e.id] = scenes;
                } catch (err) {
                  console.error(`刷新分集 ${e.id} 场次失败:`, err);
                  toastApiError(err, `刷新分集 ${e.id} 场次失败`);
                }
              }
            }),
          );

          setEpisodeScenes(newEpisodeScenes);
        } else {
          setActiveEpisodeId(null);
          setSelectedSceneId(null);
          setEpisodeScenes({});
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

  // 所有已提交的业务资源变化触发刷新
  const scriptsInvalidation = useResourceChangeStore(
    (s) => s.revisions.scripts,
  );
  const scriptsInvRef = useRef(scriptsInvalidation);
  useEffect(() => {
    if (scriptsInvRef.current !== scriptsInvalidation) {
      scriptsInvRef.current = scriptsInvalidation;
      refreshScript();
    }
  }, [scriptsInvalidation, refreshScript]);

  const storyboardsInvalidation = useResourceChangeStore(
    (s) => s.revisions.storyboards,
  );
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
    cancelSceneScroll();
    setActiveEpisodeId(episodeId);
    setSelectedSceneId(null);
    if (!expandedEpisodes.has(episodeId)) {
      setExpandedEpisodes((prev) => new Set(prev).add(episodeId));
    }
    setLeftSheetOpen(false);
    await loadEpisodeScenes(episodeId);
  };

  const {
    scrollContainerRef,
    isAutoScrollingRef,
    scrollToScene,
    cancelSceneScroll,
  } = useSceneNavigation(
    activeEpisodeId ? episodeScenes[activeEpisodeId] : undefined,
    setSelectedSceneId,
    !loading && !!script,
    activeEpisodeId ? loadingEpisodes.has(activeEpisodeId) : false,
  );

  const handleSelectScene = (sceneId: number, episodeId: number) => {
    selectEpisode(episodeId);
    setLeftSheetOpen(false);
    scrollToScene(sceneId);
  };

  // ========== 增删操作 ==========

  const {
    handleAddEpisode,
    handleDeleteEpisode,
    handleAddScene,
    handleInsertScene,
    handleDeleteScene,
    handleSceneUpdated,
    handleReorderScenes,
  } = useScriptMutations({
    script,
    episodes,
    episodeScenes,
    setEpisodeScenes,
    setExpandedEpisodes,
    setActiveEpisodeId,
    setSelectedSceneId,
    refreshScript,
    scrollToScene,
  });
  const episodeParseRequest = useCommandRequestKey();

  // ========== 分集 AI 解析 ==========

  const handleOpenEpisodeParse = (episodeId: number) => {
    setEpisodeToParseId(episodeId);
    setShowEpisodeParseDialog(true);
  };

  const handleEpisodeParse = async (
    rawContent: string,
    version: number,
    replaceExisting: boolean,
  ) => {
    if (!script || !episodeToParseId) {
      return;
    }
    const ep = episodes.find((e) => e.id === episodeToParseId);
    if (!ep) {
      return;
    }

    try {
      // 1. 先保存原文到分集记录
      const update = { id: episodeToParseId, version, rawContent };
      await scriptApi.updateEpisode(
        update,
        episodeParseRequest.forPayload(update),
      );
      episodeParseRequest.clear();
      await refreshScript();

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
          message: `解析剧本分集 ID ${episodeToParseId} 的全部原文并保存梗概。${replaceExisting ? "本次已确认替换该集现有场次。" : "保留已有场次，只补齐缺失内容和资产关联，完成后更新本集梗概和解析状态。"}`,
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
      throw err;
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
    if (!script) {
      return;
    }
    const ep = episodes.find((item) => item.id === episodeId);
    if (!ep) {
      return;
    }

    try {
      const targetStoryboard = await ensureStoryboardForScript();
      const currentStoryboardEpisodes = await storyboardApi.listEpisodes(
        targetStoryboard.id,
      );

      const boundEpisode = currentStoryboardEpisodes.find(
        (item) => item.scriptEpisodeId === episodeId,
      );
      const legacyUnboundEpisode = currentStoryboardEpisodes.find(
        (item) =>
          (item.scriptEpisodeId == null ||
            !episodes.some((source) => source.id === item.scriptEpisodeId)) &&
          item.episodeNumber === ep.episodeNumber,
      );

      if (!boundEpisode && legacyUnboundEpisode) {
        await alert({
          title: "未绑定分镜集",
          description: `分镜中存在未绑定的第 ${ep.episodeNumber} 集，请先到分镜页将旧分镜集绑定到当前剧本集后再重新生成。`,
          variant: "warning",
        });
        return;
      }

      if (boundEpisode) {
        const confirmed = await confirm({
          title: "覆盖生成本集分镜",
          description: `将覆盖第 ${ep.episodeNumber} 集已有分镜内容，不影响其它集。确定继续？`,
          variant: "ai",
          confirmText: "确定覆盖生成",
        });
        if (!confirmed) {
          return;
        }
      }

      await scriptApi.getEpisode(ep.id);
      const pipelineId = addPipeline({
        label: `AI 分镜 · 第 ${ep.episodeNumber} 集`,
        projectId,
        request: {
          agentType: "episode_storyboard_writer",
          toolExecutionMode: "FULL_ACCESS",
          category: "pipeline",
          title: `AI 分镜 · 第 ${ep.episodeNumber} 集`,
          message: `请为剧本分集（scriptEpisodeId: ${ep.id}）生成分镜，并保存到分镜脚本 ${targetStoryboard.id}。${boundEpisode ? `已确认覆盖分镜集 ${boundEpisode.id}，先读取剧本并设计第一场，再以 replaceEpisodeContent=true 保存第一场，后续逐场追加。` : ""}`,
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
  const activeScenes = useMemo(
    () => (activeEpisodeId ? episodeScenes[activeEpisodeId] || [] : []),
    [activeEpisodeId, episodeScenes],
  );
  const isActiveLoading = activeEpisodeId
    ? loadingEpisodes.has(activeEpisodeId)
    : false;

  // ========== 滚动监听：IntersectionObserver ==========
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || isActiveLoading || activeScenes.length === 0) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (isAutoScrollingRef.current) {
          return;
        }
        // 找到最靠近顶部且可见的场次
        let topEntry: IntersectionObserverEntry | null = null;
        for (const entry of entries) {
          if (entry.isIntersecting) {
            if (
              !topEntry ||
              entry.boundingClientRect.top < topEntry.boundingClientRect.top
            ) {
              topEntry = entry;
            }
          }
        }
        if (topEntry) {
          const id = Number((topEntry.target as HTMLElement).dataset.sceneId);
          if (id) {
            setSelectedSceneId(id);
          }
        }
      },
      {
        root: container,
        rootMargin: "-10% 0px -70% 0px",
        threshold: 0,
      },
    );

    const elements = container.querySelectorAll("[data-scene-id]");
    elements.forEach((el) => observer.observe(el));

    return () => {
      observer.disconnect();
    };
  }, [activeScenes, isActiveLoading, isAutoScrollingRef, scrollContainerRef]);

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
        <p className="mt-1 text-sm text-muted-foreground">
          请检查项目创建流程后重试。
        </p>
      </div>
    );
  }

  const treeProps = {
    onRefresh: handleManualRefreshScript,
    isRefreshing: isRefreshingScript,
    episodes,
    expandedEpisodes,
    activeEpisodeId,
    selectedSceneId,
    loadingEpisodes,
    episodeScenes,
    onToggleEpisode: toggleEpisode,
    onSelectEpisode: selectEpisode,
    onSelectScene: handleSelectScene,
    onAddEpisode: handleAddEpisode,
    onDeleteEpisode: handleDeleteEpisode,
    onAddScene: handleAddScene,
    onDeleteScene: handleDeleteScene,
    onReorderScenes: handleReorderScenes,
    onParseEpisode: handleOpenEpisodeParse,
    onGenerateStoryboard: handleGenerateEpisodeStoryboard,
  };
  const details = selectedScene ? (
    <SceneDetail
      scene={selectedScene}
      projectId={projectId}
      showHeading={false}
    />
  ) : (
    <ScriptOverview
      script={script}
      episodes={episodes}
      projectName={project?.name}
      showHeading={false}
    />
  );
  const detailsTitle = selectedScene ? "场景详情" : "剧本信息";

  return (
    <motion.div
      variants={{
        hidden: { opacity: 0 },
        visible: {
          opacity: 1,
          transition: { staggerChildren: 0.08, delayChildren: 0.1 },
        },
      }}
      initial={reduceMotion ? false : "hidden"}
      animate="visible"
      className="flex min-h-0 grow basis-0 rounded-xl border border-border/20 overflow-hidden bg-card/10"
    >
      <ScriptWorkspacePanel
        side="left"
        collapsed={isScriptSidebarCollapsed}
        onToggle={() =>
          handleSetScriptSidebarCollapsed(!isScriptSidebarCollapsed)
        }
      >
        <EpisodeTree
          {...treeProps}
          headerAction={<span className="size-9 shrink-0" aria-hidden />}
        />
      </ScriptWorkspacePanel>

      {/* 中栏：场次卡片列表 */}
      <motion.div
        variants={{
          hidden: { opacity: 0, y: 16 },
          visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
          },
        }}
        className="flex-1 flex flex-col min-w-0 min-h-0"
      >
        <ScriptWorkspaceActions
          script={script}
          episodes={episodes}
          onImported={refreshScript}
          onAddEpisode={handleAddEpisode}
          directoryControl={
            <ScriptPanelToggle
              side="left"
              collapsed={isScriptSidebarCollapsed}
              onToggle={() =>
                handleSetScriptSidebarCollapsed(!isScriptSidebarCollapsed)
              }
              sheetOpen={leftSheetOpen}
              onOpenSheet={() => setLeftSheetOpen(true)}
            />
          }
          detailsControl={
            <ScriptPanelToggle
              side="right"
              collapsed={isDetailsCollapsed}
              onToggle={toggleDetails}
              sheetOpen={rightSheetOpen}
              onOpenSheet={() => setRightSheetOpen(true)}
            />
          }
        />
        <div ref={scrollContainerRef} className="flex-1 overflow-y-auto">
          <SceneList
            onParseEpisode={handleOpenEpisodeParse}
            duplicateTitle={
              !!activeEpisode &&
              episodes.some(
                (ep) =>
                  ep.id !== activeEpisode.id &&
                  ep.title
                    ?.normalize("NFKC")
                    .replace(/\s+/g, "")
                    .toLowerCase() ===
                    activeEpisode.title
                      ?.normalize("NFKC")
                      .replace(/\s+/g, "")
                      .toLowerCase(),
              )
            }
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
              void refreshScript();
              setEpisodes((prev) =>
                prev.map((ep) => (ep.id === updated.id ? updated : ep)),
              );
            }}
          />
        </div>
      </motion.div>

      <ScriptWorkspacePanel side="right" collapsed={isDetailsCollapsed}>
        <h3 className="shrink-0 px-4 pt-4 text-sm font-semibold">
          {detailsTitle}
        </h3>
        <OverlayScrollArea className="min-h-0 flex-1">
          {details}
        </OverlayScrollArea>
      </ScriptWorkspacePanel>
      <ScriptDirectorySheet
        open={leftSheetOpen}
        onOpenChange={setLeftSheetOpen}
        treeProps={treeProps}
      />
      <ScriptDetailsSheet
        open={rightSheetOpen}
        onOpenChange={setRightSheetOpen}
        title={detailsTitle}
      >
        {details}
      </ScriptDetailsSheet>

      {/* 分集解析弹窗 */}
      <EpisodeParseDialog
        open={showEpisodeParseDialog}
        episode={episodes.find((e) => e.id === episodeToParseId) ?? null}
        hasExistingScenes={
          (episodeToParseId &&
            (episodeScenes[episodeToParseId]?.length ?? 0) > 0) ||
          false
        }
        existingSceneCount={
          episodeToParseId ? (episodeScenes[episodeToParseId]?.length ?? 0) : 0
        }
        onClose={() => setShowEpisodeParseDialog(false)}
        onStartParse={handleEpisodeParse}
      />
    </motion.div>
  );
}
