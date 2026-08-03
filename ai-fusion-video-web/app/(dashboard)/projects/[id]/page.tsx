"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  FileText,
  Clock,
  Loader2,
  BookOpen,
  Eye,
  RefreshCw,
  Film,
  Type,
  Palette,
  Monitor,
  ChevronRight,
  Images,
  Users,
  MapPin,
  Wrench,
} from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import type { Script } from "@/lib/api/script";
import type { Storyboard, StoryboardStatistics } from "@/lib/api/storyboard";
import { assetApi, type Asset } from "@/lib/api/asset";
import { artStyleApi, type ArtStylePreset } from "@/lib/api/art-style";
import { projectApi } from "@/lib/api/project";
import { resolveMediaUrl } from "@/lib/api/client";
import { MainContentFrame } from "@/components/dashboard/main-content-frame";
import { SafeImage } from "@/components/ui/safe-image";
import { Button } from "@/components/ui/button";
import { useProject } from "./project-context";
import { ParseScriptDialog } from "@/components/dashboard/parse-script-dialog";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { useConfirm } from "@/components/ui/confirm-dialog";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};
const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

function formatTime(iso: string) {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "刚刚";
  if (mins < 60) return `${mins} 分钟前`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} 天前`;
  return d.toLocaleDateString("zh-CN");
}

const parsingStatusMap: Record<number, { label: string; cls: string }> = {
  0: { label: "待解析", cls: "text-muted-foreground bg-muted/50 border-border/30" },
  1: { label: "解析中", cls: "text-blue-400 bg-blue-500/10 border-blue-500/20" },
  2: { label: "已完成", cls: "text-green-400 bg-green-500/10 border-green-500/20" },
  3: { label: "解析失败", cls: "text-destructive bg-destructive/10 border-destructive/20" },
};

const storyboardStatusMap: Record<number, { label: string; cls: string }> = {
  0: { label: "草稿", cls: "text-muted-foreground bg-muted/50 border-border/30" },
  1: { label: "进行中", cls: "text-blue-400 bg-blue-500/10 border-blue-500/20" },
  2: { label: "已完成", cls: "text-green-400 bg-green-500/10 border-green-500/20" },
};

export default function ProjectOverviewPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = Number(params.id);
  const { project } = useProject();
  const { confirm, alert } = useConfirm();
  const properties = (project?.properties as Record<string, string>) || {};

  const [script, setScript] = useState<Script | null>(null);
  const [scriptEpisodeCount, setScriptEpisodeCount] = useState(0);
  const [scriptSceneCount, setScriptSceneCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showParseDialog, setShowParseDialog] = useState(false);

  // 分镜状态
  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
  const [storyboardStatistics, setStoryboardStatistics] = useState<StoryboardStatistics>({
    episodeCount: 0,
    sceneCount: 0,
    itemCount: 0,
  });
  const [loadingStoryboard, setLoadingStoryboard] = useState(true);
  const [initializingWorkspace, setInitializingWorkspace] = useState(false);

  // 画风预设
  const [artPresets, setArtPresets] = useState<ArtStylePreset[]>([]);

  useEffect(() => {
    artStyleApi.getPresets().then(setArtPresets).catch((error) => {
      console.error("加载画风预设失败:", error);
      toastApiError(error, "加载画风预设失败");
    });
  }, []);

  // 资产状态
  const [assets, setAssets] = useState<Asset[]>([]);
  const [assetCounts, setAssetCounts] = useState<Record<string, number>>({});
  const [loadingAssets, setLoadingAssets] = useState(true);

  // 串行加载概览数据：剧本 → 分镜统计 → 资产
  const loadAllData = useCallback(async () => {
    // 1. 加载固定剧本、分镜容器及真实内容统计
    try {
      setLoading(true);
      setLoadingStoryboard(true);
      const overview = await projectApi.getWorkspaceOverview(projectId);
      setScript(overview.script);
      setScriptEpisodeCount(overview.scriptEpisodeCount);
      setScriptSceneCount(overview.scriptSceneCount);
      setStoryboard(overview.storyboard);
      setStoryboardStatistics(overview.storyboardStatistics);
    } catch (err) {
      console.error("加载项目工作区失败:", err);
      toastApiError(err, "加载项目工作区失败");
    } finally {
      setLoading(false);
      setLoadingStoryboard(false);
    }

    // 2. 加载资产
    try {
      setLoadingAssets(true);
      const data = await assetApi.list(projectId);
      setAssets(data);
      // 统计各类型数量
      const counts: Record<string, number> = {};
      data.forEach((a) => {
        counts[a.type] = (counts[a.type] || 0) + 1;
      });
      setAssetCounts(counts);
    } catch (err) {
      console.error("加载资产数据失败:", err);
      toastApiError(err, "加载资产数据失败");
    } finally {
      setLoadingAssets(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  // AI 生成剧本：创建成功后触发 pipeline
  const { addPipeline, setPanelExpanded, setExpandedTaskId } =
    usePipelineStore();

  const handleAiScriptCreated = (script: { id: number; title: string }) => {
    const scriptDisplayTitle = project?.name?.trim() || "未命名项目";

    // 刷新列表
    loadAllData();

    // 启动 pipeline
    const pipelineId = addPipeline({
      label: `AI 生成剧本 - ${scriptDisplayTitle}`,
      projectId,
      request: {
        agentType: "script_full_parse",
        toolExecutionMode: "FULL_ACCESS",
        category: "pipeline",
        title: `AI 剧本解析：${scriptDisplayTitle}`,
        projectId,
        context: { scriptId: script.id },
      },
      onComplete: () => {
        // pipeline 完成后刷新剧本数据
        loadAllData();
      },
    });

    // 打开任务中心大面板并展开该 pipeline
    setPanelExpanded(true);
    setExpandedTaskId(pipelineId);

    // 跳转到剧本页
    router.push(`/projects/${projectId}/scripts`);
  };

  // AI 生成分镜：启动 pipeline
  const handleAiStoryboard = async () => {
    if (!script) return;

    const scriptDisplayTitle = project?.name?.trim() || "未命名项目";

    try {
      // 固定复用项目唯一分镜容器，生成过程只补充内部集、场次和镜头。
      if (!storyboard) {
        throw new Error("项目分镜工作区未初始化");
      }
      const targetStoryboard = storyboard;

      const pipelineId = addPipeline({
        label: `AI 生成分镜 - ${scriptDisplayTitle}`,
        projectId,
        request: {
          agentType: "script_to_storyboard",
          toolExecutionMode: "FULL_ACCESS",
          category: "pipeline",
          title: `AI 生成分镜：${scriptDisplayTitle}`,
          projectId,
          context: { scriptId: script.id, storyboardId: targetStoryboard.id },
        },
        onComplete: () => {
          // pipeline 完成后刷新数据
          loadAllData();
        },
      });

      // 打开任务中心大面板并展开该 pipeline
      setPanelExpanded(true);
      setExpandedTaskId(pipelineId);

      // 跳转到分镜页
      router.push(`/projects/${projectId}/storyboards`);
    } catch (err) {
      console.error("生成分镜失败:", err);
      toastApiError(err, "生成分镜失败，请重试");
    }
  };

  const handleInitializeWorkspace = async () => {
    const ok = await confirm({
      title: "初始化项目工作区",
      description: "将只补齐缺失的剧本或分镜工作区，不会删除或覆盖任何现有数据。确定继续？",
      variant: "ai",
      confirmText: "确定初始化",
    });
    if (!ok) return;

    try {
      setInitializingWorkspace(true);
      const overview = await projectApi.initializeWorkspace(projectId);
      setScript(overview.script);
      setScriptEpisodeCount(overview.scriptEpisodeCount);
      setScriptSceneCount(overview.scriptSceneCount);
      setStoryboard(overview.storyboard);
      setStoryboardStatistics(overview.storyboardStatistics);
      toast.success("项目工作区初始化完成");
    } catch (err) {
      console.error("初始化项目工作区失败:", err);
      toastApiError(err, "初始化项目工作区失败");
    } finally {
      setInitializingWorkspace(false);
    }
  };

  if (loading && loadingStoryboard) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const status = script
    ? parsingStatusMap[script.parsingStatus] || parsingStatusMap[0]
    : null;

  const sbStatus = storyboard
    ? storyboardStatusMap[storyboard.status] || storyboardStatusMap[0]
    : null;
  const scriptHasContent = Boolean(script?.rawContent?.trim())
    || scriptEpisodeCount > 0
    || scriptSceneCount > 0;
  const storyboardHasContent = storyboardStatistics.episodeCount > 0
    || storyboardStatistics.sceneCount > 0;

  return (
    <MainContentFrame>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
      {/* 项目详情概览条 */}
      <motion.div variants={itemVariants} className="mb-8">
        <button
          type="button"
          onClick={() => router.push(`/projects/${projectId}/settings`)}
          className={cn(
            "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
            "w-full px-5 py-4 flex items-center gap-6 cursor-pointer group text-left",
            "hover:border-primary/30 hover:bg-primary/2 transition-colors duration-200",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          )}
        >
          {/* 项目名称 */}
          <div className="flex items-center gap-2 shrink-0">
            <h2 className="text-lg font-semibold">{project?.name || "未命名项目"}</h2>
          </div>

          <div className="h-5 w-px bg-border/40" />

          {/* 项目类型 */}
          <div className="flex items-center gap-1.5 text-sm">
            <Type className="h-3.5 w-3.5 text-primary/70" />
            <span className="text-muted-foreground text-xs">类型</span>
            <span className={cn(
              "px-2 py-0.5 rounded-md text-xs font-medium",
              properties.type
                ? "bg-primary/10 text-primary"
                : "bg-muted/50 text-muted-foreground"
            )}>
              {properties.type || "未设置"}
            </span>
          </div>

          {/* 画风 */}
          <div className="flex items-center gap-1.5 text-sm">
            <Palette className="h-3.5 w-3.5 text-primary/70" />
            <span className="text-muted-foreground text-xs">画风</span>
            <span className={cn(
              "px-2 py-0.5 rounded-md text-xs font-medium",
              project?.artStyle
                ? "bg-primary/10 text-primary"
                : "bg-muted/50 text-muted-foreground"
            )}>
              {project?.artStyle
                ? (artPresets.find((p) => p.key === project.artStyle)?.name ?? project.artStyle)
                : "未设置"}
            </span>
          </div>

          {/* 画面比例 */}
          <div className="flex items-center gap-1.5 text-sm">
            <Monitor className="h-3.5 w-3.5 text-primary/70" />
            <span className="text-muted-foreground text-xs">比例</span>
            <span className={cn(
              "px-2 py-0.5 rounded-md text-xs font-medium",
              properties.aspectRatio
                ? "bg-primary/10 text-primary"
                : "bg-muted/50 text-muted-foreground"
            )}>
              {properties.aspectRatio || "未设置"}
            </span>
          </div>

          {/* 跳转提示 */}
          <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground group-hover:text-primary transition-colors">
            修改设置
            <ChevronRight className="h-3.5 w-3.5" />
          </div>
        </button>
      </motion.div>

      {/* 剧本区域 */}
      <motion.div variants={itemVariants}>
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-5">
          <BookOpen className="h-5 w-5 text-primary" />
          总剧本
        </h2>

        {script ? (
          /* 已有剧本：展示卡片 */
          <div
            className={cn(
              "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
              "p-6 space-y-4"
            )}
          >
            {/* 剧本信息 */}
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-4 min-w-0">
                <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
                  <FileText className="h-6 w-6 text-primary" />
                </div>
                <div className="min-w-0">
                  <h3 className="text-lg font-semibold truncate">
                    {project?.name || "未命名项目"}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-0.5 line-clamp-2">
                    {script.storySynopsis || script.genre || (scriptHasContent ? "剧本内容待完善" : "尚未添加剧本内容")}
                  </p>
                </div>
              </div>
              <span
                className={cn(
                  "text-xs px-2.5 py-1 rounded-md border font-medium shrink-0 whitespace-nowrap",
                  status?.cls
                )}
              >
                {status?.label}
              </span>
            </div>

            {/* 元信息 */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span>{scriptEpisodeCount} 集</span>
              <span>{scriptSceneCount} 场次</span>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatTime(script.updateTime)}
              </span>
            </div>

            {/* 操作按钮 */}
            <div className="flex items-center gap-2 pt-2 border-t border-border/20">
              <Button onClick={() => router.push(`/projects/${projectId}/scripts`)}>
                <Eye className="h-4 w-4" />
                查看剧本
              </Button>
              <Button variant="ai" onClick={() => setShowParseDialog(true)}>
                <RefreshCw className="h-4 w-4" />
                {scriptHasContent ? "重新解析内容" : "AI 解析"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="rounded-xl border border-destructive/30 bg-card p-6">
            <p className="font-medium">项目剧本工作区未初始化</p>
            <p className="mt-1 text-sm text-muted-foreground">
              可补齐缺失的剧本和分镜容器，不会删除或覆盖任何现有数据。
            </p>
            <div className="mt-4">
              <Button onClick={handleInitializeWorkspace} disabled={initializingWorkspace}>
                <RefreshCw className={cn("h-4 w-4", initializingWorkspace && "animate-spin")} />
                {initializingWorkspace ? "正在初始化" : "重新初始化项目"}
              </Button>
            </div>
          </div>
        )}

        {script && (
          <ParseScriptDialog
            open={showParseDialog}
            scriptId={script.id}
            mode={scriptHasContent ? "reparse" : "create"}
            onClose={() => setShowParseDialog(false)}
            onCreated={handleAiScriptCreated}
          />
        )}
      </motion.div>

      {/* 分镜区域 */}
      <motion.div variants={itemVariants} className="mt-10">
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-5">
          <Film className="h-5 w-5 text-cyan-400" />
          分镜
        </h2>

        {loadingStoryboard ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : storyboard ? (
          /* 已有分镜：展示卡片 */
          <div
            className={cn(
              "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
              "p-6 space-y-4"
            )}
          >
            {/* 分镜信息 */}
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-4 min-w-0">
                <div className="h-12 w-12 rounded-xl bg-cyan-500/10 flex items-center justify-center shrink-0">
                  <Film className="h-6 w-6 text-cyan-400" />
                </div>
                <div className="min-w-0">
                  <h3 className="text-lg font-semibold truncate">
                    {project?.name || "未命名项目"}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-0.5 line-clamp-2">
                    {storyboard.description
                      || (!storyboardHasContent
                        ? "暂无分镜内容"
                        : storyboardStatistics.sceneCount === 0
                          ? "已创建分集，待添加场次"
                          : "分镜内容已建立")}
                  </p>
                </div>
              </div>
              <span
                className={cn(
                  "text-xs px-2.5 py-1 rounded-md border font-medium shrink-0 whitespace-nowrap",
                  sbStatus?.cls
                )}
              >
                {sbStatus?.label}
              </span>
            </div>

            {/* 统计信息 */}
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              <span>{storyboardStatistics.episodeCount} 集</span>
              <span>{storyboardStatistics.sceneCount} 场次</span>
              <span>{storyboardStatistics.itemCount} 镜头</span>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatTime(storyboard.updateTime)}
              </span>
            </div>

            {/* 操作按钮 */}
            <div className="flex items-center gap-2 pt-2 border-t border-border/20">
              <Button onClick={() => router.push(`/projects/${projectId}/storyboards`)}>
                <Eye className="h-4 w-4" />
                查看分镜
              </Button>
              <Button
                variant="ai"
                onClick={async () => {
                  if (!script) {
                    await alert({
                      title: "工作区未初始化",
                      description: "项目剧本工作区未初始化，请先初始化工作区。",
                      variant: "warning",
                    });
                    return;
                  }
                  if (storyboardHasContent) {
                    const ok = await confirm({
                      title: "确认 AI 补全分镜？",
                      description: "AI 将补齐缺失的分镜集，不会删除已有分镜内容。确定继续？",
                      variant: "ai",
                      confirmText: "开始 AI 补全",
                    });
                    if (!ok) return;
                  } else {
                    const ok = await confirm({
                      title: "确认 AI 生成分镜？",
                      description: "AI 将基于当前剧本内容，自动分析剧情并生成全套分镜与镜头结构。确定继续？",
                      variant: "ai",
                      confirmText: "开始 AI 生成",
                    });
                    if (!ok) return;
                  }
                  void handleAiStoryboard();
                }}
              >
                <RefreshCw className="h-4 w-4" />
                {storyboardHasContent ? "AI 补全" : "AI 生成"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="rounded-xl border border-destructive/30 bg-card p-6">
            <p className="font-medium">项目分镜工作区未初始化</p>
            <p className="mt-1 text-sm text-muted-foreground">
              可补齐缺失的分镜容器，不会删除或覆盖任何现有数据。
            </p>
            {script && (
              <div className="mt-4">
                <Button onClick={handleInitializeWorkspace} disabled={initializingWorkspace}>
                  <RefreshCw className={cn("h-4 w-4", initializingWorkspace && "animate-spin")} />
                  {initializingWorkspace ? "正在初始化" : "重新初始化项目"}
                </Button>
              </div>
            )}
          </div>
        )}
      </motion.div>

      {/* 资产概览区域 */}
      <motion.div variants={itemVariants} className="mt-10">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Images className="h-5 w-5 text-amber-400" />
            资产库
          </h2>
          <button
            onClick={() => router.push(`/projects/${projectId}/assets`)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-primary transition-colors"
          >
            查看全部
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {loadingAssets ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : assets.length > 0 ? (
          <div className="space-y-5">
            {/* 类型统计卡片 */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { key: "character", label: "角色", icon: Users, color: "text-blue-400", bg: "bg-blue-500/10" },
                { key: "scene", label: "场景", icon: MapPin, color: "text-green-400", bg: "bg-green-500/10" },
                { key: "prop", label: "道具", icon: Wrench, color: "text-amber-400", bg: "bg-amber-500/10" },
              ].map((t) => (
                <div
                  key={t.key}
                  onClick={() => router.push(`/projects/${projectId}/assets`)}
                  className={cn(
                    "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-4",
                    "flex items-center gap-3 cursor-pointer",
                    "hover:border-primary/30 hover:bg-primary/2 transition-all"
                  )}
                >
                  <div className={cn("h-10 w-10 rounded-lg flex items-center justify-center", t.bg)}>
                    <t.icon className={cn("h-5 w-5", t.color)} />
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{assetCounts[t.key] || 0}</p>
                    <p className="text-xs text-muted-foreground">{t.label}</p>
                  </div>
                </div>
              ))}
              <div
                onClick={() => router.push(`/projects/${projectId}/assets`)}
                className={cn(
                  "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-4",
                  "flex items-center gap-3 cursor-pointer",
                  "hover:border-primary/30 hover:bg-primary/2 transition-all"
                )}
              >
                <div className="h-10 w-10 rounded-lg flex items-center justify-center bg-muted/50">
                  <Images className="h-5 w-5 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{assets.length}</p>
                  <p className="text-xs text-muted-foreground">全部</p>
                </div>
              </div>
            </div>

            {/* 最近资产列表 */}
            <div className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm overflow-hidden">
              <div className="divide-y divide-border/20">
                {assets.slice(0, 6).map((asset) => {
                  const typeConfig: Record<string, { label: string; cls: string }> = {
                    character: { label: "角色", cls: "text-blue-400 bg-blue-500/10" },
                    scene: { label: "场景", cls: "text-green-400 bg-green-500/10" },
                    prop: { label: "道具", cls: "text-amber-400 bg-amber-500/10" },
                  };
                  const tc = typeConfig[asset.type] || { label: asset.type, cls: "text-muted-foreground bg-muted/50" };
                  return (
                    <div
                      key={asset.id}
                      onClick={() => router.push(`/projects/${projectId}/assets`)}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-muted/30 transition-colors cursor-pointer"
                    >
                      <div className="h-10 w-10 rounded-lg overflow-hidden shrink-0">
                        <SafeImage
                          src={resolveMediaUrl(asset.coverUrl) || ""}
                          alt={asset.name}
                          fallbackType={
                            asset.type === "character"
                              ? "avatar"
                              : asset.type === "scene"
                              ? "scene"
                              : asset.type === "prop"
                              ? "prop"
                              : "image"
                          }
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">{asset.name}</p>
                        {asset.description && (
                          <p className="text-xs text-muted-foreground truncate">{asset.description}</p>
                        )}
                      </div>
                      <span className={cn("px-2 py-0.5 rounded text-[10px] font-medium shrink-0", tc.cls)}>
                        {tc.label}
                      </span>
                    </div>
                  );
                })}
              </div>
              {assets.length > 6 && (
                <div
                  onClick={() => router.push(`/projects/${projectId}/assets`)}
                  className="px-4 py-2.5 text-center text-xs text-muted-foreground hover:text-primary transition-colors cursor-pointer border-t border-border/20"
                >
                  查看全部 {assets.length} 个资产 →
                </div>
              )}
            </div>
          </div>
        ) : (
          /* 没有资产：引导 */
          <div
            onClick={() => router.push(`/projects/${projectId}/assets`)}
            className={cn(
              "rounded-xl border border-dashed border-border/40 p-10",
              "flex flex-col items-center justify-center text-center",
              "bg-card/20 hover:border-amber-500/40 hover:bg-amber-500/5 transition-all cursor-pointer"
            )}
          >
            <div className="h-14 w-14 rounded-xl bg-amber-500/10 flex items-center justify-center mb-4">
              <Images className="h-7 w-7 text-amber-400" />
            </div>
            <p className="text-lg font-medium mb-1">管理资产</p>
            <p className="text-muted-foreground text-sm">
              进入资产库，创建角色、场景、道具等素材资产
            </p>
          </div>
        )}
      </motion.div>
      </motion.div>
    </MainContentFrame>
  );
}
