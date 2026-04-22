"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  FileText,
  Clock,
  Sparkles,
  Trash2,
  Loader2,
  BookOpen,
  Eye,
  RefreshCw,
  Plus,
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
import { cn } from "@/lib/utils";
import { scriptApi, type Script } from "@/lib/api/script";
import {
  storyboardApi,
  type Storyboard,
  type StoryboardEpisode,
} from "@/lib/api/storyboard";
import { assetApi, type Asset } from "@/lib/api/asset";
import { artStyleApi, type ArtStylePreset } from "@/lib/api/art-style";
import { resolveMediaUrl } from "@/lib/api/client";
import AssetTypePlaceholder from "@/components/dashboard/asset-type-placeholder";
import { useProject } from "./project-context";
import { CreateScriptDialog } from "@/components/dashboard/create-script-dialog";
import { ParseScriptDialog } from "@/components/dashboard/parse-script-dialog";
import { usePipelineStore } from "@/lib/store/pipeline-store";

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
  const properties = (project?.properties as Record<string, string>) || {};

  const [script, setScript] = useState<Script | null>(null);
  const [loading, setLoading] = useState(true);
  const [deletingScript, setDeletingScript] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showParseDialog, setShowParseDialog] = useState(false);
  const [parseMode, setParseMode] = useState<"create" | "reparse">("create");

  // 分镜状态
  const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
  const [storyboardEpisodes, setStoryboardEpisodes] = useState<StoryboardEpisode[]>([]);
  const [storyboardSceneCount, setStoryboardSceneCount] = useState(0);
  const [storyboardItemCount, setStoryboardItemCount] = useState(0);
  const [loadingStoryboard, setLoadingStoryboard] = useState(true);
  const [deletingStoryboard, setDeletingStoryboard] = useState(false);

  // 画风预设
  const [artPresets, setArtPresets] = useState<ArtStylePreset[]>([]);

  useEffect(() => {
    artStyleApi.getPresets().then(setArtPresets).catch(console.error);
  }, []);

  // 资产状态
  const [assets, setAssets] = useState<Asset[]>([]);
  const [assetCounts, setAssetCounts] = useState<Record<string, number>>({});
  const [loadingAssets, setLoadingAssets] = useState(true);

  // 串行加载所有概览数据：剧本 → 分镜 → 集 → 场次/镜头 → 资产
  const loadAllData = useCallback(async () => {
    // 1. 加载剧本
    try {
      setLoading(true);
      const scripts = await scriptApi.list(projectId);
      setScript(scripts.length > 0 ? scripts[0] : null);
    } catch (err) {
      console.error("加载剧本数据失败:", err);
    } finally {
      setLoading(false);
    }

    // 2. 加载分镜
    try {
      setLoadingStoryboard(true);
      const list = await storyboardApi.list(projectId);
      if (list.length > 0) {
        const sb = list[0];
        setStoryboard(sb);

        // 3. 加载集
        const episodes = await storyboardApi.listEpisodes(sb.id);
        setStoryboardEpisodes(episodes);

        // 4. 加载场次和镜头
        const [scenes, items] = await Promise.all([
          storyboardApi.listScenesByStoryboard(sb.id),
          storyboardApi.listItems(sb.id),
        ]);
        setStoryboardSceneCount(scenes.length);
        setStoryboardItemCount(items.length);
      } else {
        setStoryboard(null);
        setStoryboardEpisodes([]);
        setStoryboardSceneCount(0);
        setStoryboardItemCount(0);
      }
    } catch (err) {
      console.error("加载分镜数据失败:", err);
    } finally {
      setLoadingStoryboard(false);
    }

    // 3. 加载资产
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
    } finally {
      setLoadingAssets(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  const handleDeleteScript = async () => {
    if (!script) return;
    if (!confirm("确定要删除该剧本吗？所有分集和场次数据将一并删除。")) return;
    setDeletingScript(true);
    try {
      await scriptApi.delete(script.id);
      await loadAllData();
    } catch (err) {
      console.error("删除剧本失败:", err);
    } finally {
      setDeletingScript(false);
    }
  };

  const handleDeleteStoryboard = async () => {
    if (!storyboard) return;
    if (!confirm("确定要删除该分镜吗？所有分镜集、场次和镜头数据将一并删除。")) return;
    setDeletingStoryboard(true);
    try {
      await storyboardApi.delete(storyboard.id);
      await loadAllData();
    } catch (err) {
      console.error("删除分镜失败:", err);
    } finally {
      setDeletingStoryboard(false);
    }
  };

  // AI 生成剧本：创建成功后触发 pipeline
  const { addPipeline, setPanelExpanded, setExpandedTaskId } =
    usePipelineStore();

  const handleAiScriptCreated = (script: { id: number; title: string }) => {
    const scriptDisplayTitle = script.title?.trim() || project?.name?.trim() || "未命名项目";

    // 刷新列表
    loadAllData();

    // 启动 pipeline
    const pipelineId = addPipeline({
      label: `AI 生成剧本 - ${scriptDisplayTitle}`,
      projectId,
      request: {
        agentType: "script_full_parse",
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

    const scriptDisplayTitle = script.title?.trim() || project?.name?.trim() || "未命名项目";

    try {
      // 先创建分镜记录，获取 storyboardId
      const newStoryboard = await storyboardApi.create({
        projectId,
        scriptId: script.id,
        title: script.title?.trim() || project?.name?.trim() || "AI 分镜",
      });

      const pipelineId = addPipeline({
        label: `AI 生成分镜 - ${scriptDisplayTitle}`,
        projectId,
        request: {
          agentType: "script_to_storyboard",
          category: "pipeline",
          title: `AI 生成分镜：${scriptDisplayTitle}`,
          projectId,
          context: { scriptId: script.id, storyboardId: newStoryboard.id },
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
      console.error("创建分镜记录失败:", err);
      alert("创建分镜记录失败，请重试");
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

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* 项目详情概览条 */}
      <motion.div variants={itemVariants} className="mb-8">
        <div
          onClick={() => router.push(`/projects/${projectId}/settings`)}
          className={cn(
            "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
            "px-5 py-4 flex items-center gap-6 cursor-pointer group",
            "hover:border-primary/30 hover:bg-primary/2 transition-all duration-200"
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
        </div>
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
                    {script.title || "未命名剧本"}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-0.5 line-clamp-2">
                    {script.storySynopsis || script.genre || "暂无描述"}
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
              {script.totalEpisodes > 0 && (
                <span>{script.totalEpisodes} 集</span>
              )}
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatTime(script.updateTime)}
              </span>
            </div>

            {/* 操作按钮 */}
            <div className="flex items-center gap-2 pt-2 border-t border-border/20">
              <button
                onClick={() =>
                  router.push(`/projects/${projectId}/scripts`)
                }
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
                  "bg-primary text-primary-foreground",
                  "hover:opacity-90 transition-opacity"
                )}
              >
                <Eye className="h-4 w-4" />
                查看剧本
              </button>
              <button
                onClick={async () => {
                  if (!confirm("重新解析将删除当前剧本及其所有分集、场次数据，确定继续？")) return;
                  try {
                    await scriptApi.delete(script.id);
                    setParseMode("reparse");
                    setShowParseDialog(true);
                  } catch (err) {
                    console.error("删除旧剧本失败:", err);
                  }
                }}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
                  "bg-linear-to-r from-purple-600 to-pink-600",
                  "text-white shadow-lg shadow-purple-500/20",
                  "hover:shadow-purple-500/30 transition-all duration-200"
                )}
              >
                <RefreshCw className="h-4 w-4" />
                重新解析
              </button>
              <button
                onClick={handleDeleteScript}
                disabled={deletingScript}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
                  "border border-destructive/30 text-destructive",
                  "hover:bg-destructive/10 transition-colors ml-auto"
                )}
              >
                {deletingScript ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                删除剧本
              </button>
            </div>
          </div>
        ) : (
          /* 没有剧本：引导创建 */
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* 手动创建 */}
            <div
              onClick={() => setShowCreateDialog(true)}
              className={cn(
                "rounded-xl border border-dashed border-border/40 p-10",
                "flex flex-col items-center justify-center text-center",
                "bg-card/20 hover:border-primary/40 hover:bg-primary/5 transition-all cursor-pointer"
              )}
            >
              <div className="h-14 w-14 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                <Plus className="h-7 w-7 text-primary" />
              </div>
              <p className="text-lg font-medium mb-1">手动创建剧本</p>
              <p className="text-muted-foreground text-sm">
                创建空白剧本，手动添加分集和场次
              </p>
            </div>
            {/* AI 生成 */}
            <div
              onClick={() => {
                setParseMode("create");
                setShowParseDialog(true);
              }}
              className={cn(
                "rounded-xl border border-dashed border-border/40 p-10",
                "flex flex-col items-center justify-center text-center",
                "bg-card/20 hover:border-purple-500/40 hover:bg-purple-500/5 transition-all cursor-pointer"
              )}
            >
              <div className="h-14 w-14 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4">
                <Sparkles className="h-7 w-7 text-purple-400" />
              </div>
              <p className="text-lg font-medium mb-1">AI 生成剧本</p>
              <p className="text-muted-foreground text-sm">
                粘贴剧本原文，AI 将自动解析为结构化数据
              </p>
            </div>
          </div>
        )}

        <CreateScriptDialog
          open={showCreateDialog}
          projectId={projectId}
          onClose={() => setShowCreateDialog(false)}
          onCreated={() => loadAllData()}
        />
        <ParseScriptDialog
          open={showParseDialog}
          projectId={projectId}
          mode={parseMode}
          onClose={() => setShowParseDialog(false)}
          onCreated={handleAiScriptCreated}
        />
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
                    {storyboard.title || "未命名分镜"}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-0.5 line-clamp-2">
                    {storyboard.description || "暂无描述"}
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
              {storyboardEpisodes.length > 0 && (
                <span>{storyboardEpisodes.length} 集</span>
              )}
              <span>{storyboardSceneCount} 场次</span>
              <span>{storyboardItemCount} 镜头</span>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {formatTime(storyboard.updateTime)}
              </span>
            </div>

            {/* 操作按钮 */}
            <div className="flex items-center gap-2 pt-2 border-t border-border/20">
              <button
                onClick={() =>
                  router.push(`/projects/${projectId}/storyboards`)
                }
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
                  "bg-primary text-primary-foreground",
                  "hover:opacity-90 transition-opacity"
                )}
              >
                <Eye className="h-4 w-4" />
                查看分镜
              </button>
              <button
                onClick={async () => {
                  if (!script) {
                    alert("请先创建剧本后再生成分镜");
                    return;
                  }
                  if (!confirm("重新解析将删除当前分镜及其所有集、场次和镜头数据，确定继续？")) return;
                  try {
                    await storyboardApi.delete(storyboard!.id);
                    handleAiStoryboard();
                  } catch (err) {
                    console.error("删除旧分镜失败:", err);
                  }
                }}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
                  "bg-linear-to-r from-purple-600 to-pink-600",
                  "text-white shadow-lg shadow-purple-500/20",
                  "hover:shadow-purple-500/30 transition-all duration-200"
                )}
              >
                <RefreshCw className="h-4 w-4" />
                重新解析
              </button>
              <button
                onClick={handleDeleteStoryboard}
                disabled={deletingStoryboard}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium",
                  "border border-destructive/30 text-destructive",
                  "hover:bg-destructive/10 transition-colors ml-auto"
                )}
              >
                {deletingStoryboard ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                删除分镜
              </button>
            </div>
          </div>
        ) : (
          /* 没有分镜：引导 */
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* 手动创建 */}
            <div
              onClick={() =>
                router.push(`/projects/${projectId}/storyboards`)
              }
              className={cn(
                "rounded-xl border border-dashed border-border/40 p-10",
                "flex flex-col items-center justify-center text-center",
                "bg-card/20 hover:border-cyan-500/40 hover:bg-cyan-500/5 transition-all cursor-pointer"
              )}
            >
              <div className="h-14 w-14 rounded-xl bg-cyan-500/10 flex items-center justify-center mb-4">
                <Plus className="h-7 w-7 text-cyan-400" />
              </div>
              <p className="text-lg font-medium mb-1">创建分镜</p>
              <p className="text-muted-foreground text-sm">
                进入分镜管理页面，手动创建分镜表
              </p>
            </div>
            {/* AI 生成 */}
            <div
              onClick={() => {
                if (!script) {
                  alert("请先创建剧本后再使用 AI 生成分镜");
                  return;
                }
                handleAiStoryboard();
              }}
              className={cn(
                "rounded-xl border border-dashed border-border/40 p-10",
                "flex flex-col items-center justify-center text-center",
                "bg-card/20 hover:border-purple-500/40 hover:bg-purple-500/5 transition-all cursor-pointer"
              )}
            >
              <div className="h-14 w-14 rounded-xl bg-purple-500/10 flex items-center justify-center mb-4">
                <Sparkles className="h-7 w-7 text-purple-400" />
              </div>
              <p className="text-lg font-medium mb-1">AI 生成分镜</p>
              <p className="text-muted-foreground text-sm">
                基于剧本内容，AI 将自动生成结构化分镜表
              </p>
            </div>
          </div>
        )}
      </motion.div>

      {/* 资产概览区域 */}
      <motion.div variants={itemVariants} className="mt-10 pb-8">
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
                      <div className="h-10 w-10 rounded-lg overflow-hidden shrink-0 bg-muted/30">
                        {asset.coverUrl ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={resolveMediaUrl(asset.coverUrl) || ""}
                            alt={asset.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <AssetTypePlaceholder type={asset.type} className="w-full h-full" />
                        )}
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
  );
}
