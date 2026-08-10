"use client";

import { useEffect, useState, useMemo, type ComponentType } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth-store";
import { projectApi, type Project } from "@/lib/api/project";
import { assetApi, type Asset, type AssetPageResp } from "@/lib/api/asset";
import { aiModelApi, type AiModel } from "@/lib/api/ai-model";
import { storageConfigApi, type StorageConfig } from "@/lib/api/storage";
import { http, resolveMediaUrl } from "@/lib/api/client";
import {
  FolderKanban,
  Images,
  Film,
  Loader2,
  ArrowRight,
  Plus,
  Package,
  Users,
  MapPin,
  Wrench,
  ChevronRight,
  Cpu,
  Globe,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { toastApiError } from "@/lib/api/toast-api-error";
import { AssistantBrandIcon } from "@/components/dashboard/assistant/assistant-brand-icon";
import { requestAssistantOpen } from "@/components/dashboard/assistant/open-assistant";
import { SafeImage } from "@/components/ui/safe-image";

// ============================================================
// 动画
// ============================================================

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.08 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

// ============================================================
// 配置
// ============================================================

const ASSET_TYPE_CONFIG: Record<string, { label: string; icon: typeof Users; color: string; bg: string }> = {
  character: { label: "角色", icon: Users, color: "text-blue-400", bg: "bg-blue-500/10" },
  scene: { label: "场景", icon: MapPin, color: "text-green-400", bg: "bg-green-500/10" },
  prop: { label: "道具", icon: Wrench, color: "text-amber-400", bg: "bg-amber-500/10" },
};

function DashboardAssistantIcon({ className }: { className?: string }) {
  return <AssistantBrandIcon className={className} loading="eager" />;
}

// ============================================================
// 工具
// ============================================================

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
  return d.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}

function getGreeting() {
  const hour = new Date().getHours();
  if (hour < 6) return "夜深了";
  if (hour < 12) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}

// ============================================================
// 页面
// ============================================================

export default function DashboardPage() {
  const user = useAuthStore((s) => s.user);
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [assetData, setAssetData] = useState<AssetPageResp | null>(null);
  const [models, setModels] = useState<AiModel[]>([]);
  const [resourceBaseUrl, setResourceBaseUrl] = useState<string | null>(null);
  const [storageConfigs, setStorageConfigs] = useState<StorageConfig[]>([]);

  useEffect(() => {
    (async () => {
      try {
        const [list, assets, aiModels, resUrl, storages] = await Promise.all([
          projectApi.list(),
          assetApi.listAll({ page: 1, size: 10 }),
          aiModelApi.list().catch(() => []),
          http.get<never, string | null>("/api/system/config/resource_base_url").catch(() => null),
          storageConfigApi.list().catch(() => []),
        ]);
        setProjects(list);
        setAssetData(assets);
        setModels(aiModels);
        setResourceBaseUrl(resUrl);
        setStorageConfigs(storages);
      } catch (err) {
        console.error("加载仪表盘数据失败:", err);
        toastApiError(err, "加载仪表盘数据失败");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const recentProjects = useMemo(
    () =>
      [...projects]
        .sort((a, b) => new Date(b.updateTime).getTime() - new Date(a.updateTime).getTime())
        .slice(0, 6),
    [projects]
  );

  const totalAssetCount = useMemo(() => {
    if (!assetData?.typeCounts) return 0;
    return Object.values(assetData.typeCounts).reduce((s, c) => s + c, 0);
  }, [assetData]);

  const recentAssets = assetData?.records ?? [];

  // 模型配置状态 (modelType: 1-文本/对话, 2-图像生成, 3-视频生成)
  const hasTextModel = useMemo(
    () => models.some((m) => (m.status === 1 || m.status === undefined) && m.modelType === 1),
    [models]
  );
  const hasImageModel = useMemo(
    () => models.some((m) => (m.status === 1 || m.status === undefined) && m.modelType === 2),
    [models]
  );
  const hasVideoModel = useMemo(
    () => models.some((m) => (m.status === 1 || m.status === undefined) && m.modelType === 3),
    [models]
  );
  const configuredModelCount = useMemo(
    () => (hasTextModel ? 1 : 0) + (hasImageModel ? 1 : 0) + (hasVideoModel ? 1 : 0),
    [hasTextModel, hasImageModel, hasVideoModel]
  );

  // 外部访问就绪状态 (配置了公网 S3 或配置了公网访问地址 resourceBaseUrl)
  const hasPublicS3 = useMemo(
    () => storageConfigs.some((c) => c.isDefault && (c.status === 1 || c.status === undefined) && c.type === "s3"),
    [storageConfigs]
  );
  const hasResourceUrl = useMemo(() => Boolean(resourceBaseUrl?.trim()), [resourceBaseUrl]);
  const isExternalReady = useMemo(() => hasPublicS3 || hasResourceUrl, [hasPublicS3, hasResourceUrl]);



  if (loading) {
    return (
      <div className="flex items-center justify-center py-32">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <motion.div
      className="max-w-[1200px]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ========== 问候 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">
          {getGreeting()}，
          <span className="bg-linear-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            {user?.nickname || user?.username}
          </span>
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          欢迎回到融光，开始你的视频创作之旅
        </p>
      </motion.div>

      {/* ========== 统计 ========== */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {/* 项目 */}
        <StatCard
          label="项目"
          value={String(projects.length)}
          icon={Film}
          iconColor="text-blue-400"
          iconBg="bg-blue-500/10"
          onClick={() => router.push("/projects")}
        />
        {/* 资产 */}
        <StatCard
          label="素材资产"
          value={String(totalAssetCount)}
          icon={Package}
          iconColor="text-orange-400"
          iconBg="bg-orange-500/10"
          onClick={() => router.push("/assets")}
        />
        {/* 模型配置状态 */}
        <div
          onClick={() => router.push("/settings/ai-models")}
          className={cn(
            "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-4",
            "hover:border-border/50 transition-colors cursor-pointer flex flex-col justify-between"
          )}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-lg bg-purple-500/10 flex items-center justify-center">
                <Cpu className="h-3.5 w-3.5 text-purple-400" />
              </div>
              <span className="text-xs text-muted-foreground">模型配置状态</span>
            </div>
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded-full font-medium border",
                configuredModelCount === 3
                  ? "bg-green-500/10 text-green-400 border-green-500/20"
                  : "bg-amber-500/10 text-amber-400 border-amber-500/20"
              )}
            >
              {configuredModelCount}/3 已配置
            </span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 mt-2.5">
            <div className="flex flex-col items-center py-1 px-1 rounded-md bg-background/40 border border-border/15">
              <span className="text-[10px] text-muted-foreground/70 mb-0.5">文本</span>
              <span className={cn("text-[11px] font-medium", hasTextModel ? "text-green-400" : "text-muted-foreground/40")}>
                {hasTextModel ? "已配置" : "未配置"}
              </span>
            </div>
            <div className="flex flex-col items-center py-1 px-1 rounded-md bg-background/40 border border-border/15">
              <span className="text-[10px] text-muted-foreground/70 mb-0.5">图片</span>
              <span className={cn("text-[11px] font-medium", hasImageModel ? "text-green-400" : "text-muted-foreground/40")}>
                {hasImageModel ? "已配置" : "未配置"}
              </span>
            </div>
            <div className="flex flex-col items-center py-1 px-1 rounded-md bg-background/40 border border-border/15">
              <span className="text-[10px] text-muted-foreground/70 mb-0.5">视频</span>
              <span className={cn("text-[11px] font-medium", hasVideoModel ? "text-green-400" : "text-muted-foreground/40")}>
                {hasVideoModel ? "已配置" : "未配置"}
              </span>
            </div>
          </div>
        </div>
        {/* 外部访问就绪状态 */}
        <div
          onClick={() => router.push("/settings/storage")}
          className={cn(
            "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-4",
            "hover:border-border/50 transition-colors cursor-pointer flex flex-col justify-between"
          )}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                <Globe className="h-3.5 w-3.5 text-emerald-400" />
              </div>
              <span className="text-xs text-muted-foreground">外部访问就绪</span>
            </div>
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded-full font-medium border",
                isExternalReady
                  ? "bg-green-500/10 text-green-400 border-green-500/20"
                  : "bg-amber-500/10 text-amber-400 border-amber-500/20"
              )}
            >
              {isExternalReady ? "已就绪" : "未就绪"}
            </span>
          </div>
          <div className="mt-2.5">
            <p className={cn("text-base font-bold tracking-tight", isExternalReady ? "text-foreground" : "text-amber-400/90")}>
              {isExternalReady ? "服务就绪" : "未就绪"}
            </p>
            <p className="text-[11px] text-muted-foreground/70 mt-0.5 truncate">
              {hasPublicS3 && hasResourceUrl
                ? "已配置公网 S3 & 访问域名"
                : hasPublicS3
                ? "已配置公网 S3 存储"
                : hasResourceUrl
                ? "已配置公网访问地址"
                : "需配置公网 S3 或访问地址"}
            </p>
          </div>
        </div>
      </motion.div>

      {/* ========== 快捷操作 ========== */}
      <motion.div
        variants={itemVariants}
        className="mb-8 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4"
      >
        <QuickAction
          icon={FolderKanban}
          label="新建项目"
          desc="开始全新的创作"
          color="text-blue-400"
          bg="bg-blue-500/10"
          onClick={() => router.push("/projects")}
        />
        <QuickAction
          icon={Images}
          label="管理素材"
          desc="查看所有创作资产"
          color="text-orange-400"
          bg="bg-orange-500/10"
          onClick={() => router.push("/assets")}
        />
        <QuickAction
          icon={DashboardAssistantIcon}
          label="融光助手"
          desc="智能辅助创作"
          bg="bg-purple-500/10"
          onClick={requestAssistantOpen}
        />
        <QuickAction
          icon={Wrench}
          label="工具"
          desc="使用 AI 创作工具"
          color="text-cyan-400"
          bg="bg-cyan-500/10"
          onClick={() => router.push("/generate/image")}
        />
      </motion.div>

      {/* ========== 最近项目 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <SectionHeader
          title="最近项目"
          icon={<Film className="h-4 w-4 text-primary" />}
          action={projects.length > 0 ? { label: "全部项目", onClick: () => router.push("/projects") } : undefined}
        />

        {recentProjects.length === 0 ? (
          <div
            onClick={() => router.push("/projects")}
            className={cn(
              "rounded-xl border border-dashed border-border/40 p-10",
              "flex flex-col items-center justify-center text-center",
              "bg-card/20 cursor-pointer hover:border-primary/40 hover:bg-primary/5 transition-all"
            )}
          >
            <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-3">
              <Plus className="h-6 w-6 text-primary/60" />
            </div>
            <p className="text-sm font-medium mb-0.5">创建你的第一个项目</p>
            <p className="text-xs text-muted-foreground">
              点击此处开始创建项目
            </p>
          </div>
        ) : (
          <div className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm overflow-hidden divide-y divide-border/15">
            {recentProjects.map((proj) => (
              <div
                key={proj.id}
                onClick={() => router.push(`/projects/${proj.id}`)}
                className="group flex items-center gap-4 px-4 py-3.5 cursor-pointer hover:bg-muted/20 transition-colors"
              >
                <div className="h-9 w-9 rounded-lg bg-primary/8 flex items-center justify-center shrink-0 group-hover:bg-primary/12 transition-colors">
                  <Film className="h-4.5 w-4.5 text-primary/70" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate group-hover:text-primary transition-colors">
                    {proj.name}
                  </p>
                  <p className="text-xs text-muted-foreground/60 truncate mt-0.5">
                    {proj.description || "暂无描述"}
                  </p>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-[11px] text-muted-foreground/40 tabular-nums">
                    {formatTime(proj.updateTime)}
                  </span>
                  <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/20 group-hover:text-muted-foreground/60 transition-colors" />
                </div>
              </div>
            ))}
          </div>
        )}
      </motion.div>

      {/* ========== 最近资产 ========== */}
      {recentAssets.length > 0 && (
        <motion.div variants={itemVariants}>
          <SectionHeader
            title="最近资产"
            icon={<Images className="h-4 w-4 text-orange-400" />}
            action={{ label: "全部资产", onClick: () => router.push("/assets") }}
          />

          <div className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1 scrollbar-none">
            {recentAssets.map((asset) => (
              <AssetCard key={asset.id} asset={asset} onClick={() =>
                router.push(`/projects/${asset.projectId}/assets?highlight=${asset.id}`)
              } />
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

// ============================================================
// 子组件
// ============================================================

/** 统计卡片 */
function StatCard({
  label,
  value,
  icon: Icon,
  iconColor,
  iconBg,
  onClick,
  small,
}: {
  label: string;
  value: string;
  icon: typeof Film;
  iconColor: string;
  iconBg: string;
  onClick?: () => void;
  small?: boolean;
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-4",
        "hover:border-border/50 transition-colors",
        onClick && "cursor-pointer"
      )}
    >
      <div className="flex items-center gap-2 mb-2.5">
        <div className={cn("h-7 w-7 rounded-lg flex items-center justify-center", iconBg)}>
          <Icon className={cn("h-3.5 w-3.5", iconColor)} />
        </div>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className={cn("font-bold tracking-tight", small ? "text-base" : "text-2xl")}>{value}</p>
    </div>
  );
}

/** 快捷操作 */
function QuickAction({
  icon: Icon,
  label,
  desc,
  color,
  bg,
  onClick,
}: {
  icon: ComponentType<{ className?: string }>;
  label: string;
  desc: string;
  color?: string;
  bg: string;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "group rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
        "px-4 py-3.5 cursor-pointer",
        "hover:border-border/50 hover:bg-card/80 transition-all"
      )}
    >
      <div className="flex items-center gap-3">
        <div className={cn("h-8 w-8 rounded-lg flex items-center justify-center shrink-0", bg,
          "group-hover:scale-105 transition-transform duration-200"
        )}>
          <Icon className={cn("h-4 w-4", color)} />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium truncate">{label}</p>
          <p className="text-[11px] text-muted-foreground/50 truncate">{desc}</p>
        </div>
      </div>
    </div>
  );
}

/** Section 头 */
function SectionHeader({
  title,
  icon,
  action,
}: {
  title: string;
  icon: React.ReactNode;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <div className="flex items-center justify-between mb-3">
      <h2 className="text-sm font-semibold flex items-center gap-2 text-foreground/80">
        {icon}
        {title}
      </h2>
      {action && (
        <button
          onClick={action.onClick}
          className="text-xs text-muted-foreground/50 hover:text-foreground flex items-center gap-0.5 transition-colors"
        >
          {action.label}
          <ArrowRight className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}

/** 资产卡片 */
function AssetCard({ asset, onClick }: { asset: Asset; onClick: () => void }) {
  const coverSrc = resolveMediaUrl(asset.coverUrl);
  const typeConf = ASSET_TYPE_CONFIG[asset.type];
  const TypeIcon = typeConf?.icon || Package;

  return (
    <div
      onClick={onClick}
      className={cn(
        "group rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm",
        "w-36 shrink-0 cursor-pointer overflow-hidden",
        "hover:border-border/50 hover:shadow-lg hover:shadow-black/5 transition-all duration-200"
      )}
    >
      <div className="aspect-[3/4] relative overflow-hidden bg-muted/5">
        <SafeImage
          src={coverSrc}
          fallbackType={
            asset.type === "character"
              ? "avatar"
              : asset.type === "scene"
              ? "scene"
              : asset.type === "prop"
              ? "prop"
              : "image"
          }
          alt={asset.name}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />
        {/* 主题自适应遮罩，确保封面与占位图上的名称始终清晰 */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-linear-to-t from-background/95 via-background/70 to-transparent" />

        {/* 类型标签 */}
        <div className={cn(
          "absolute top-1.5 left-1.5 flex items-center gap-0.5 px-1.5 py-0.5 rounded-md",
          "bg-black/40 backdrop-blur-sm text-white/80 text-[9px] font-medium"
        )}>
          <TypeIcon className="h-2.5 w-2.5" />
          {typeConf?.label || asset.type}
        </div>

        {/* 名称 */}
        <div className="absolute bottom-0 inset-x-0 px-2.5 pb-2">
          <p className="truncate text-[11px] font-semibold text-foreground">{asset.name}</p>
        </div>
      </div>
    </div>
  );
}
