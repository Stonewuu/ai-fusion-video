"use client";

import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/store/auth-store";
import { projectApi, type Project } from "@/lib/api/project";
import { assetApi, type Asset, type AssetPageResp } from "@/lib/api/asset";
import {
  FolderKanban,
  Images,
  Sparkles,
  Clock,
  Film,
  Loader2,
  ArrowRight,
  Plus,
  Package,
  Zap,
  TrendingUp,
  Users,
  MapPin,
  Wrench,
  ChevronRight,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import AssetTypePlaceholder from "@/components/dashboard/asset-type-placeholder";
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

  useEffect(() => {
    (async () => {
      try {
        const [list, assets] = await Promise.all([
          projectApi.list(),
          assetApi.listAll({ page: 1, size: 10 }),
        ]);
        setProjects(list);
        setAssetData(assets);
      } catch (err) {
        console.error("加载仪表盘数据失败:", err);
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

  // 近 7 天活跃
  const activityDots = useMemo(() => {
    const now = new Date();
    const result: { label: string; count: number }[] = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      const label = date.toLocaleDateString("zh-CN", { weekday: "short" }).replace("周", "");
      const count = projects.filter(
        (p) => new Date(p.updateTime).toDateString() === date.toDateString()
      ).length;
      result.push({ label, count });
    }
    return result;
  }, [projects]);

  const maxActivity = Math.max(...activityDots.map((d) => d.count), 1);

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
      <motion.div variants={itemVariants} className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
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
        {/* 最近更新 */}
        <StatCard
          label="最近更新"
          value={recentProjects.length > 0 ? formatTime(recentProjects[0].updateTime) : "—"}
          icon={Clock}
          iconColor="text-purple-400"
          iconBg="bg-purple-500/10"
          small
        />
        {/* 活跃度 */}
        <div
          className={cn(
            "rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-4",
            "hover:border-border/50 transition-colors"
          )}
        >
          <div className="flex items-center justify-between mb-2.5">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-lg bg-green-500/10 flex items-center justify-center">
                <TrendingUp className="h-3.5 w-3.5 text-green-400" />
              </div>
              <span className="text-xs text-muted-foreground">近 7 天</span>
            </div>
          </div>
          <div className="flex items-end gap-[3px] h-7">
            {activityDots.map((dot, i) => (
              <div key={i} className="flex-1 flex flex-col items-center">
                <div
                  className={cn(
                    "w-full rounded-[2px]",
                    dot.count > 0 ? "bg-green-400/50" : "bg-border/20"
                  )}
                  style={{
                    height: dot.count > 0
                      ? `${Math.max(20, (dot.count / maxActivity) * 100)}%`
                      : "3px",
                  }}
                />
              </div>
            ))}
          </div>
          <div className="flex items-center gap-[3px] mt-1">
            {activityDots.map((dot, i) => (
              <span key={i} className="flex-1 text-center text-[8px] text-muted-foreground/30 leading-none">
                {dot.label}
              </span>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ========== 快捷操作 ========== */}
      <motion.div variants={itemVariants} className="grid grid-cols-3 gap-3 mb-8">
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
          icon={Sparkles}
          label="AI 助手"
          desc="智能辅助创作"
          color="text-purple-400"
          bg="bg-purple-500/10"
          onClick={() => router.push("/projects")}
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
  icon: typeof Film;
  label: string;
  desc: string;
  color: string;
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
        {coverSrc && (
          /* 底部渐变遮罩 */
          <div className="absolute inset-x-0 bottom-0 h-16 bg-linear-to-t from-black/70 to-transparent pointer-events-none" />
        )}

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
          <p className="text-[11px] font-medium text-white truncate">{asset.name}</p>
        </div>
      </div>
    </div>
  );
}
