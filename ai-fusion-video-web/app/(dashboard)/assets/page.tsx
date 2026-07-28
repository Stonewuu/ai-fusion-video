"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  Grid3X3,
  List,
  Loader2,
  Package,
  Users,
  MapPin,
  Wrench,
  ExternalLink,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { assetApi, type Asset } from "@/lib/api/asset";
import { projectApi, type Project } from "@/lib/api/project";
import { resolveMediaUrl } from "@/lib/api/client";
import { toastApiError } from "@/lib/api/toast-api-error";
import AssetTypePlaceholder from "@/components/dashboard/asset-type-placeholder";
import { SafeImage } from "@/components/ui/safe-image";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// ============================================================
// 常量
// ============================================================

const PAGE_SIZE = 20;

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] as [number, number, number, number] },
  },
};

/** 资产类型配置 */
const ASSET_TYPES = [
  { value: "character", label: "角色", icon: Users, color: "text-blue-400", bg: "bg-blue-500/10" },
  { value: "scene", label: "场景", icon: MapPin, color: "text-green-400", bg: "bg-green-500/10" },
  { value: "prop", label: "道具", icon: Wrench, color: "text-amber-400", bg: "bg-amber-500/10" },
] as const;

const typeMap = Object.fromEntries(ASSET_TYPES.map((t) => [t.value, t]));

// ============================================================
// 页面
// ============================================================

export default function AssetsPage() {
  const router = useRouter();

  // 数据
  const [assets, setAssets] = useState<Asset[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [typeCounts, setTypeCounts] = useState<Record<string, number>>({});

  // 筛选
  const [selectedProjectId, setSelectedProjectId] = useState<string>("all");
  const [selectedType, setSelectedType] = useState<string>("all");
  const [keyword, setKeyword] = useState("");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  // 搜索防抖
  const [debouncedKeyword, setDebouncedKeyword] = useState("");
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => setDebouncedKeyword(keyword), 300);
    return () => { if (debounceTimer.current) clearTimeout(debounceTimer.current); };
  }, [keyword]);

  // 滚动加载 sentinel
  const sentinelRef = useRef<HTMLDivElement>(null);

  // 是否还有更多
  const hasMore = assets.length < total;

  // 项目名称映射
  const projectMap = useMemo(() => {
    const map: Record<number, string> = {};
    projects.forEach((p) => { map[p.id] = p.name; });
    return map;
  }, [projects]);

  // 总资产数 = typeCounts 之和
  const totalAssetCount = useMemo(() => {
    return Object.values(typeCounts).reduce((sum, c) => sum + c, 0);
  }, [typeCounts]);

  // 加载项目列表
  useEffect(() => {
    projectApi.list().then(setProjects).catch((error) => {
      toastApiError(error, "加载项目列表失败");
      setProjects([]);
    });
  }, []);

  // 加载第一页（筛选改变时重置）
  const loadFirstPage = useCallback(async () => {
    setLoading(true);
    setAssets([]);
    setCurrentPage(1);
    try {
      const params: { projectId?: number; type?: string; keyword?: string; page: number; size: number } = {
        page: 1,
        size: PAGE_SIZE,
      };
      if (selectedProjectId !== "all") params.projectId = Number(selectedProjectId);
      if (selectedType !== "all") params.type = selectedType;
      if (debouncedKeyword.trim()) params.keyword = debouncedKeyword.trim();
      const resp = await assetApi.listAll(params);
      setAssets(resp.records || []);
      setTotal(resp.total);
      setTypeCounts(resp.typeCounts || {});
      setCurrentPage(1);
    } catch (error) {
      toastApiError(error, "加载资产失败");
      setAssets([]);
      setTotal(0);
      setTypeCounts({});
    } finally {
      setLoading(false);
    }
  }, [selectedProjectId, selectedType, debouncedKeyword]);

  useEffect(() => {
    loadFirstPage();
  }, [loadFirstPage]);

  // 加载更多
  const loadMore = useCallback(async () => {
    if (loadingMore || !hasMore) return;
    setLoadingMore(true);
    try {
      const nextPage = currentPage + 1;
      const params: { projectId?: number; type?: string; keyword?: string; page: number; size: number } = {
        page: nextPage,
        size: PAGE_SIZE,
      };
      if (selectedProjectId !== "all") params.projectId = Number(selectedProjectId);
      if (selectedType !== "all") params.type = selectedType;
      if (debouncedKeyword.trim()) params.keyword = debouncedKeyword.trim();
      const resp = await assetApi.listAll(params);
      setAssets((prev) => [...prev, ...(resp.records || [])]);
      setTotal(resp.total);
      setCurrentPage(nextPage);
    } catch (error) {
      toastApiError(error, "加载更多资产失败");
    } finally {
      setLoadingMore(false);
    }
  }, [loadingMore, hasMore, currentPage, selectedProjectId, selectedType, debouncedKeyword]);

  // IntersectionObserver 滚动加载
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading && !loadingMore) {
          loadMore();
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loading, loadingMore, loadMore]);

  // 项目选择器选项
  const projectOptions = useMemo(() => [
    { value: "all", label: "全部项目" },
    ...projects.map((p) => ({ value: String(p.id), label: p.name })),
  ], [projects]);

  // 点击资产卡片
  const handleAssetClick = (asset: Asset) => {
    router.push(`/projects/${asset.projectId}/assets?highlight=${asset.id}`);
  };

  return (
    <motion.div
      className="max-w-[1200px]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ========== 页面标题 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">素材资产</h1>
        <p className="text-muted-foreground mt-1">
          跨项目查看和管理所有创作素材
        </p>
      </motion.div>

      {/* ========== 统计卡片 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {/* 总计 */}
          <div
            className={cn(
              "col-span-2 sm:col-span-1 rounded-xl border border-border/30 p-4",
              "bg-card/50 backdrop-blur-sm"
            )}
          >
            <div className="flex items-center gap-2.5">
              <div className="h-9 w-9 rounded-lg bg-foreground/5 flex items-center justify-center shrink-0">
                <Package className="h-4.5 w-4.5 text-foreground/60" />
              </div>
              <div>
                <p className="text-xl font-bold">{totalAssetCount}</p>
                <p className="text-[10px] text-muted-foreground">全部资产</p>
              </div>
            </div>
          </div>
          {/* 各类型 */}
          {ASSET_TYPES.map((t) => {
            const Icon = t.icon;
            const count = typeCounts[t.value] || 0;
            return (
              <button
                key={t.value}
                onClick={() => setSelectedType(selectedType === t.value ? "all" : t.value)}
                className={cn(
                  "rounded-xl border p-3.5 text-left transition-all",
                  "bg-card/50 backdrop-blur-sm",
                  selectedType === t.value
                    ? "border-border/60 ring-1 ring-border/30"
                    : "border-border/20 hover:border-border/40"
                )}
              >
                <div className="flex items-center gap-2">
                  <div className={cn("h-7 w-7 rounded-lg flex items-center justify-center shrink-0", t.bg)}>
                    <Icon className={cn("h-3.5 w-3.5", t.color)} />
                  </div>
                  <div className="min-w-0">
                    <p className="text-lg font-bold leading-none">{count}</p>
                    <p className="text-[10px] text-muted-foreground truncate">{t.label}</p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </motion.div>

      {/* ========== 筛选工具栏 ========== */}
      <motion.div variants={itemVariants} className="flex items-center gap-3 mb-6 flex-wrap">
        {/* 项目选择器 */}
        <div className="w-44 shrink-0">
          <Select
            value={selectedProjectId}
            onValueChange={(v) => setSelectedProjectId(v ?? "all")}
            items={projectOptions}
          >
            <SelectTrigger className="w-full text-xs">
              <SelectValue placeholder="全部项目" />
            </SelectTrigger>
            <SelectContent className="text-xs">
              <SelectGroup>
                {projectOptions.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value} className="text-xs">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>

        {/* 搜索框 */}
        <div
          className={cn(
            "flex-1 min-w-[200px] flex items-center gap-2.5 px-3.5 py-2 rounded-xl",
            "border border-border/30 bg-card/50 backdrop-blur-sm",
            "transition-[border-color,box-shadow] duration-150 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50 motion-reduce:transition-none"
          )}
        >
          <Search className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <input
            type="text"
            placeholder="搜索资产名称..."
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground/50"
          />
        </div>

        {/* 视图切换 */}
        <div className="flex rounded-xl border border-border/30 bg-card/50 overflow-hidden shrink-0">
          <button
            onClick={() => setViewMode("grid")}
            className={cn(
              "p-2.5 transition-colors",
              viewMode === "grid"
                ? "bg-white/10 text-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Grid3X3 className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={cn(
              "p-2.5 transition-colors",
              viewMode === "list"
                ? "bg-white/10 text-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <List className="h-4 w-4" />
          </button>
        </div>
      </motion.div>

      {/* ========== 内容区 ========== */}
      <motion.div variants={itemVariants}>
        {loading ? (
          /* 加载中 */
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <Loader2 className="h-8 w-8 animate-spin mb-3 text-muted-foreground/40" />
            <p className="text-sm">加载资产中...</p>
          </div>
        ) : assets.length === 0 ? (
          /* 空状态 */
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="h-16 w-16 rounded-2xl bg-orange-500/10 flex items-center justify-center mb-4">
              <Package className="h-8 w-8 text-orange-400/60" />
            </div>
            <h3 className="text-lg font-semibold mb-1">暂无资产</h3>
            <p className="text-sm text-muted-foreground max-w-sm">
              {debouncedKeyword || selectedType !== "all" || selectedProjectId !== "all"
                ? "没有找到匹配的资产，试试调整筛选条件"
                : "在项目中创建角色、场景、道具等资产后，这里会集中展示"}
            </p>
          </div>
        ) : viewMode === "grid" ? (
          /* 网格视图 */
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {assets.map((asset) => {
              const typeInfo = typeMap[asset.type];
              const TypeIcon = typeInfo?.icon || Package;
              const coverSrc = resolveMediaUrl(asset.coverUrl);

              return (
                <motion.div
                  key={asset.id}
                  whileHover={{ y: -4, transition: { duration: 0.2 } }}
                  onClick={() => handleAssetClick(asset)}
                  className={cn(
                    "group relative rounded-xl border border-border/30 overflow-hidden cursor-pointer",
                    "bg-card/50 backdrop-blur-sm",
                    "hover:border-border/50 hover:shadow-lg hover:shadow-black/5 transition-all duration-300"
                  )}
                >
                  {/* 封面 / 占位 */}
                  <div className="aspect-[4/3] relative overflow-hidden bg-muted/5">
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

                    {/* 类型标签 */}
                    <div
                      className={cn(
                        "absolute top-2 left-2 flex items-center gap-1 px-2 py-0.5 rounded-lg",
                        "bg-black/40 backdrop-blur-sm text-white/90 text-[10px] font-medium"
                      )}
                    >
                      <TypeIcon className="h-3 w-3" />
                      {typeInfo?.label || asset.type}
                    </div>

                    {/* 悬浮跳转提示 */}
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                      <div className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/20 backdrop-blur text-white text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                        <ExternalLink className="h-3 w-3" />
                        查看详情
                      </div>
                    </div>
                  </div>

                  {/* 信息 */}
                  <div className="p-3.5">
                    <p className="text-sm font-medium truncate mb-1">{asset.name}</p>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] text-muted-foreground truncate">
                        {projectMap[asset.projectId] || `项目 ${asset.projectId}`}
                      </span>
                      <span className="text-[10px] text-muted-foreground/60 whitespace-nowrap">
                        {formatDate(asset.updateTime)}
                      </span>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        ) : (
          /* 列表视图 */
          <div className="flex flex-col gap-2">
            {assets.map((asset) => {
              const typeInfo = typeMap[asset.type];
              const TypeIcon = typeInfo?.icon || Package;
              const coverSrc = resolveMediaUrl(asset.coverUrl);

              return (
                <div
                  key={asset.id}
                  onClick={() => handleAssetClick(asset)}
                  className={cn(
                    "group flex items-center gap-4 px-4 py-3 rounded-xl cursor-pointer",
                    "border border-border/30 bg-card/50",
                    "hover:border-border/50 hover:bg-card/80 transition-all"
                  )}
                >
                  {/* 缩略图 */}
                  <div className="h-12 w-12 rounded-lg overflow-hidden bg-muted/10 shrink-0">
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
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* 信息 */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{asset.name}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span
                        className={cn(
                          "px-1.5 py-0.5 rounded-md text-[10px] font-medium",
                          typeInfo?.bg || "bg-muted/10",
                          typeInfo?.color || "text-muted-foreground"
                        )}
                      >
                        {typeInfo?.label || asset.type}
                      </span>
                      <span className="text-xs text-muted-foreground truncate">
                        {projectMap[asset.projectId] || `项目 ${asset.projectId}`}
                      </span>
                    </div>
                  </div>

                  {/* 时间 */}
                  <span className="text-xs text-muted-foreground whitespace-nowrap shrink-0">
                    {formatDate(asset.updateTime)}
                  </span>

                  {/* 跳转指示 */}
                  <ExternalLink className="h-3.5 w-3.5 text-muted-foreground/30 group-hover:text-muted-foreground/70 transition-colors shrink-0" />
                </div>
              );
            })}
          </div>
        )}

        {/* 滚动加载 sentinel */}
        {!loading && hasMore && (
          <div ref={sentinelRef} className="flex items-center justify-center py-8">
            {loadingMore && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                加载更多...
              </div>
            )}
          </div>
        )}

        {/* 已加载全部 */}
        {!loading && !hasMore && assets.length > 0 && (
          <div className="flex items-center justify-center py-6">
            <p className="text-xs text-muted-foreground/40">
              已显示全部 {total} 个资产
            </p>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}

// ============================================================
// 工具函数
// ============================================================

function formatDate(dateStr: string): string {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  const diffHour = Math.floor(diffMs / 3600000);
  const diffDay = Math.floor(diffMs / 86400000);

  if (diffMin < 1) return "刚刚";
  if (diffMin < 60) return `${diffMin} 分钟前`;
  if (diffHour < 24) return `${diffHour} 小时前`;
  if (diffDay < 7) return `${diffDay} 天前`;
  return date.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}
