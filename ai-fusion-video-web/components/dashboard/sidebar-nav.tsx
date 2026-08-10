"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  LayoutDashboard,
  BarChart3,
  FolderKanban,
  BookOpen,
  Film,
  Images,
  Users,
  Settings,
  ArrowLeft,
  Bot,
  HardDrive,
  ImagePlus,
  Video,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { projectApi, type Project } from "@/lib/api/project";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useAuthStore } from "@/lib/store/auth-store";

// ========== 各模块的二级菜单配置 ==========

interface SidebarItem {
  key: string;
  label: string;
  icon: typeof LayoutDashboard;
  href: string;
  iconColor: string;
}

const dashboardItems: SidebarItem[] = [
  { key: "overview", label: "总览", icon: LayoutDashboard, href: "/dashboard", iconColor: "text-blue-400" },
  { key: "analytics", label: "数据分析", icon: BarChart3, href: "/dashboard/analytics", iconColor: "text-purple-400" },
];

const projectListItems: SidebarItem[] = [
  { key: "list", label: "项目列表", icon: FolderKanban, href: "/projects", iconColor: "text-purple-400" },
];

const projectDetailItems: SidebarItem[] = [
  { key: "", label: "概览", icon: LayoutDashboard, href: "", iconColor: "text-blue-400" },
  { key: "scripts", label: "剧本", icon: BookOpen, href: "/scripts", iconColor: "text-purple-400" },
  { key: "storyboards", label: "分镜", icon: Film, href: "/storyboards", iconColor: "text-cyan-400" },
  { key: "assets", label: "资产", icon: Images, href: "/assets", iconColor: "text-orange-400" },
  // { key: "members", label: "成员", icon: Users, href: "/members", iconColor: "text-green-400" },
  { key: "settings", label: "设置", icon: Settings, href: "/settings", iconColor: "text-rose-400" },
];

const assetItems: SidebarItem[] = [
  { key: "list", label: "全部资产", icon: Images, href: "/assets", iconColor: "text-orange-400" },
];

const generationItems: SidebarItem[] = [
  { key: "image", label: "生图", icon: ImagePlus, href: "/generate/image", iconColor: "text-fuchsia-400" },
  { key: "video", label: "生视频", icon: Video, href: "/generate/video", iconColor: "text-cyan-400" },
];

interface SidebarNavProps {
  /** 导航后回调，移动端抽屉用于关闭面板 */
  onNavigate?: () => void;
  /** 当前项目，未传入时组件会按路由自行加载 */
  project?: Project | null;
  /** 桌面端是否以图标栏形态收起 */
  collapsed?: boolean;
  /** 切换桌面端侧栏收起状态 */
  onCollapsedChange?: (collapsed: boolean) => void;
}

// ========== 侧边栏组件 ==========

export function SidebarNav({
  onNavigate,
  project: projectProp,
  collapsed = false,
  onCollapsedChange,
}: SidebarNavProps) {
  const router = useRouter();
  const pathname = usePathname();
  const currentUser = useAuthStore((state) => state.user);
  const isAdmin = currentUser?.roles?.includes("admin") ?? false;

  const projectMatch = pathname.match(/^\/projects\/(\d+)/);
  const projectId = projectMatch ? Number(projectMatch[1]) : null;
  const [projectLocalState, setProjectLocalState] = useState<{
    id: number;
    project: Project;
  } | null>(null);

  const settingsItems: SidebarItem[] = isAdmin
    ? [
      { key: "general", label: "通用设置", icon: Settings, href: "/settings/general", iconColor: "text-green-400" },
      { key: "users", label: "用户列表", icon: Users, href: "/settings/users", iconColor: "text-cyan-400" },
      { key: "profile", label: "个人设置", icon: Users, href: "/settings/profile", iconColor: "text-blue-400" },
      { key: "ai-models", label: "AI 配置", icon: Bot, href: "/settings/ai-models", iconColor: "text-purple-400" },
      { key: "agents", label: "智能体配置", icon: Bot, href: "/settings/agents", iconColor: "text-violet-400" },
      { key: "storage", label: "存储配置", icon: HardDrive, href: "/settings/storage", iconColor: "text-orange-400" },
    ]
    : [
      { key: "profile", label: "个人设置", icon: Users, href: "/settings/profile", iconColor: "text-blue-400" },
      { key: "agents", label: "智能体配置", icon: Bot, href: "/settings/agents", iconColor: "text-violet-400" },
    ];

  // 若外部已传入 project，则不在组件内自行请求
  const project =
    projectProp !== undefined
      ? projectProp
      : projectLocalState?.id === projectId
        ? projectLocalState.project
        : null;

  useEffect(() => {
    if (projectProp !== undefined) return; // 由外部管理，跳过
    if (!projectId) return;

    let cancelled = false;
    projectApi.get(projectId)
      .then((projectData) => {
        if (!cancelled) {
          setProjectLocalState({ id: projectId, project: projectData });
        }
      })
      .catch((error) => {
        if (!cancelled) toastApiError(error, "加载项目信息失败");
      });

    return () => {
      cancelled = true;
    };
  }, [projectId, projectProp]);

  let items: SidebarItem[] = [];
  let sectionTitle = "";
  let backAction: (() => void) | null = null;

  if (projectId) {
    sectionTitle = project?.name || "加载中...";
    items = projectDetailItems.map((item) => ({
      ...item,
      href: `/projects/${projectId}${item.href}`,
    }));
    backAction = () => {
      router.push("/projects");
      onNavigate?.();
    };
  } else if (pathname.startsWith("/dashboard")) {
    sectionTitle = "仪表盘";
    items = dashboardItems;
  } else if (pathname.startsWith("/projects")) {
    sectionTitle = "项目";
    items = projectListItems;
  } else if (pathname.startsWith("/assets")) {
    sectionTitle = "资产";
    items = assetItems;
  } else if (pathname.startsWith("/generate")) {
    sectionTitle = "工具";
    items = generationItems;
  } else if (pathname.startsWith("/settings")) {
    sectionTitle = "系统设置";
    items = settingsItems;
  }

  const getIsActive = (href: string) => {
    if (projectId) {
      const basePath = `/projects/${projectId}`;
      if (href === basePath) return pathname === basePath;
      return pathname.startsWith(href);
    }
    return pathname === href;
  };

  const handleNav = (href: string) => {
    router.push(href);
    onNavigate?.();
  };

  return (
    <div
      className={cn(
        "w-full h-[calc(100vh-6rem)] rounded-2xl p-2",
        "bg-linear-to-b from-background/80 to-background/40",
        "backdrop-blur-lg border border-border/40",
        "shadow-lg"
      )}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={sectionTitle}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          {/* 标题区 */}
          {collapsed ? (
            <div className="px-1 pt-2 pb-3 flex flex-col items-center gap-2">
              {onCollapsedChange && (
                <button
                  type="button"
                  onClick={() => onCollapsedChange(false)}
                  className="h-10 w-10 rounded-xl flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-primary/8 transition-colors"
                  title="展开菜单栏"
                  aria-label="展开菜单栏"
                >
                  <PanelLeftOpen className="h-4 w-4" />
                </button>
              )}
              {backAction && (
                <button
                  type="button"
                  onClick={backAction}
                  className="h-10 w-10 rounded-xl flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-foreground/5 transition-colors"
                  title="返回项目列表"
                  aria-label="返回项目列表"
                >
                  <ArrowLeft className="h-4 w-4" />
                </button>
              )}
            </div>
          ) : (
            <div className="px-3 pt-3 pb-3">
              {backAction && (
                <>
                  <button
                    onClick={backAction}
                    className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors mb-2 group"
                  >
                    <ArrowLeft className="h-3.5 w-3.5 group-hover:-translate-x-0.5 transition-transform" />
                    返回项目列表
                  </button>
                  <div className="border-b border-border/30 -mx-3 mb-2" />
                </>
              )}
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold truncate">
                    {projectId && "项目: "}
                    {sectionTitle}
                  </h3>
                  {projectId && project?.description && (
                    <p className="text-xs text-muted-foreground/60 mt-0.5 line-clamp-1">
                      {project.description}
                    </p>
                  )}
                </div>
                {onCollapsedChange && (
                  <button
                    type="button"
                    onClick={() => onCollapsedChange(true)}
                    className="h-7 w-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-primary/8 transition-colors shrink-0"
                    title="收起菜单栏"
                    aria-label="收起菜单栏"
                  >
                    <PanelLeftClose className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>
          )}

          {/* 导航项 */}
          <nav className={cn("space-y-0.5", collapsed ? "px-0" : "px-1")}>
            {items.map((item) => {
              const Icon = item.icon;
              const isActive = getIsActive(item.href);

              return (
                <button
                  key={item.key}
                  onClick={() => handleNav(item.href)}
                  title={item.label}
                  aria-label={item.label}
                  className={cn(
                    "w-full flex items-center rounded-xl text-sm",
                    "transition-all duration-150 relative",
                    collapsed
                      ? "h-10 justify-center px-0"
                      : "gap-2.5 px-3 py-2",
                    isActive
                      ? "font-medium text-foreground bg-foreground/6"
                      : "text-muted-foreground hover:text-foreground hover:bg-foreground/3"
                  )}
                >
                  <span className={cn("transition-colors", isActive ? item.iconColor : "")}>
                    <Icon className="h-[18px] w-[18px]" />
                  </span>
                  {!collapsed && <span>{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
