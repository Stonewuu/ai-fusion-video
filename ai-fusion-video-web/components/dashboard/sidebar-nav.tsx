"use client";

import { useEffect, useId, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FolderKanban,
  BookOpen,
  Film,
  Images,
  Users,
  Settings,
  Bot,
  HardDrive,
  ImagePlus,
  Video,
  Workflow,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { SidebarHeading } from "./sidebar-heading";
import { projectApi, type Project } from "@/lib/api/project";
import { toastApiError } from "@/lib/api/toast-api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { usePrefersReducedMotion } from "@/hooks/use-prefers-reduced-motion";

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
];

const projectListItems: SidebarItem[] = [
  { key: "list", label: "项目列表", icon: FolderKanban, href: "/projects", iconColor: "text-purple-400" },
];

const projectDetailItems: SidebarItem[] = [
  { key: "", label: "概览", icon: LayoutDashboard, href: "", iconColor: "text-blue-400" },
  { key: "canvases", label: "画布", icon: Workflow, href: "/canvases", iconColor: "text-primary" },
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
  const navigationId = useId();
  const reducedMotion = usePrefersReducedMotion();
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
    if (projectProp !== undefined || !projectId) {
      return;
    }

    let cancelled = false;
    projectApi.get(projectId)
      .then((projectData) => {
        if (!cancelled) {
          setProjectLocalState({ id: projectId, project: projectData });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          toastApiError(error, "加载项目信息失败");
        }
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
  } else if (pathname.startsWith("/canvases")) {
    sectionTitle = "画布";
    items = [{ key: "canvases", label: "全部画布", icon: Workflow, href: "/canvases", iconColor: "text-primary" }];
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
      if (href === basePath) {
        return pathname === basePath;
      }
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
      data-dashboard-sidebar
      data-sidebar-navigation="global"
      className={cn(
        "w-full h-[calc(100vh-6rem)] rounded-2xl p-2",
        "bg-card [contain:paint]",
        "border border-border/40",
        "shadow-lg"
      )}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={sectionTitle}
          initial={{ opacity: 0, y: reducedMotion ? 0 : 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.2, ease: "easeOut" }}
        >
          <SidebarHeading
            title={sectionTitle}
            description={project?.description}
            collapsed={collapsed}
            navigationId={navigationId}
            onBack={backAction}
            onCollapsedChange={onCollapsedChange}
          />
          <nav id={navigationId} className={cn("flex flex-col gap-0.5", !collapsed && "px-1")}>
            {items.map((item) => {
              const Icon = item.icon;
              const isActive = getIsActive(item.href);

              return (
                <button
                  key={item.key}
                  onClick={() => handleNav(item.href)}
                  title={item.label}
                  aria-label={item.label}
                  aria-current={isActive ? "page" : undefined}
                  className={cn(
                    "flex h-9 w-full items-center rounded-xl text-sm leading-5",
                    "relative outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
                    collapsed
                      ? "justify-center"
                      : "gap-3 px-3",
                    isActive
                      ? "font-medium text-foreground bg-foreground/10"
                      : "text-muted-foreground hover:text-foreground hover:bg-foreground/6"
                  )}
                >
                  <Icon aria-hidden="true" className={cn("block size-4 shrink-0", isActive && item.iconColor)} />
                  {!collapsed && <span className="min-w-0 truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
