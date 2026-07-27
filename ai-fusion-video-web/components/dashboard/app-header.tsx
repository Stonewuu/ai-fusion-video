"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import {
  LayoutDashboard,
  FolderKanban,
  Images,
  Wrench,
  Settings,
  Bell,
  Github,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { MenuBar, MobileMenuPanel } from "@/components/ui/glow-menu";
import type { MenuDisplayMode } from "@/components/ui/glow-menu";
import { AnimatedThemeToggler } from "@/components/ui/animated-theme-toggler";
import { UserAvatarDropdown } from "@/components/ui/menu";
import { NotificationPanel } from "@/components/dashboard/notification-panel";
import { useAuthStore } from "@/lib/store/auth-store";
import { usePipelineStore } from "@/lib/store/pipeline-store";
import { cn } from "@/lib/utils";

// 一级菜单项配置
const menuItems = [
  {
    icon: LayoutDashboard,
    label: "仪表盘",
    href: "/dashboard",
    gradient:
      "radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(37,99,235,0.06) 50%, rgba(29,78,216,0) 85%, rgba(29,78,216,0) 100%)",
    iconColor: "text-blue-500",
  },
  {
    icon: FolderKanban,
    label: "项目",
    href: "/projects",
    gradient:
      "radial-gradient(circle, rgba(168,85,247,0.15) 0%, rgba(147,51,234,0.06) 50%, rgba(126,34,206,0) 85%, rgba(126,34,206,0) 100%)",
    iconColor: "text-purple-500",
  },
  {
    icon: Images,
    label: "资产",
    href: "/assets",
    gradient:
      "radial-gradient(circle, rgba(249,115,22,0.15) 0%, rgba(234,88,12,0.06) 50%, rgba(194,65,12,0) 85%, rgba(194,65,12,0) 100%)",
    iconColor: "text-orange-500",
  },
  {
    icon: Wrench,
    label: "工具",
    href: "/generate/image",
    gradient:
      "radial-gradient(circle, rgba(6,182,212,0.15) 0%, rgba(8,145,178,0.06) 50%, rgba(14,116,144,0) 85%, rgba(14,116,144,0) 100%)",
    iconColor: "text-cyan-500",
  },
  {
    icon: Settings,
    label: "系统设置",
    href: "/settings",
    gradient:
      "radial-gradient(circle, rgba(34,197,94,0.15) 0%, rgba(22,163,74,0.06) 50%, rgba(21,128,61,0) 85%, rgba(21,128,61,0) 100%)",
    iconColor: "text-green-500",
  },
];

// 路由与菜单标签的映射关系
const routeToLabel: Record<string, string> = {
  "/dashboard": "仪表盘",
  "/projects": "项目",
  "/assets": "资产",
  "/generate": "工具",
  "/settings": "系统设置",
};

/**
 * 顶部浮动导航栏组件
 * 三级响应式适配：图标+文字 → 纯图标 → 汉堡菜单
 */
export function AppHeader() {
  const router = useRouter();
  const pathname = usePathname();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const restoreRunningPipelines = usePipelineStore(
    (s) => s.restoreRunningPipelines
  );
  const [mobileMenuRoute, setMobileMenuRoute] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<MenuDisplayMode>("full");
  const headerRef = useRef<HTMLDivElement>(null);
  const bellRef = useRef<HTMLButtonElement>(null);
  const mobileMenuOpen = displayMode === "mobile" && mobileMenuRoute === pathname;

  // Pipeline 通知
  const { tasks, setNotificationOpen, panelExpanded, setPanelExpanded } = usePipelineStore();
  const runningCount = tasks.filter((t) => t.status === "running").length;
  const hasAnyTasks = tasks.length > 0;

  // 根据当前路由确定高亮菜单项
  const activeLabel =
    Object.entries(routeToLabel).find(([route]) =>
      pathname.startsWith(route)
    )?.[1] || "仪表盘";

  // 页面加载时只恢复 running Pipeline 列表；SSE 由面板选中态管理。
  useEffect(() => {
    restoreRunningPipelines();
  }, [restoreRunningPipelines]);

  // 点击外部区域关闭移动端菜单
  useEffect(() => {
    if (!mobileMenuOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (headerRef.current && !headerRef.current.contains(e.target as Node)) {
        setMobileMenuRoute(null);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [mobileMenuOpen]);

  // 处理导航菜单点击
  const handleMenuClick = (label: string) => {
    const item = menuItems.find((m) => m.label === label);
    if (item) {
      router.push(item.href);
      setMobileMenuRoute(null);
    }
  };

  // 处理退出登录（layout 会在退出动画完成后自动跳转到登录页）
  const handleLogout = async () => {
    // 等待下拉菜单关闭动画（~200ms）
    await new Promise((resolve) => setTimeout(resolve, 200));
    document.cookie = "auth-token=; path=/; max-age=0";
    await logout();
  };

  // 稳定引用的模式变化回调
  const handleDisplayModeChange = useCallback((mode: MenuDisplayMode) => {
    setDisplayMode(mode);
    if (mode !== "mobile") {
      setMobileMenuRoute(null);
    }
  }, []);

  // 用户头像下拉菜单项
  const userMenuItems = [
    {
      label: "个人设置",
      icon: <Settings className="h-full w-full" />,
      onClick: () => router.push("/settings/profile"),
    },
  ];

  const userLogoutItem = {
    label: "退出登录",
    icon: <LogOut className="h-full w-full" />,
    onClick: handleLogout,
  };

  return (
    <header ref={headerRef} className="fixed top-0 left-0 right-0 z-50 px-3 pt-3">
      <MenuBar
        items={menuItems}
        activeItem={activeLabel}
        onItemClick={handleMenuClick}
        onDisplayModeChange={handleDisplayModeChange}
        className="w-full"
        leftContent={
          <div
            className="flex items-center cursor-pointer shrink-0"
            onClick={() => router.push("/dashboard")}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/logo.png"
              alt="融光"
              className="h-8 w-auto ml-2 rounded-lg"
            />
            <span className="ml-2 text-lg font-bold text-foreground">融光</span>
          </div>
        }
        mobileControls={
          <button
            onClick={() => setMobileMenuRoute(mobileMenuOpen ? null : pathname)}
            className={cn(
              "p-2 rounded-xl transition-all duration-200",
              "text-muted-foreground hover:text-foreground",
              mobileMenuOpen
                ? "bg-foreground/10 text-foreground"
                : "hover:bg-foreground/5"
            )}
            aria-label={mobileMenuOpen ? "关闭菜单" : "打开菜单"}
          >
            {mobileMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </button>
        }
        rightContent={
          <div className="flex items-center gap-2 justify-end">
            {/* 主题切换按钮 */}
            <AnimatedThemeToggler className="rounded-xl text-violet-500 hover:text-violet-600 hover:bg-violet-500/10 dark:text-violet-300 dark:hover:text-violet-200 dark:hover:bg-violet-400/10 transition-colors" />

            {/* 通知按钮 */}
            <button
              ref={bellRef}
              onClick={() => {
                if (panelExpanded) {
                  setPanelExpanded(false);
                  setNotificationOpen(false);
                } else {
                  setPanelExpanded(true);
                }
              }}
              className={cn(
                "relative p-2 rounded-xl transition-colors",
                "text-amber-500 hover:text-amber-600 hover:bg-amber-500/10 dark:text-amber-300 dark:hover:text-amber-200 dark:hover:bg-amber-400/10",
                panelExpanded
                  ? "bg-amber-500/15 text-amber-600 dark:bg-amber-400/15 dark:text-amber-200"
                  : ""
              )}
            >
              <Bell className="h-5 w-5" />
              {(runningCount > 0 || hasAnyTasks) && (
                <>
                  {/* 扩散光圈：仅运行中时显示 */}
                  {runningCount > 0 && (
                    <span className="absolute top-1 right-1 h-4 w-4 rounded-full bg-blue-500/40 animate-ping" />
                  )}
                  {/* 静态数字徽标 */}
                  <span
                    className={cn(
                      "absolute top-1 right-1 flex items-center justify-center rounded-full ring-2 ring-background/60",
                      runningCount > 0
                        ? "h-4 w-4 bg-blue-500 text-[9px] text-white font-bold"
                        : "h-2 w-2 bg-muted-foreground/40"
                    )}
                  >
                    {runningCount > 0 ? runningCount : null}
                  </span>
                </>
              )}
            </button>

            {/* 通知面板 */}
            <NotificationPanel anchorRef={bellRef} />

            <a
              href="https://github.com/Stonewuu/ai-fusion-video"
              target="_blank"
              rel="noreferrer"
              className={cn(
                "p-2 rounded-xl transition-colors",
                "text-sky-500 hover:text-sky-600 hover:bg-sky-500/10 dark:text-sky-300 dark:hover:text-sky-200 dark:hover:bg-sky-400/10"
              )}
              aria-label="打开 GitHub 仓库"
              title="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>

            {/* 用户头像下拉菜单 */}
            <UserAvatarDropdown
              user={{
                name: user?.nickname || user?.username || "用户",
                email: user?.email || user?.username,
                avatarUrl: user?.avatar,
              }}
              menuItems={userMenuItems}
              logoutItem={userLogoutItem}
            />
          </div>
        }
      />

      {/* 移动端下拉菜单面板 */}
      <MobileMenuPanel
        items={menuItems}
        activeItem={activeLabel}
        onItemClick={handleMenuClick}
        isOpen={mobileMenuOpen}
      />
    </header>
  );
}
