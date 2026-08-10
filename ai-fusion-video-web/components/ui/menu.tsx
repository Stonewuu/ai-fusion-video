// 用户头像下拉菜单组件
"use client";

import * as React from "react";
import { useState, useRef, useEffect, useCallback, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

// ============ 类型定义 ============

/** 菜单项 */
export interface AvatarMenuItem {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
}

/** 用户信息 */
export interface AvatarUserInfo {
  name: string;
  email?: string;
  avatarUrl?: string | null;
}

/** UserAvatarDropdown 组件 Props */
export interface UserAvatarDropdownProps {
  /** 用户信息 */
  user: AvatarUserInfo;
  /** 普通菜单项列表 */
  menuItems: AvatarMenuItem[];
  /** 退出登录按钮（独立样式） */
  logoutItem: {
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
  };
  /** 自定义外层 className */
  className?: string;
}

// ============ 动画配置 ============

const dropdownVariants = {
  hidden: { opacity: 0, y: -8, scale: 0.96 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.15, ease: "easeOut" as const },
  },
  exit: {
    opacity: 0,
    y: -8,
    scale: 0.96,
    transition: { duration: 0.12, ease: "easeIn" as const },
  },
};

const itemVariants = {
  hidden: { opacity: 0, x: -12 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { type: "spring" as const, stiffness: 120, damping: 18 },
  },
};

const staggerContainer = {
  visible: {
    transition: { staggerChildren: 0.05 },
  },
};

// ============ 组件 ============

export const UserAvatarDropdown = React.forwardRef<
  HTMLDivElement,
  UserAvatarDropdownProps
>(({ user, menuItems, logoutItem, className }, ref) => {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, right: 0 });
  const isClient = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  // 用户名首字母（头像回退）
  const avatarInitial = (user.name || "U").charAt(0).toUpperCase();

  // 计算下拉菜单位置
  const updateDropdownPos = useCallback(() => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setDropdownPos({
        top: rect.bottom + 8,
        right: window.innerWidth - rect.right,
      });
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      updateDropdownPos();
    }
  }, [isOpen, updateDropdownPos]);

  // 点击菜单项后关闭菜单
  const handleItemClick = (onClick: () => void) => {
    setIsOpen(false);
    onClick();
  };

  return (
    <div ref={ref} className={cn("relative", className)}>
      {/* 触发按钮：头像 + 用户名 */}
      <button
        ref={triggerRef}
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 p-1.5 pr-3 rounded-xl transition-colors",
          "hover:bg-white/5",
          isOpen && "bg-white/5"
        )}
      >
        {/* 头像 */}
        {user.avatarUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={user.avatarUrl}
            alt={`${user.name}的头像`}
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <div className="h-8 w-8 rounded-full bg-linear-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center text-white text-sm font-semibold shadow-lg">
            {avatarInitial}
          </div>
        )}
        <span className="text-sm text-muted-foreground hidden md:inline">
          {user.name}
        </span>
      </button>

      {/* 下拉菜单 — Portal 渲染到 body */}
      {isClient &&
        createPortal(
          <AnimatePresence>
            {isOpen && (
              <>
                {/* 遮罩层：点击外部关闭 */}
                <div
                  className="fixed inset-0 z-60"
                  onClick={() => setIsOpen(false)}
                />
                {/* 下拉面板 */}
                <motion.div
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  variants={dropdownVariants}
                  className={cn(
                    "fixed z-61",
                    "w-64 rounded-2xl overflow-hidden",
                    "border border-border/40",
                    "bg-card text-card-foreground",
                    "shadow-xl shadow-black/10"
                  )}
                  style={{
                    top: dropdownPos.top,
                    right: dropdownPos.right,
                  }}
                >
                  {/* 用户信息头部 */}
                  <div className="px-4 py-4 border-b border-border/20">
                    <div className="flex items-center gap-3">
                      {user.avatarUrl ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={user.avatarUrl}
                          alt={`${user.name}的头像`}
                          className="h-11 w-11 rounded-full object-cover shrink-0"
                        />
                      ) : (
                        <div className="h-11 w-11 rounded-full bg-linear-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center text-white text-base font-semibold shadow-lg shrink-0">
                          {avatarInitial}
                        </div>
                      )}
                      <div className="flex flex-col truncate">
                        <span className="text-base font-semibold truncate">
                          {user.name}
                        </span>
                        {user.email && (
                          <span className="text-xs text-muted-foreground truncate mt-0.5">
                            {user.email}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 菜单项列表 */}
                  <motion.nav
                    className="px-2 py-2"
                    variants={staggerContainer}
                    initial="hidden"
                    animate="visible"
                    role="navigation"
                  >
                    {menuItems.map((item, index) => (
                      <motion.button
                        key={index}
                        variants={itemVariants}
                        onClick={() => handleItemClick(item.onClick)}
                        className="group flex items-center gap-3 w-full px-3 py-3 rounded-lg text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                      >
                        <span className="h-5 w-5 shrink-0">{item.icon}</span>
                        <span>{item.label}</span>
                        <ChevronRight className="ml-auto h-4 w-4 opacity-0 translate-x-[-4px] transition-all duration-200 group-hover:opacity-100 group-hover:translate-x-0" />
                      </motion.button>
                    ))}
                  </motion.nav>

                  {/* 分隔线 + 退出按钮 */}
                  <div className="border-t border-border/20 px-2 py-2">
                    <motion.button
                      variants={itemVariants}
                      initial="hidden"
                      animate="visible"
                      onClick={() => handleItemClick(logoutItem.onClick)}
                      className="group flex items-center gap-3 w-full px-3 py-3 rounded-lg text-sm font-medium text-destructive transition-colors hover:bg-destructive/10"
                    >
                      <span className="h-5 w-5 shrink-0">
                        {logoutItem.icon}
                      </span>
                      <span>{logoutItem.label}</span>
                    </motion.button>
                  </div>
                </motion.div>
              </>
            )}
          </AnimatePresence>,
          document.body
        )}
    </div>
  );
});

UserAvatarDropdown.displayName = "UserAvatarDropdown";
