"use client"

import * as React from "react"
import { motion, AnimatePresence, type Variants, type Transition } from "framer-motion"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"
import { type LucideIcon } from "lucide-react"

interface MenuItem {
  icon: LucideIcon | React.FC
  label: string
  href: string
  gradient: string
  iconColor: string
}

/** 菜单显示模式：图标+文字 / 纯图标 / 汉堡菜单 */
export type MenuDisplayMode = "full" | "compact" | "mobile"

interface MenuBarProps {
  items: MenuItem[]
  activeItem?: string
  onItemClick?: (label: string) => void
  className?: string
  /** 左侧额外内容（如 Logo） */
  leftContent?: React.ReactNode
  /** 右侧额外内容（如用户头像） */
  rightContent?: React.ReactNode
  /** 移动端汉堡按钮，在 mobile 模式下显示 */
  mobileControls?: React.ReactNode
  /** 显示模式变化回调 */
  onDisplayModeChange?: (mode: MenuDisplayMode) => void
}

const itemVariants: Variants = {
  initial: { rotateX: 0, opacity: 1 },
  hover: { rotateX: -90, opacity: 0 },
}

const backVariants: Variants = {
  initial: { rotateX: 90, opacity: 0 },
  hover: { rotateX: 0, opacity: 1 },
}

// 完整模式下的光晕效果
const glowVariants: Variants = {
  initial: { opacity: 0, scale: 0.8 },
  hover: {
    opacity: 1,
    scale: 2,
    transition: {
      opacity: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as [number, number, number, number] },
      scale: { duration: 0.5, type: "spring" as const, stiffness: 300, damping: 25 },
    },
  },
}

// 紧凑模式下的光晕效果（与完整模式相同 scale，保持边缘渐变消散效果）
const compactGlowVariants: Variants = {
  initial: { opacity: 0, scale: 0.8 },
  hover: {
    opacity: 1,
    scale: 3,
    transition: {
      opacity: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as [number, number, number, number] },
      scale: { duration: 0.5, type: "spring" as const, stiffness: 300, damping: 25 },
    },
  },
}

const navGlowVariants: Variants = {
  initial: { opacity: 0 },
  hover: {
    opacity: 1,
    transition: {
      duration: 0.5,
      ease: [0.4, 0, 0.2, 1] as [number, number, number, number],
    },
  },
}

const sharedTransition: Transition = {
  type: "spring" as const,
  stiffness: 100,
  damping: 20,
  duration: 0.5,
}

/**
 * 基于实际内容宽度的动态菜单模式检测
 * 测量左侧和右侧内容的实际宽度，计算菜单可用空间，自动决定显示模式
 */
function useMenuDisplayMode(
  navRef: React.RefObject<HTMLElement | null>,
  leftRef: React.RefObject<HTMLElement | null>,
  rightRef: React.RefObject<HTMLElement | null>,
  itemCount: number,
): MenuDisplayMode {
  const [mode, setMode] = React.useState<MenuDisplayMode>("full")

  React.useEffect(() => {
    const nav = navRef.current
    if (!nav) return

    const calculate = () => {
      const navWidth = nav.clientWidth
      const leftWidth = leftRef.current?.offsetWidth ?? 0
      const rightWidth = rightRef.current?.offsetWidth ?? 0

      // 区域间距（左区 mr + 右区 ml + nav 内边距 p-2）
      const sectionGaps = 48
      const availableForMenu = navWidth - leftWidth - rightWidth - sectionGaps

      // 基于菜单项数量估算所需宽度
      const itemGap = 8
      // 图标+文字+内边距的估算值（px-4=32 + icon 20 + gap 8 + 平均中文文字约48）
      const fullItemWidth = 108
      // 纯图标+内边距的估算值（px-3=24 + icon 20）
      const compactItemWidth = 48

      const fullMenuWidth = itemCount * fullItemWidth + (itemCount - 1) * itemGap
      const compactMenuWidth = itemCount * compactItemWidth + (itemCount - 1) * itemGap

      if (availableForMenu >= fullMenuWidth) {
        setMode("full")
      } else if (availableForMenu >= compactMenuWidth) {
        setMode("compact")
      } else {
        setMode("mobile")
      }
    }

    const observer = new ResizeObserver(calculate)
    observer.observe(nav)
    // 同时监听左右内容的尺寸变化
    if (leftRef.current) observer.observe(leftRef.current)
    if (rightRef.current) observer.observe(rightRef.current)

    calculate()

    return () => observer.disconnect()
  }, [navRef, leftRef, rightRef, itemCount])

  return mode
}

/**
 * 单个菜单项按钮组件
 */
function MenuItemButton({
  item,
  isActive,
  onItemClick,
  iconOnly = false,
}: {
  item: MenuItem
  isActive: boolean
  onItemClick?: (label: string) => void
  iconOnly?: boolean
}) {
  const Icon = item.icon

  return (
    <button
      onClick={() => onItemClick?.(item.label)}
      className="block w-full"
      title={iconOnly ? item.label : undefined}
    >
      <motion.div
        className="block rounded-xl overflow-visible group relative"
        style={{ perspective: "600px" }}
        whileHover="hover"
        initial="initial"
      >
        <motion.div
          className="absolute inset-0 z-0 pointer-events-none"
          variants={iconOnly ? compactGlowVariants : glowVariants}
          animate={isActive ? "hover" : "initial"}
          style={{
            background: item.gradient,
            opacity: isActive ? 1 : 0,
            borderRadius: "16px",
          }}
        />
        {/* 正面 */}
        <motion.div
          className={cn(
            "flex items-center relative z-10 bg-transparent transition-colors rounded-xl",
            iconOnly ? "px-4 py-2 justify-center" : "gap-2 px-4 py-2",
            isActive
              ? "text-foreground"
              : "text-muted-foreground group-hover:text-foreground",
          )}
          variants={itemVariants}
          transition={sharedTransition}
          style={{
            transformStyle: "preserve-3d",
            transformOrigin: "center bottom",
          }}
        >
          <span
            className={cn(
              "transition-colors duration-300",
              isActive ? item.iconColor : "text-foreground",
              `group-hover:${item.iconColor}`,
            )}
          >
            <Icon className="h-5 w-5" />
          </span>
          {!iconOnly && <span>{item.label}</span>}
        </motion.div>
        {/* 背面（hover 翻转效果） */}
        <motion.div
          className={cn(
            "flex items-center absolute inset-0 z-10 bg-transparent transition-colors rounded-xl",
            iconOnly ? "px-3 py-2 justify-center" : "gap-2 px-4 py-2",
            isActive
              ? "text-foreground"
              : "text-muted-foreground group-hover:text-foreground",
          )}
          variants={backVariants}
          transition={sharedTransition}
          style={{
            transformStyle: "preserve-3d",
            transformOrigin: "center top",
            rotateX: 90,
          }}
        >
          <span
            className={cn(
              "transition-colors duration-300",
              isActive ? item.iconColor : "text-foreground",
              `group-hover:${item.iconColor}`,
            )}
          >
            <Icon className="h-5 w-5" />
          </span>
          {!iconOnly && <span>{item.label}</span>}
        </motion.div>
      </motion.div>
    </button>
  )
}

/**
 * 移动端下拉菜单面板
 * 简洁列表样式，选中项左侧彩色边框 + 微妙背景高亮
 */
export function MobileMenuPanel({
  items,
  activeItem,
  onItemClick,
  isOpen,
}: {
  items: MenuItem[]
  activeItem?: string
  onItemClick?: (label: string) => void
  isOpen: boolean
}) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
          className="overflow-hidden mt-1 rounded-2xl bg-background/95 backdrop-blur-xl border border-border/40 shadow-xl z-50"
        >
          <div className="py-2 px-1">
            {items.map((item, index) => {
              const Icon = item.icon
              const isActive = item.label === activeItem
              return (
                <motion.button
                  key={item.label}
                  onClick={() => onItemClick?.(item.label)}
                  initial={{ opacity: 0, x: -16 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2, delay: index * 0.04 }}
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-colors duration-150",
                    isActive
                      ? "bg-foreground/5 text-foreground"
                      : "text-muted-foreground hover:bg-foreground/3 hover:text-foreground",
                  )}
                >
                  {/* 选中指示条，颜色与菜单项图标一致 */}
                  <span
                    className={cn(
                      "transition-colors duration-200",
                      isActive ? item.iconColor : "text-transparent",
                    )}
                  >
                    <div className="w-0.5 h-5 rounded-full bg-current" />
                  </span>
                  <span
                    className={cn(
                      "transition-colors duration-200",
                      isActive ? item.iconColor : "",
                    )}
                  >
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className="text-sm font-medium">{item.label}</span>
                </motion.button>
              )
            })}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export const MenuBar = React.forwardRef<HTMLDivElement, MenuBarProps>(
  ({ className, items, activeItem, onItemClick, leftContent, rightContent, mobileControls, onDisplayModeChange }, ref) => {
    const { resolvedTheme } = useTheme()
    const [mounted, setMounted] = React.useState(false)
    const navRef = React.useRef<HTMLElement>(null)
    const leftRef = React.useRef<HTMLDivElement>(null)
    const rightRef = React.useRef<HTMLDivElement>(null)

    // 基于实际内容宽度的动态模式检测
    const displayMode = useMenuDisplayMode(navRef, leftRef, rightRef, items.length)

    // 通知父组件模式变化
    React.useEffect(() => {
      onDisplayModeChange?.(displayMode)
    }, [displayMode, onDisplayModeChange])

    React.useEffect(() => {
      setMounted(true)
    }, [])

    const isDarkTheme = mounted ? resolvedTheme === "dark" : true

    return (
      <motion.nav
        ref={(node) => {
          (navRef as React.MutableRefObject<HTMLElement | null>).current = node
          const divNode = node as HTMLDivElement | null
          if (typeof ref === "function") ref(divNode)
          else if (ref) (ref as React.MutableRefObject<HTMLDivElement | null>).current = divNode
        }}
        className={cn(
          "p-2 rounded-2xl bg-linear-to-b from-background/80 to-background/40 backdrop-blur-lg border border-border/40 shadow-lg relative overflow-hidden",
          className,
        )}
        initial="initial"
        whileHover="hover"
      >
        <motion.div
          className={`absolute -inset-2 bg-gradient-radial from-transparent ${
            isDarkTheme
              ? "via-blue-400/30 via-30% via-purple-400/30 via-60% via-red-400/30 via-90%"
              : "via-blue-400/20 via-30% via-purple-400/20 via-60% via-red-400/20 via-90%"
          } to-transparent rounded-3xl z-0 pointer-events-none`}
          variants={navGlowVariants}
        />
        <div className="flex items-center w-full relative z-10 justify-between">
          {/* 左侧区域 */}
          {leftContent && (
            <div ref={leftRef} className="shrink-0 mr-2 md:mr-4">{leftContent}</div>
          )}

          {/* 桌面端 / 紧凑模式：水平菜单项列表 */}
          {displayMode !== "mobile" && (
            <ul className={cn(
              "flex items-center shrink-0 justify-center",
              displayMode === "compact" ? "gap-2" : "gap-1",
            )}>
              {items.map((item) => {
                const isActive = item.label === activeItem
                return (
                  <motion.li key={item.label} className="relative whitespace-nowrap">
                    <MenuItemButton
                      item={item}
                      isActive={isActive}
                      onItemClick={onItemClick}
                      iconOnly={displayMode === "compact"}
                    />
                  </motion.li>
                )
              })}
            </ul>
          )}

          {/* 移动端模式：汉堡按钮 */}
          {displayMode === "mobile" && mobileControls && (
            <div className="flex items-center">
              {mobileControls}
            </div>
          )}

          {/* 右侧区域 */}
          {rightContent && (
            <div ref={rightRef} className="shrink-0 ml-2 md:ml-4">{rightContent}</div>
          )}
        </div>
      </motion.nav>
    )
  },
)

MenuBar.displayName = "MenuBar"
