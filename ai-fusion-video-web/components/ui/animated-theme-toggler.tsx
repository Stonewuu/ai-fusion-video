"use client"

import { useEffect, useRef, useState, useCallback } from "react"

import { Moon, Sun } from "lucide-react"

import { motion, AnimatePresence } from "framer-motion"

import { cn } from "@/lib/utils"

type AnimatedThemeTogglerProps = {
  className?: string
}

export const AnimatedThemeToggler = ({ className }: AnimatedThemeTogglerProps) => {
  const buttonRef = useRef<HTMLButtonElement>(null)
  const [mounted, setMounted] = useState(false)
  const [darkMode, setDarkMode] = useState(false)

  // 挂载后同步真实主题状态，并监听后续变更
  useEffect(() => {
    setDarkMode(document.documentElement.classList.contains("dark"))
    setMounted(true)

    const syncTheme = () =>
      setDarkMode(document.documentElement.classList.contains("dark"))

    const observer = new MutationObserver(syncTheme)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })
    return () => observer.disconnect()
  }, [])

  const onToggle = useCallback(async () => {
    if (!buttonRef.current) return

    const toggled = !darkMode

    // 不支持 View Transitions API 或用户偏好减少动画 → 降级直接切换
    if (
      !document.startViewTransition ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      document.documentElement.classList.toggle("dark", toggled)
      localStorage.setItem("theme", toggled ? "dark" : "light")
      return
    }

    // 提前记录按钮位置
    const { left, top, width, height } = buttonRef.current.getBoundingClientRect()
    const x = left + width / 2
    const y = top + height / 2
    const maxRadius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    )

    // 全局检测：找出 main 区域内所有窗口外的元素（旧截图捕获前，元素仍可见）
    // 使用 TreeWalker + 子树剪枝，父元素已经在屏幕外则直接跳过所有后代
    const offScreenEls: { el: HTMLElement; h: number }[] = []
    const mainEl = document.querySelector("main")
    if (mainEl) {
      const vh = window.innerHeight
      const walk = (parent: Element) => {
        for (const child of parent.children) {
          if (!(child instanceof HTMLElement)) continue
          const rect = child.getBoundingClientRect()
          if (rect.height === 0) continue
          // 完全在窗口外（上方或下方留 200px 缓冲区）→ 收集并跳过子树
          if (rect.bottom < -200 || rect.top > vh + 200) {
            if (child.children.length > 0) {
              offScreenEls.push({ el: child, h: rect.height })
            }
            continue // 剪枝：不再遍历子节点
          }
          // 在窗口内 → 递归检查子节点
          if (child.children.length > 0) walk(child)
        }
      }
      walk(mainEl)
    }

    try {
      const transition = document.startViewTransition(() => {
        // 旧截图已捕获后：将窗口外元素设为 content-visibility: hidden
        // 浏览器跳过它们的样式重算，大幅加速新截图生成
        for (const { el, h } of offScreenEls) {
          el.style.contentVisibility = "hidden"
          el.style.containIntrinsicBlockSize = `auto ${h}px`
        }
        document.documentElement.classList.toggle("dark", toggled)
        localStorage.setItem("theme", toggled ? "dark" : "light")
      })

      await transition.ready

      document.documentElement.animate(
        {
          clipPath: [
            `circle(0px at ${x}px ${y}px)`,
            `circle(${maxRadius}px at ${x}px ${y}px)`,
          ],
        },
        {
          duration: 500,
          easing: "ease-in-out",
          pseudoElement: "::view-transition-new(root)",
        }
      )

      await transition.finished
    } catch {
      // View Transition 失败时降级
      if (!document.documentElement.classList.contains(toggled ? "dark" : "")) {
        document.documentElement.classList.toggle("dark", toggled)
        localStorage.setItem("theme", toggled ? "dark" : "light")
      }
    } finally {
      // 恢复所有被隐藏的元素
      for (const { el } of offScreenEls) {
        el.style.contentVisibility = ""
        el.style.containIntrinsicBlockSize = ""
      }
    }
  }, [darkMode])

  return (
    <button
      ref={buttonRef}
      onClick={onToggle}
      aria-label="切换主题"
      className={cn(
        "flex items-center justify-center p-2 rounded-full outline-none focus:outline-none active:outline-none focus:ring-0 cursor-pointer text-foreground",
        className
      )}
      type="button"
    >
      {!mounted ? (
        <span className="h-5 w-5" />
      ) : (
        <AnimatePresence mode="wait" initial={false}>
          {darkMode ? (
            <motion.span
              key="sun-icon"
              initial={{ opacity: 0, scale: 0.55, rotate: 25 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.33 }}
              className="text-current"
            >
              <Sun className="h-5 w-5" />
            </motion.span>
          ) : (
            <motion.span
              key="moon-icon"
              initial={{ opacity: 0, scale: 0.55, rotate: -25 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.33 }}
              className="text-current"
            >
              <Moon className="h-5 w-5" />
            </motion.span>
          )}
        </AnimatePresence>
      )}
    </button>
  )
}
