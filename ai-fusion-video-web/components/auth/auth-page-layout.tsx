"use client";

import { useEffect, useRef, type ReactNode } from "react";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { Check } from "lucide-react";
import { OverlayScrollArea } from "@/components/dashboard/overlay-scroll-area";
import { cn } from "@/lib/utils";
import { AuthAnimatedBackground } from "./auth-animated-background";
import { usePrefersReducedMotion } from "@/hooks/use-prefers-reduced-motion";

const AUTH_PATHS = new Set(["/login", "/register", "/setup", "/forgot-password"]);

interface AuthPageLayoutProps {
  className?: string;
  children: ReactNode;
  showSuccess?: boolean;
  successTitle?: string;
  successSubtitle?: string;
  onTransitionComplete?: () => void;
}

/** 点阵只挂载在认证路由，并与表单滚动区域分开，保持背景为一屏大小。 */
export function AuthPageLayout({
  className,
  children,
  showSuccess = false,
  successTitle = "操作成功",
  successSubtitle = "正在跳转",
  onTransitionComplete,
}: AuthPageLayoutProps) {
  const reducedMotion = usePrefersReducedMotion();
  const pathname = usePathname();
  const successNotifiedRef = useRef(false);

  useEffect(() => {
    if (!showSuccess) {
      successNotifiedRef.current = false;
    }
  }, [showSuccess]);

  const completeTransition = () => {
    // 零时长过渡可能重复通知，同一次成功状态只触发一次跳转。
    if (!successNotifiedRef.current) {
      successNotifiedRef.current = true;
      onTransitionComplete?.();
    }
  };

  return (
    <div className={cn("relative isolate h-svh w-full bg-background", className)}>
      {AUTH_PATHS.has(pathname) ? <AuthAnimatedBackground paused={showSuccess} /> : null}
      <OverlayScrollArea className="h-full w-full" viewportClassName="flex flex-col">
        <div className="relative flex min-h-full w-full shrink-0 flex-col">
          <header className="flex shrink-0 items-center gap-3 p-6">
            <Image src="/logo.png" alt="" width={480} height={96} priority className="h-10 w-auto" />
            <span className="text-2xl font-semibold tracking-wide text-foreground">融光</span>
          </header>
          <main className="flex flex-1 items-center justify-center px-6 pb-12 pt-6">
            <div className="w-full min-w-0 max-w-sm p-1">
              <AnimatePresence mode="wait">
                {showSuccess ? (
                  <motion.div
                    key="success"
                    role="status"
                    initial={{ opacity: 0, y: reducedMotion ? 0 : 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: reducedMotion ? 0 : 0.2, ease: "easeOut" }}
                    onAnimationComplete={completeTransition}
                    className="flex flex-col items-center gap-4 text-center"
                  >
                    <span className="flex size-14 items-center justify-center rounded-full bg-primary/15 text-primary">
                      <Check aria-hidden="true" className="size-7" />
                    </span>
                    <div className="flex flex-col gap-2">
                      <h1 className="text-3xl font-semibold text-foreground">{successTitle}</h1>
                      <p className="text-base text-muted-foreground">{successSubtitle}</p>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="form"
                    initial={{ opacity: 0, y: reducedMotion ? 0 : 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: reducedMotion ? 0 : 0.15, ease: "easeOut" }}
                    className="flex flex-col gap-6 text-center"
                  >
                    {children}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </main>
        </div>
      </OverlayScrollArea>
    </div>
  );
}
