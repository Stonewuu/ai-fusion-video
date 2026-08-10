"use client";

import React, { useState, useEffect, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { CanvasRevealEffect } from "@/components/ui/sign-in-flow";

interface AuthLayoutProps {
  className?: string;
  children: React.ReactNode;
  // 成功状态由外部控制
  showSuccess?: boolean;
  // 成功后显示的内容
  successTitle?: string;
  successSubtitle?: string;
  // 扩散过渡动画完成后的回调
  onTransitionComplete?: () => void;
}

/**
 * 认证页面共享布局
 * 包含：Canvas 粒子背景动画、左上角 Logo、成功后打勾+圆形扩散过渡
 * 业务逻辑由外部（页面层）控制
 */
export const AuthLayout = ({
  className,
  children,
  showSuccess = false,
  successTitle = "操作成功",
  successSubtitle = "正在跳转...",
  onTransitionComplete,
}: AuthLayoutProps) => {
  const [initialCanvasHidden, setInitialCanvasHidden] = useState(false);
  const isClient = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false
  );

  // 外部通过 showSuccess prop 控制，这里延迟隐藏初始 Canvas，形成转场叠加
  useEffect(() => {
    if (!showSuccess) return;
    const timer = setTimeout(() => setInitialCanvasHidden(true), 50);
    return () => clearTimeout(timer);
  }, [showSuccess]);

  const initialCanvasVisible = !initialCanvasHidden;
  const reverseCanvasVisible = showSuccess;

  return (
    <div
      className={cn(
        "flex w-full flex-col min-h-screen bg-black relative overflow-hidden",
        className
      )}
    >
      {/* 背景动画层 */}
      <div className="absolute inset-0 z-0">
        {initialCanvasVisible && (
          <div className="absolute inset-0">
            <CanvasRevealEffect
              animationSpeed={3}
              containerClassName="bg-black"
              colors={[
                [40, 60, 140],
                [70, 100, 180],
              ]}
              dotSize={6}
              reverse={false}
            />
          </div>
        )}

        {reverseCanvasVisible && (
          <div className="absolute inset-0">
            <CanvasRevealEffect
              animationSpeed={4}
              containerClassName="bg-black"
              colors={[
                [40, 60, 140],
                [70, 100, 180],
              ]}
              dotSize={6}
              reverse={true}
            />
          </div>
        )}

        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,0,0,1)_0%,transparent_100%)]" />
        <div className="absolute top-0 left-0 right-0 h-1/3 bg-linear-to-b from-black to-transparent" />
      </div>

      {/* 内容层 */}
      <div className="relative z-10 flex flex-col flex-1">
        {/* 左上角 Logo + 标题 */}
        <div className="absolute top-6 left-6 z-20 flex items-center gap-3">
          <Image
            src="/logo.png"
            alt="融光"
            width={480}
            height={96}
            priority
            className="h-10 w-auto"
          />
          <div className="flex items-center text-[1.72rem] font-semibold tracking-[0.02em]">
            <span className="text-transparent bg-[linear-gradient(96deg,#7adcf2_0%,#2EC7D5_38%,#2488C8_100%)] bg-clip-text drop-shadow-[0_0_10px_rgba(46,199,213,0.18)]">
              融
            </span>
            <span className="text-transparent bg-[linear-gradient(96deg,#FCB659_0%,#FF9F68_34%,#E65979_100%)] bg-clip-text drop-shadow-[0_0_10px_rgba(230,89,121,0.14)]">
              光
            </span>
          </div>
        </div>

        <div className="flex flex-1 flex-col lg:flex-row">
          <div className="flex-1 flex flex-col justify-center items-center">
            <div className="w-full max-w-sm px-4">
              <AnimatePresence mode="wait">
                {!showSuccess ? (
                  <motion.div
                    key="form-content"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.4, ease: "easeOut" }}
                    className="space-y-6 text-center"
                  >
                    {children}
                  </motion.div>
                ) : (
                  // 成功状态 - 只显示文字，打勾圆形通过 portal 传送到 body
                  <motion.div
                    key="success"
                    initial={{ opacity: 0, y: 50 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                      duration: 0.3,
                      ease: "easeOut",
                      delay: 0.0,
                    }}
                    className="space-y-6 text-center -translate-y-24"
                  >
                    <div className="space-y-1">
                      <h1 className="text-[2.5rem] font-bold leading-[1.1] tracking-tight text-white">
                        {successTitle}
                      </h1>
                      <p className="text-[1.25rem] text-white/50 font-light">
                        {successSubtitle}
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>

      {/* 打勾 + 扩散圆形 - 通过 createPortal 传送到 body，脱离层叠上下文 */}
      {showSuccess &&
        isClient &&
        createPortal(
          <div
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 9999,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              pointerEvents: "none",
            }}
          >
            {/* 打勾 + 扩散（唯一的圆形，通过 portal 渲染到 body） */}
            <motion.div
              style={{
                position: "absolute",
                top: "calc(50% - 1rem)",
                left: "50%",
                translateX: "-50%",
              }}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.6 }}
            >
              <motion.div
                className="w-16 h-16 rounded-full bg-white flex items-center justify-center"
                initial={{ scale: 1 }}
                animate={{ scale: 60 }}
                transition={{
                  duration: 0.6,
                  delay: 1.5,
                  ease: [0.55, 0, 1, 0.45],
                }}
                onAnimationComplete={() => onTransitionComplete?.()}
              >
                <motion.svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-8 w-8 text-black"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  initial={{ opacity: 1 }}
                  animate={{ opacity: 0 }}
                  transition={{
                    duration: 0.3,
                    delay: 1.3,
                  }}
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </motion.svg>
              </motion.div>
            </motion.div>
          </div>,
          document.body
        )}
    </div>
  );
};
