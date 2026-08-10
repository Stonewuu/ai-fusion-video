"use client";

import type { LucideIcon } from "lucide-react";
import {
  User,
  Mountain,
  Box,
  Car,
  Building2,
  Shirt,
  Sparkles,
  Images,
  ImageOff,
} from "lucide-react";
import { cn } from "@/lib/utils";

/** 类型→图标 + 渐变色配置 */
const typeConfig: Record<string, {
  icon: LucideIcon;
  gradient: string;
  glowColor: string;
  iconColor: string;
}> = {
  character: {
    icon: User,
    gradient: "from-blue-500/8 via-cyan-400/12 to-blue-600/8",
    glowColor: "bg-blue-400/20",
    iconColor: "text-blue-400/30",
  },
  scene: {
    icon: Mountain,
    gradient: "from-green-500/8 via-emerald-400/12 to-green-600/8",
    glowColor: "bg-green-400/20",
    iconColor: "text-green-400/30",
  },
  prop: {
    icon: Box,
    gradient: "from-amber-500/8 via-yellow-400/12 to-amber-600/8",
    glowColor: "bg-amber-400/20",
    iconColor: "text-amber-400/30",
  },
  vehicle: {
    icon: Car,
    gradient: "from-cyan-500/8 via-sky-400/12 to-cyan-600/8",
    glowColor: "bg-cyan-400/20",
    iconColor: "text-cyan-400/30",
  },
  building: {
    icon: Building2,
    gradient: "from-purple-500/8 via-violet-400/12 to-purple-600/8",
    glowColor: "bg-purple-400/20",
    iconColor: "text-purple-400/30",
  },
  costume: {
    icon: Shirt,
    gradient: "from-pink-500/8 via-rose-400/12 to-pink-600/8",
    glowColor: "bg-pink-400/20",
    iconColor: "text-pink-400/30",
  },
  effect: {
    icon: Sparkles,
    gradient: "from-orange-500/8 via-amber-400/12 to-orange-600/8",
    glowColor: "bg-orange-400/20",
    iconColor: "text-orange-400/30",
  },
};

const defaultConfig: { icon: LucideIcon; gradient: string; glowColor: string; iconColor: string } = {
  icon: Images,
  gradient: "from-muted/10 via-muted/15 to-muted/10",
  glowColor: "bg-muted/20",
  iconColor: "text-muted-foreground/20",
};

interface Props {
  type: string;
  className?: string;
  /** 图标大小 class，默认 h-8 w-8 */
  iconSize?: string;
  /** 是否为加载失败状态 */
  isError?: boolean;
  onClick?: React.MouseEventHandler<HTMLDivElement>;
}

/**
 * 资产类型占位图：不同类型显示不同图标 + 颜色流光动画
 */
export default function AssetTypePlaceholder({ type, className, iconSize, isError, onClick }: Props) {
  let config = typeConfig[type] || defaultConfig;
  
  if (isError) {
    config = {
      ...config,
      icon: ImageOff,
      gradient: "from-red-500/5 via-neutral-100/40 to-neutral-200/30 dark:from-red-950/5 dark:via-neutral-900/40 dark:to-neutral-950/5",
      glowColor: "bg-red-400/10 dark:bg-red-950/10",
      iconColor: "text-red-400/35 dark:text-red-500/25",
    };
  }

  const Icon = config.icon;
  const finalIconSize = iconSize || "w-5 h-5 @[120px]:w-8 @[120px]:h-8";

  return (
    <div 
      onClick={onClick}
      className={cn(
        "@container relative flex flex-col items-center justify-center overflow-hidden shrink-0 select-none",
        `bg-linear-to-br ${config.gradient}`,
        className
      )}
    >
      {/* 流光效果 */}
      <div
        className={cn(
          "absolute inset-0",
          isError ? "opacity-20" : "opacity-60",
          "before:absolute before:inset-0",
          "before:bg-linear-to-r before:from-transparent before:via-white/4 before:to-transparent",
          isError 
            ? "before:animate-[shimmer_5s_ease-in-out_infinite]"
            : "before:animate-[shimmer_3s_ease-in-out_infinite]",
        )}
      />
      {/* 光晕 */}
      <div className={cn(
        "absolute rounded-full blur-2xl opacity-40 animate-pulse",
        config.glowColor,
        "w-1/2 h-1/2"
      )} />
      {/* 图标 */}
      <div className="relative z-10">
        <Icon className={cn(finalIconSize, config.iconColor)} />
      </div>
      {/* 精致的失效胶囊状态微章（大图顶部居中显示，小图隐藏） */}
      {isError && (
        <div className="absolute top-2.5 left-1/2 -translate-x-1/2 items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-white/75 dark:bg-neutral-950/50 border border-neutral-200/50 dark:border-white/10 backdrop-blur-md text-[10px] text-neutral-500 dark:text-neutral-400 font-medium tracking-wide shadow-xs select-none pointer-events-none animate-in fade-in slide-in-from-top-1 duration-300 @[120px]:flex hidden">
          <ImageOff className="h-3 w-3 text-red-500/70 dark:text-red-400/70" />
          <span>图片失效</span>
        </div>
      )}
    </div>
  );
}
