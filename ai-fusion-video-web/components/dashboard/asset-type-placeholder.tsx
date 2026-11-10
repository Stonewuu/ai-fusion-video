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
  iconColor: string;
}> = {
  character: {
    icon: User,
    gradient: "from-blue-500/8 via-cyan-400/12 to-blue-600/8",
    iconColor: "text-blue-400/30",
  },
  scene: {
    icon: Mountain,
    gradient: "from-green-500/8 via-emerald-400/12 to-green-600/8",
    iconColor: "text-green-400/30",
  },
  prop: {
    icon: Box,
    gradient: "from-amber-500/8 via-yellow-400/12 to-amber-600/8",
    iconColor: "text-amber-400/30",
  },
  vehicle: {
    icon: Car,
    gradient: "from-cyan-500/8 via-sky-400/12 to-cyan-600/8",
    iconColor: "text-cyan-400/30",
  },
  building: {
    icon: Building2,
    gradient: "from-purple-500/8 via-violet-400/12 to-purple-600/8",
    iconColor: "text-purple-400/30",
  },
  costume: {
    icon: Shirt,
    gradient: "from-pink-500/8 via-rose-400/12 to-pink-600/8",
    iconColor: "text-pink-400/30",
  },
  effect: {
    icon: Sparkles,
    gradient: "from-orange-500/8 via-amber-400/12 to-orange-600/8",
    iconColor: "text-orange-400/30",
  },
};

const defaultConfig: { icon: LucideIcon; gradient: string; iconColor: string } = {
  icon: Images,
  gradient: "from-muted/10 via-muted/15 to-muted/10",
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
 * 空素材和失效图片使用静态占位，避免列表中的每张占位图持续重绘。
 */
export default function AssetTypePlaceholder({ type, className, iconSize, isError, onClick }: Props) {
  let config = typeConfig[type] || defaultConfig;
  
  if (isError) {
    config = {
      ...config,
      icon: ImageOff,
      gradient: "from-red-500/5 via-neutral-100/40 to-neutral-200/30 dark:from-red-950/5 dark:via-neutral-900/40 dark:to-neutral-950/5",
      iconColor: "text-red-400/35 dark:text-red-500/25",
    };
  }

  const Icon = config.icon;
  const finalIconSize = iconSize || "w-5 h-5 @[120px]:w-8 @[120px]:h-8";

  return (
    <div 
      data-slot="asset-placeholder"
      onClick={onClick}
      className={cn(
        "@container relative flex flex-col items-center justify-center overflow-hidden shrink-0 select-none",
        `bg-linear-to-br ${config.gradient}`,
        className
      )}
    >
      {/* 图标 */}
      <div className="relative z-10">
        <Icon className={cn(finalIconSize, config.iconColor)} />
      </div>
      {/* 大图显示失效提示，小图通过图标区分。 */}
      {isError && (
        <div className="absolute top-2.5 left-1/2 -translate-x-1/2 items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-background/90 border border-border/30 text-[10px] text-muted-foreground font-medium tracking-wide select-none pointer-events-none @[120px]:flex hidden">
          <ImageOff className="h-3 w-3 text-red-500/70 dark:text-red-400/70" />
          <span>图片失效</span>
        </div>
      )}
    </div>
  );
}
