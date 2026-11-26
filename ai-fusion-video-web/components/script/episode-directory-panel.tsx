"use client";

import { useId, type ReactNode } from "react";
import { PanelLeftClose, PanelLeftOpen, RotateCw } from "lucide-react";
import { DirectoryHeader } from "@/components/shared/directory-header";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface EpisodeDirectoryPanelProps {
  navigation: "scripts" | "storyboards";
  title: string;
  count: number;
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  headerAction?: ReactNode;
  children: ReactNode;
}

export function EpisodeDirectoryPanel({
  navigation,
  title,
  count,
  collapsed = false,
  onCollapsedChange,
  onRefresh,
  isRefreshing,
  headerAction,
  children,
}: EpisodeDirectoryPanelProps) {
  const bodyId = useId();
  const toggleLabel = `${collapsed ? "展开" : "收起"}${title}`;
  const ToggleIcon = collapsed ? PanelLeftOpen : PanelLeftClose;

  return (
    <div
      data-sidebar-navigation={navigation}
      className={cn(
        "relative flex h-full min-h-0 shrink-0 flex-col overflow-hidden bg-card/20",
        onCollapsedChange
          ? "transition-[width] duration-200 ease-out after:pointer-events-none after:absolute after:inset-y-0 after:right-0 after:border-r after:border-border/20 motion-reduce:transition-none"
          : "w-full",
        onCollapsedChange && (collapsed ? "w-12" : "w-64"),
      )}
    >
      <DirectoryHeader
        title={title}
        collapsed={collapsed}
        badge={
          <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-xs tabular-nums text-muted-foreground">
            {count} 集
          </span>
        }
        className={cn("border-b border-border/20 py-3", collapsed ? "px-1" : "px-3")}
      >
        {!collapsed && onRefresh && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={isRefreshing}
            aria-label={`刷新${title}`}
            title={`刷新${title}`}
          >
            <RotateCw aria-hidden="true" className={cn(isRefreshing && "animate-spin motion-reduce:animate-none")} />
          </Button>
        )}
        {onCollapsedChange && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={toggleLabel}
            title={toggleLabel}
            aria-expanded={!collapsed}
            aria-controls={bodyId}
            onClick={() => onCollapsedChange(!collapsed)}
          >
            <ToggleIcon aria-hidden="true" />
          </Button>
        )}
        {headerAction}
      </DirectoryHeader>
      <div
        id={bodyId}
        inert={collapsed}
        aria-hidden={collapsed}
        className={cn(
          "flex min-h-0 flex-1 flex-col transition-opacity duration-150 motion-reduce:transition-none",
          onCollapsedChange && "w-64",
          collapsed && "invisible opacity-0",
        )}
      >
        {children}
      </div>
    </div>
  );
}
