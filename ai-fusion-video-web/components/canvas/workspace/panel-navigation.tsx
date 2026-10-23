"use client";

import {
  BookOpen,
  Images,
  Pencil,
  MessageSquare,
  History,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export const canvasLeftTabs = [
  { key: "content", label: "项目", title: "项目内容", icon: BookOpen },
  { key: "media", label: "素材", title: "媒体素材", icon: Images },
] as const;
export const canvasRightTabs = [
  { key: "properties", label: "编辑", title: "编辑与预览", icon: Pencil },
  { key: "runs", label: "运行", title: "运行与结果", icon: Play },
  { key: "comments", label: "评论", title: "画布评论", icon: MessageSquare },
  { key: "history", label: "历史", title: "历史与版本", icon: History },
] as const;

export type CanvasLeftPanel = (typeof canvasLeftTabs)[number]["key"];
export type CanvasRightPanel = (typeof canvasRightTabs)[number]["key"];
export type CanvasPanel = CanvasLeftPanel | CanvasRightPanel;
export type CanvasMobilePanel = "left" | "right" | null;

export function CanvasPanelNavigation<T extends CanvasPanel>({
  tabs,
  active,
  collapsed = false,
  icons = true,
  onSelect,
  onClose,
  className,
  label,
}: {
  tabs: readonly {
    key: T;
    label: string;
    title: string;
    icon: typeof BookOpen;
  }[];
  active: T | null;
  collapsed?: boolean;
  icons?: boolean;
  onSelect: (panel: T) => void;
  onClose: () => void;
  className?: string;
  label: string;
}) {
  return (
    <nav
      aria-label={label}
      className={cn("flex gap-2", collapsed && "flex-col", className)}
    >
      {tabs.map((tab) => (
        <Button
          key={tab.key}
          data-canvas-panel={tab.key}
          variant={active === tab.key ? "secondary" : "ghost"}
          size="default"
          className={collapsed ? "w-full" : undefined}
          aria-label={tab.title}
          aria-expanded={active === tab.key}
          aria-controls={
            active === tab.key ? `canvas-panel-${tab.key}` : undefined
          }
          title={`${tab.title}${active === tab.key ? "（再次点击收起）" : ""}`}
          onClick={() => (active === tab.key ? onClose() : onSelect(tab.key))}
        >
          {!collapsed && icons && <tab.icon className="hidden sm:block" />}
          {tab.label}
        </Button>
      ))}
    </nav>
  );
}
