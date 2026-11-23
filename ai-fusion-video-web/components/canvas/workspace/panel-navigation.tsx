"use client";

import {
  BookOpen,
  Images,
  Layers,
  MessageSquare,
  History,
  Play,
  type LucideIcon,
} from "lucide-react";

export const canvasLeftTabs = [
  { key: "content", label: "项目", title: "项目内容", icon: BookOpen },
  { key: "media", label: "素材", title: "媒体素材", icon: Images },
] as const;

export const canvasRightTabs = [
  { key: "properties", label: "选区", title: "选区与连线", icon: Layers },
  { key: "runs", label: "运行", title: "运行与结果", icon: Play },
  { key: "comments", label: "评论", title: "画布评论", icon: MessageSquare },
  { key: "history", label: "历史", title: "历史与版本", icon: History },
] as const;

export type CanvasLeftPanel = (typeof canvasLeftTabs)[number]["key"];
export type CanvasRightPanel = (typeof canvasRightTabs)[number]["key"];
export type CanvasPanel = CanvasLeftPanel | CanvasRightPanel;
export type CanvasMobilePanel = "left" | "right" | null;

export interface CanvasPanelTab<T extends CanvasPanel = CanvasPanel> {
  key: T;
  label: string;
  title: string;
  icon: LucideIcon;
}
