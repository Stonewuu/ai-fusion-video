"use client";

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { AtSign, Check, FolderKanban, Loader2, PlugZap, Sparkles } from "lucide-react";
import type {
  AssistantReferencePickerItem,
  AssistantReferencePickerMode,
} from "./reference-types";

interface AssistantReferencePickerProps {
  anchorRef: RefObject<HTMLElement | null>;
  mode: AssistantReferencePickerMode;
  query: string;
  items: AssistantReferencePickerItem[];
  activeIndex: number;
  loading: boolean;
  error?: string | null;
  selectedKeys: Set<string>;
  onActiveIndexChange: (index: number) => void;
  onSelect: (item: AssistantReferencePickerItem) => void;
}

function itemKey(item: AssistantReferencePickerItem) {
  if (item.kind === "project") return `project:${item.value.id}`;
  if (item.kind === "skill") return `skill:${item.value.id}`;
  return `mcp:${item.value.serverName}:${item.value.toolName}`;
}

function itemPresentation(item: AssistantReferencePickerItem) {
  if (item.kind === "project") {
    return {
      icon: <FolderKanban className="size-4" />,
      eyebrow: `项目 #${item.value.id}`,
      label: item.value.name,
      description: item.value.description || "将项目作为本轮对话上下文",
    };
  }
  if (item.kind === "skill") {
    return {
      icon: <Sparkles className="size-4" />,
      eyebrow: `Skill · ${item.value.name}`,
      label: item.value.displayName,
      description: item.value.description,
    };
  }
  return {
    icon: <PlugZap className="size-4" />,
    eyebrow: `MCP · ${item.value.serverName}`,
    label: item.value.toolName,
    description: item.value.description || "引用 MCP 工具",
  };
}

export function AssistantReferencePicker({
  anchorRef,
  mode,
  query,
  items,
  activeIndex,
  loading,
  error,
  selectedKeys,
  onActiveIndexChange,
  onSelect,
}: AssistantReferencePickerProps) {
  const [position, setPosition] = useState<CSSProperties | null>(null);
  const activeRef = useRef<HTMLButtonElement>(null);

  useLayoutEffect(() => {
    const updatePosition = () => {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      const viewportPadding = 12;
      const availableWidth = Math.max(0, window.innerWidth - viewportPadding * 2);
      const width = Math.min(availableWidth, Math.min(480, Math.max(280, rect.width)));
      const left = Math.min(
        Math.max(viewportPadding, rect.left),
        Math.max(viewportPadding, window.innerWidth - width - viewportPadding),
      );
      setPosition({
        position: "fixed",
        left,
        bottom: window.innerHeight - rect.top + 8,
        width,
        maxHeight: Math.max(160, Math.min(320, rect.top - 24)),
        zIndex: 120,
      });
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    const observer = new ResizeObserver(updatePosition);
    if (anchorRef.current) observer.observe(anchorRef.current);
    return () => {
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
      observer.disconnect();
    };
  }, [anchorRef, mode]);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  if (!position || typeof document === "undefined") return null;

  const title = mode === "project" ? "引用项目" : "引用 Skill 或 MCP";
  const hint = mode === "project" ? "输入名称或项目 ID 模糊查找" : "输入名称、来源或描述模糊查找";

  return createPortal(
    <div
      className="flex flex-col overflow-hidden rounded-2xl border border-border/40 bg-popover/80 p-2 shadow-xl backdrop-blur-xl"
      style={position}
      role="listbox"
      aria-label={title}
      data-assistant-interactive="true"
    >
      <div className="flex shrink-0 items-center gap-2 border-b border-border/20 px-2 pb-2 pt-1">
        <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
          {mode === "project" ? <AtSign className="size-4" /> : <span className="font-mono text-sm">/</span>}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-xs font-medium">{title}</span>
          <span className="block truncate text-[10px] text-muted-foreground">
            {query ? `正在查找“${query}”` : hint}
          </span>
        </span>
      </div>

      <div className="min-h-0 overflow-y-auto pt-1">
        {loading ? (
          <div className="flex h-20 items-center justify-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-4 animate-spin motion-reduce:animate-none" /> 加载引用项
          </div>
        ) : error ? (
          <div className="px-3 py-5 text-center text-xs text-destructive">{error}</div>
        ) : items.length ? (
          items.map((item, index) => {
            const presentation = itemPresentation(item);
            const key = itemKey(item);
            const selected = selectedKeys.has(key);
            const active = index === activeIndex;
            return (
              <button
                key={key}
                ref={active ? activeRef : undefined}
                type="button"
                role="option"
                aria-selected={selected}
                className={`flex min-h-12 w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50 ${active ? "bg-accent text-accent-foreground" : "hover:bg-muted/70"}`}
                data-assistant-interactive="true"
                onPointerDown={(event) => event.preventDefault()}
                onMouseEnter={() => onActiveIndexChange(index)}
                onClick={() => onSelect(item)}
              >
                <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border/20 bg-background/70 text-muted-foreground">
                  {presentation.icon}
                </span>
                <span className="min-w-0 flex-1 leading-tight">
                  <span className="block truncate text-[10px] text-muted-foreground">
                    {presentation.eyebrow}
                  </span>
                  <span className="mt-0.5 block truncate text-xs font-medium">
                    {presentation.label}
                  </span>
                  <span className="mt-0.5 block truncate text-[10px] text-muted-foreground">
                    {presentation.description}
                  </span>
                </span>
                {selected ? <Check className="size-4 shrink-0 text-primary" /> : null}
              </button>
            );
          })
        ) : (
          <div className="px-3 py-5 text-center text-xs text-muted-foreground">
            {query ? "没有匹配的引用项" : mode === "project" ? "暂无可用项目" : "暂无已配置的 Skill 或 MCP"}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
