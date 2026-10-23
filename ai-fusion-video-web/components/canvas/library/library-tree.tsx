"use client";
import { memo, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Plus,
  BookOpen,
  Clapperboard,
  Box,
  Settings2,
  Loader2,
  FileText,
  Film,
  Video,
  Image as ImageIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { CanvasLibraryEntry } from "@/lib/canvas/resource-library";
import { cn } from "@/lib/utils";

interface Actions {
  onOpen: (entry: CanvasLibraryEntry) => void;
  onAdd: (entry: CanvasLibraryEntry) => void;
  disabled: boolean;
  busy: boolean;
  opening: string | null;
  activeKey: string | null;
}
const keyOf = (entry: CanvasLibraryEntry) =>
  `${entry.ref.type}:${entry.ref.id}`;
const sections = [
  { type: "script", label: "剧本", icon: BookOpen, color: "text-violet-600 dark:text-violet-300" },
  { type: "storyboard", label: "分镜", icon: Clapperboard, color: "text-blue-600 dark:text-blue-300" },
  { type: "asset", label: "资产", icon: Box, color: "text-amber-700 dark:text-amber-300" },
  { type: "project", label: "项目设定", icon: Settings2, color: "text-muted-foreground" },
] as const;
const rowIcons = { project: Settings2, script: BookOpen, scriptEpisode: FileText, scriptScene: FileText,
  storyboard: Clapperboard, storyboardEpisode: Film, storyboardScene: Film, shot: Video, asset: Box, assetItem: ImageIcon };

export const CanvasLibraryTree = memo(function CanvasLibraryTree({
  entries,
  search,
  ...actions
}: Actions & { entries: CanvasLibraryEntry[]; search: string }) {
  const children = new Map<string, CanvasLibraryEntry[]>();
  entries.forEach((entry) => {
    if (entry.parent)
      children.set(entry.parent, [
        ...(children.get(entry.parent) ?? []),
        entry,
      ]);
  });
  if (search.trim()) {
    const found = entries.filter((entry) =>
      `${entry.title} ${entry.subtitle}`
        .toLowerCase()
        .includes(search.trim().toLowerCase()),
    );
    return (
      <div className="space-y-1">
        {found.map((entry) => (
          <LibraryRow key={keyOf(entry)} entry={entry} showType {...actions} />
        ))}
        {!found.length && (
          <p className="text-sm text-muted-foreground">没有匹配的内容。</p>
        )}
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {sections.map((section) => {
        const roots = entries.filter(
          (entry) => entry.ref.type === section.type,
        );
        if (!roots.length) return null;
        return (
          <LibrarySection key={section.type} section={section} roots={roots} childrenByParent={children} {...actions} />
        );
      })}
    </div>
  );
});

function LibrarySection({ section, roots, childrenByParent, ...actions }: Actions & {
  section: typeof sections[number]; roots: CanvasLibraryEntry[];
  childrenByParent: Map<string, CanvasLibraryEntry[]>;
}) {
  const [open, setOpen] = useState(section.type !== "project");
  return (
    <section className="space-y-1">
      <h3>
        <button type="button" className="flex h-8 w-full items-center gap-2 rounded-lg px-1 text-left text-sm font-semibold outline-none transition-colors hover:bg-muted/50 focus-visible:ring-2 focus-visible:ring-ring"
          aria-expanded={open} onClick={() => setOpen(!open)}>
          <section.icon className={cn("size-4 shrink-0", section.color)} />
          {section.label}
          <span className="ml-auto text-xs font-normal text-muted-foreground">{roots.length}</span>
          {open ? <ChevronDown className="size-3.5 text-muted-foreground" /> : <ChevronRight className="size-3.5 text-muted-foreground" />}
        </button>
      </h3>
      {open && <div className="space-y-1">
        {roots.map((entry) => <LibraryBranch key={keyOf(entry)} entry={entry} childrenByParent={childrenByParent} depth={0} {...actions} />)}
      </div>}
    </section>
  );
}

function LibraryBranch({
  entry,
  childrenByParent,
  depth,
  ...actions
}: Actions & {
  entry: CanvasLibraryEntry;
  childrenByParent: Map<string, CanvasLibraryEntry[]>;
  depth: number;
}) {
  const [open, setOpen] = useState(depth === 0);
  const children = childrenByParent.get(keyOf(entry)) ?? [];
  return (
    <div>
      <LibraryRow
        entry={entry}
        {...actions}
        leading={
          children.length ? (
            <Button
              variant="ghost"
              size="icon-xs"
              title={open ? "折叠下级内容" : "展开下级内容"}
              aria-expanded={open}
              onClick={() => setOpen(!open)}
            >
              {open ? <ChevronDown /> : <ChevronRight />}
            </Button>
          ) : (
            <span className="w-6 shrink-0" />
          )
        }
      />
      {open && children.length > 0 && (
        <div className="ml-3 border-l border-border/20 pl-1">
          {children.map((child) => (
            <LibraryBranch
              key={keyOf(child)}
              entry={child}
              childrenByParent={childrenByParent}
              depth={depth + 1}
              {...actions}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function LibraryRow({
  entry,
  onOpen,
  onAdd,
  disabled,
  busy,
  opening,
  activeKey,
  showType = false,
  leading,
}: Actions & { entry: CanvasLibraryEntry; leading?: React.ReactNode; showType?: boolean }) {
  const Icon = rowIcons[entry.ref.type];
  const selected = activeKey === keyOf(entry);
  return (
    <div
      className={cn("group flex items-center gap-1 rounded-lg transition-colors", selected ? "bg-primary/10 text-primary ring-1 ring-primary/20" : "hover:bg-muted/40")}
      draggable={!disabled && !busy}
      onDragStart={(event) => {
        if (disabled || busy) {
          event.preventDefault();
          return;
        }
        event.dataTransfer.setData(
          "application/x-fusion-resource",
          JSON.stringify(entry),
        );
        event.dataTransfer.effectAllowed = "copy";
      }}
    >
      {leading}
      <button
        type="button"
        onClick={() => onOpen(entry)}
        title={`${entry.subtitle} · ${entry.title}`}
        aria-label={`查看${entry.title}，${entry.subtitle}`}
        aria-current={selected ? "true" : undefined}
        className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-lg px-1 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Icon className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate text-sm">{entry.title}</span>
        {showType && <span className="shrink-0 text-xs text-muted-foreground">{entry.subtitle}</span>}
      </button>
      {opening === keyOf(entry) ? (
        <Loader2 className="mx-1 size-4 animate-spin text-muted-foreground motion-reduce:animate-none" />
      ) : (
        <span className="opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(pointer:coarse)]:opacity-100 motion-reduce:transition-none">
        <Button
          variant="ghost"
          size="icon-xs"
          title={`将${entry.title}放入画布`}
          disabled={disabled}
          aria-disabled={busy || undefined}
          onClick={() => {
            if (!busy) onAdd(entry);
          }}
        >
          <Plus />
        </Button>
        </span>
      )}
    </div>
  );
}
