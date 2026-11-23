"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, Expand, FileText, Replace } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import type { CanvasNodeMediaGroup } from "@/lib/canvas/node-media";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { CanvasNodeInlinePlayer } from "./node-inline-player";

type Item = CanvasNodeMediaGroup["items"][number];
type Props = {
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
  onReplace?: (media: Item, trigger: HTMLElement) => void;
  canReplace?: (media: Item) => boolean;
};

export function CanvasNodeMedia({ groups, ...actions }: Props & { groups: CanvasNodeMediaGroup[] }) {
  return groups.map((group) => <MediaGroup key={group.label} group={group} {...actions} />);
}

function MediaGroup({ group, onPreview, onReplace, canReplace }: Props & { group: CanvasNodeMediaGroup }) {
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const itemKey = (item: Item) => `${item.source}:${item.field ?? ""}`;
  const index = Math.max(0, group.items.findIndex((item) => itemKey(item) === selectedSource));
  const media = group.items[index];
  const preview = (button: HTMLElement) => onPreview(media, button.closest<HTMLElement>(".react-flow__node") ?? button);
  const replaceable = group.label === "内容预览" && onReplace && canReplace?.(media);
  const header = group.items.length > 1 || group.label !== "内容预览" || !!media.field;
  return <section aria-label={group.label + " · " + group.items.length + " 项"} className="flex flex-col gap-2">
    {header && <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
      <span className="min-w-0 truncate" title={media.title}>{group.label}{media.field ? ` · ${media.title.split(" · ").at(-1)}` : ""}</span>
      <div className="nodrag nopan flex items-center gap-2" onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
        {group.items.length > 1 && <CanvasButton size="icon-xs" aria-label={"上一项" + group.label} disabled={index === 0}
          onClick={() => setSelectedSource(itemKey(group.items[index - 1]))}><ChevronLeft /></CanvasButton>}
        {group.items.length > 1 && <span aria-live="polite">{index + 1} / {group.items.length}</span>}
        {group.items.length > 1 && <CanvasButton size="icon-xs" aria-label={"下一项" + group.label} disabled={index + 1 === group.items.length}
          onClick={() => setSelectedSource(itemKey(group.items[index + 1]))}><ChevronRight /></CanvasButton>}
      </div>
    </div>}
    <div className="canvas-node-media-frame relative flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-border/20 bg-background/70"
      onDragStart={(event) => event.preventDefault()}>
      {media.kind === "video" || media.kind === "audio" ? <CanvasNodeInlinePlayer key={media.source} media={media} />
        : media.kind === "image" ? <MediaThumbnail source={media.source} kind={media.kind} alt={media.title} />
        : <FileText className="size-8 text-muted-foreground" />}
      <div className="nodrag nopan absolute right-2 top-2 flex gap-2 opacity-0 transition-opacity duration-150 group-hover/node:opacity-100 group-focus-within/node:opacity-100 group-data-selected/node:opacity-100 motion-reduce:transition-none [@media(hover:none)]:opacity-100" onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
        {replaceable && <CanvasIconButton size="icon-sm" variant="overlay" label="更换素材" aria-label={"更换" + media.title}
          onClick={(event) => onReplace(media, event.currentTarget)}><Replace /></CanvasIconButton>}
        <CanvasIconButton size="icon-sm" variant="overlay" label="放大预览" aria-label={"放大预览" + media.title}
          onClick={(event) => preview(event.currentTarget)}><Expand /></CanvasIconButton>
      </div>
    </div>
  </section>;
}
