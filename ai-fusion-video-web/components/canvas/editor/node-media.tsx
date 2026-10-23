"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, Expand, FileText, Replace } from "lucide-react";
import { Button } from "@/components/ui/button";
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
  return <section aria-label={group.label + " · " + group.items.length + " 项"} className="flex flex-col gap-2">
    <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
      <span className="min-w-0 truncate" title={media.title}>{group.label}{media.field ? ` · ${media.title.split(" · ").at(-1)}` : ""}</span>
      <div className="nodrag nopan flex items-center gap-2" onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
        {group.items.length > 1 && <Button variant="ghost" size="icon-xs" aria-label={"上一项" + group.label} disabled={index === 0}
          onClick={() => setSelectedSource(itemKey(group.items[index - 1]))}><ChevronLeft /></Button>}
        {group.items.length > 1 && <span aria-live="polite">{index + 1} / {group.items.length}</span>}
        {group.items.length > 1 && <Button variant="ghost" size="icon-xs" aria-label={"下一项" + group.label} disabled={index + 1 === group.items.length}
          onClick={() => setSelectedSource(itemKey(group.items[index + 1]))}><ChevronRight /></Button>}
      </div>
    </div>
    <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-border/20 bg-background/70"
      onDragStart={(event) => event.preventDefault()}>
      {media.kind === "video" || media.kind === "audio" ? <CanvasNodeInlinePlayer key={media.source} media={media} />
        : media.kind === "image" ? <MediaThumbnail source={media.source} kind={media.kind} alt={media.title} />
        : <FileText className="size-8 text-muted-foreground" />}
      <div className="nodrag nopan absolute right-2 top-2 flex gap-2" onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
        {replaceable && <Button size="icon-sm" variant="secondary" aria-label={"更换" + media.title} title="更换素材"
          onClick={(event) => onReplace(media, event.currentTarget)}><Replace /></Button>}
        <Button size="icon-sm" variant="secondary" aria-label={"放大预览" + media.title} title="放大预览"
          onClick={(event) => preview(event.currentTarget)}><Expand /></Button>
      </div>
    </div>
  </section>;
}
