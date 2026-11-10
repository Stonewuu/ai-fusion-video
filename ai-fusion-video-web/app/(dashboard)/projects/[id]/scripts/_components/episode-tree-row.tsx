"use client";

import { ChevronRight, Film, Sparkles, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ScriptEpisode } from "@/lib/api/script";
import { cn } from "@/lib/utils";

interface EpisodeTreeRowProps {
  episode: ScriptEpisode;
  expanded: boolean;
  active: boolean;
  onToggle: () => void;
  onSelect: () => void;
  onDelete: () => void;
  onParse?: () => void;
  onGenerateStoryboard?: () => void;
}

export function EpisodeTreeRow({ episode, expanded, active, onToggle, onSelect, onDelete, onParse,
  onGenerateStoryboard }: EpisodeTreeRowProps) {
  const title = episode.title || `第 ${episode.episodeNumber} 集`;
  return (
    <div data-sidebar-row className={cn("group/episode-row flex min-w-0 items-center rounded-xl p-1",
      active ? "bg-primary/8" : "hover:bg-primary/5")}>
      <Button type="button" variant="ghost" size="icon-xs" onClick={onToggle} aria-expanded={expanded}
        aria-label={`${expanded ? "收起" : "展开"}${title}`}>
        <ChevronRight className={cn("transition-transform duration-150 motion-reduce:transition-none", expanded && "rotate-90")} />
      </Button>
      <Tooltip>
        <TooltipTrigger render={
          <button type="button" onClick={() => {
            if (!expanded) {
              onToggle();
            }
            onSelect();
          }} className={cn("flex h-8 min-w-0 flex-1 items-center gap-2 rounded-xl px-2 text-left text-xs outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
            active ? "font-medium text-primary" : "text-foreground/80 hover:text-foreground")}>
            <Film className="size-3.5 shrink-0" aria-hidden="true" />
            <span className="min-w-0 truncate font-medium">{title}</span>
          </button>
        } />
        <TooltipContent side="right" align="start" className="max-w-xs whitespace-normal break-words">
          {title}
        </TooltipContent>
      </Tooltip>
      <div className="hidden shrink-0 items-center gap-2 group-hover/episode-row:flex group-has-[:focus-visible]/episode-row:flex [@media(hover:none)]:flex">
        {onParse && <Button type="button" variant="ghost" size="icon-xs" onClick={onParse} title="AI 解析该集" aria-label="AI 解析该集"><Sparkles /></Button>}
        {onGenerateStoryboard && <Button type="button" variant="ghost" size="icon-xs" onClick={onGenerateStoryboard} title="AI 生成该集分镜" aria-label="AI 生成该集分镜"><Film /></Button>}
        <Button type="button" variant="destructive-ghost" size="icon-xs" onClick={onDelete} title="删除分集" aria-label="删除分集"><Trash2 /></Button>
      </div>
    </div>
  );
}
