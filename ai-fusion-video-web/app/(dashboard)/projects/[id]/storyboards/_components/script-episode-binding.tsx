"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Link2, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ScriptEpisode } from "@/lib/api/script";

interface ScriptEpisodeBindingProps {
  episodes: ScriptEpisode[];
  boundEpisodeIds: Set<number>;
  suggestedEpisodeId?: number;
  isBinding: boolean;
  onBind: (scriptEpisodeId: number) => Promise<void> | void;
}

interface ScriptEpisodeBindingReminderProps {
  isOpen: boolean;
  isBinding: boolean;
  panelId: string;
  onToggle: () => void;
}

function getEpisodeLabel(episode: ScriptEpisode) {
  const title = episode.title?.trim();
  return title
    ? `第 ${episode.episodeNumber} 集 · ${title}`
    : `第 ${episode.episodeNumber} 集`;
}

export function ScriptEpisodeBindingReminder({
  isOpen,
  isBinding,
  panelId,
  onToggle,
}: ScriptEpisodeBindingReminderProps) {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="relative mr-1 shrink-0">
      {!isOpen && !isBinding && !shouldReduceMotion && (
        <motion.span
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-xl border border-amber-500/40"
          animate={{ opacity: [0, 0.28, 0], scale: [1, 1.14, 1.28] }}
          transition={{
            duration: 2.4,
            ease: "easeOut",
            repeat: Infinity,
            repeatDelay: 0.6,
            times: [0, 0.35, 1],
          }}
        />
      )}
      <Button
        type="button"
        variant="outline"
        size="xs"
        className="relative z-10"
        aria-controls={panelId}
        aria-expanded={isOpen}
        disabled={isBinding}
        title={isOpen ? "收起剧本集绑定" : "该分镜集尚未绑定剧本集"}
        onClick={(event) => {
          event.stopPropagation();
          onToggle();
        }}
      >
        {isBinding ? (
          <LoaderCircle className="animate-spin motion-reduce:animate-none" />
        ) : (
          <Link2 className="text-amber-600 dark:text-amber-300" />
        )}
        {isBinding ? "绑定中" : "待绑定"}
      </Button>
    </div>
  );
}

export function ScriptEpisodeBinding({
  episodes,
  boundEpisodeIds,
  suggestedEpisodeId,
  isBinding,
  onBind,
}: ScriptEpisodeBindingProps) {
  const options = episodes.map((episode) => {
    const isBound = boundEpisodeIds.has(episode.id);
    const isSuggested = episode.id === suggestedEpisodeId;
    const label = getEpisodeLabel(episode);

    return {
      value: String(episode.id),
      label: `${label}${isSuggested ? "（推荐）" : ""}${isBound ? "（已绑定）" : ""}`,
      isBound,
    };
  });
  const suggestedEpisode = episodes.find(
    (episode) => episode.id === suggestedEpisodeId
  );
  const hasAvailableEpisode = options.some((option) => !option.isBound);
  const placeholder = isBinding
    ? "正在绑定…"
    : suggestedEpisode
      ? `推荐：${getEpisodeLabel(suggestedEpisode)}`
      : hasAvailableEpisode
        ? "选择对应的剧本集"
        : "暂无可绑定的剧本集";

  return (
    <div
      data-slot="script-episode-binding"
      className="mt-1 mb-2 ml-8 mr-1 rounded-lg border border-amber-500/20 bg-amber-500/5 p-2.5"
    >
      <div className="mb-2 flex items-start gap-2">
        <div className="flex size-6 shrink-0 items-center justify-center rounded-lg bg-amber-500/10 text-amber-700 dark:text-amber-300">
          {isBinding ? (
            <LoaderCircle className="size-3.5 animate-spin motion-reduce:animate-none" />
          ) : (
            <Link2 className="size-3.5" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="text-[11px] font-medium leading-4 text-foreground/90">
              选择对应的剧本集
            </p>
          </div>
          <p className="mt-0.5 text-[10px] leading-4 text-muted-foreground">
            用于重新生成本集分镜
          </p>
        </div>
      </div>

      <Select
        value={null}
        onValueChange={(value) => {
          if (value != null) void onBind(Number(value));
        }}
        items={options.map(({ value, label }) => ({ value, label }))}
        disabled={isBinding || !hasAvailableEpisode}
      >
        <SelectTrigger
          size="sm"
          className="w-full bg-background/70 text-xs"
          aria-label="选择要绑定的剧本集"
        >
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent className="text-xs">
          <SelectGroup>
            {options.map((option) => (
              <SelectItem
                key={option.value}
                value={option.value}
                disabled={option.isBound}
                className="text-xs"
              >
                {option.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </div>
  );
}
