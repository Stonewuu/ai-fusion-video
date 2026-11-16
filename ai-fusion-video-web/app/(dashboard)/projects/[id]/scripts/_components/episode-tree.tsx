"use client";
import { useCallback, useMemo, useState, type ReactNode } from "react";
import { useReducedMotion } from "framer-motion";
import { Plus, RotateCw } from "lucide-react";
import { DirectoryTooltipScope } from "@/components/shared/directory-tooltip";
import { VirtualDirectory } from "@/components/shared/virtual-directory";
import { Button } from "@/components/ui/button";
import type { ScriptEpisode, SceneItem } from "@/lib/api/script";
import { duplicateTitles, titleKey } from "@/lib/scripts/source-import";
import { cn } from "@/lib/utils";
import { EpisodeTreeBlock } from "./episode-tree-block";
import {
  useEpisodeTreeActions,
  type EpisodeTreeActions,
} from "./episode-tree-actions";

interface Props extends EpisodeTreeActions {
  episodes: ScriptEpisode[];
  expandedEpisodes: Set<number>;
  activeEpisodeId: number | null;
  selectedSceneId: number | null;
  loadingEpisodes: Set<number>;
  episodeScenes: Record<number, SceneItem[]>;
  headerAction?: ReactNode;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export function EpisodeTree(props: Props) {
  const {
    episodes,
    expandedEpisodes,
    activeEpisodeId,
    selectedSceneId,
    loadingEpisodes,
    episodeScenes,
    headerAction,
    onRefresh,
    isRefreshing = false,
  } = props;
  const reduceMotion = !!useReducedMotion();
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const actions = useEpisodeTreeActions(props);
  const repeated = useMemo(
    () => duplicateTitles(episodes.map((episode) => episode.title)),
    [episodes],
  );
  const estimateSize = useCallback(
    (index: number) => {
      const episode = episodes[index];
      return expandedEpisodes.has(episode.id)
        ? 74 +
            Math.max(
              1,
              episodeScenes[episode.id]?.length ?? episode.totalScenes,
            ) *
              36
        : 42;
    },
    [episodes, expandedEpisodes, episodeScenes],
  );

  return (
    <DirectoryTooltipScope>
      <div
        data-sidebar-navigation="scripts"
        className="flex h-full min-h-0 w-full flex-col bg-card/20"
      >
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border/20 p-3">
          <h3 className="text-sm font-semibold">剧本目录</h3>
          <div className="flex items-center gap-2">
            {onRefresh && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onRefresh}
                disabled={isRefreshing}
                title="刷新剧本目录"
                aria-label="刷新剧本目录"
              >
                <RotateCw className={cn(isRefreshing && "animate-spin")} />
              </Button>
            )}
            <span className="rounded-full bg-muted px-1.5 py-0.5 text-xs font-medium tabular-nums text-muted-foreground">
              {episodes.length} 集
            </span>
            {headerAction}
          </div>
        </div>
        <VirtualDirectory
          items={episodes}
          activeId={activeEpisodeId}
          expandedIds={expandedEpisodes}
          draggingId={draggingId}
          estimateSize={estimateSize}
          onToggle={actions.onToggleEpisode}
          label="分集列表"
        >
          {(episode) => (
            <EpisodeTreeBlock
              episode={episode}
              duplicate={repeated.has(titleKey(episode.title))}
              expanded={expandedEpisodes.has(episode.id)}
              active={activeEpisodeId === episode.id}
              loading={loadingEpisodes.has(episode.id)}
              scenes={episodeScenes[episode.id]}
              selectedSceneId={
                activeEpisodeId === episode.id ? selectedSceneId : null
              }
              actions={actions}
              canParse={!!props.onParseEpisode}
              canGenerateStoryboard={!!props.onGenerateStoryboard}
              onDraggingChange={setDraggingId}
              reduceMotion={reduceMotion}
            />
          )}
        </VirtualDirectory>
        <div className="shrink-0 border-t border-border/20 px-3 py-2">
          <Button
            variant="ghost"
            onClick={actions.onAddEpisode}
            className="w-full"
          >
            <Plus />
            新建分集
          </Button>
        </div>
      </div>
    </DirectoryTooltipScope>
  );
}
