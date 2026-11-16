"use client";

import { memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SceneItem, ScriptEpisode } from "@/lib/api/script";
import type { EpisodeTreeActions } from "./episode-tree-actions";
import { EpisodeTreeRow } from "./episode-tree-row";
import { EpisodeTreeScenes } from "./episode-tree-scenes";

const EMPTY_SCENES: SceneItem[] = [];

export const EpisodeTreeBlock = memo(function EpisodeTreeBlock({
  episode,
  duplicate,
  expanded,
  active,
  loading,
  scenes = EMPTY_SCENES,
  selectedSceneId,
  actions,
  canParse,
  canGenerateStoryboard,
  onDraggingChange,
  reduceMotion,
}: {
  episode: ScriptEpisode;
  duplicate: boolean;
  expanded: boolean;
  active: boolean;
  loading: boolean;
  scenes?: SceneItem[];
  selectedSceneId: number | null;
  actions: EpisodeTreeActions;
  canParse: boolean;
  canGenerateStoryboard: boolean;
  onDraggingChange: (id: number | null) => void;
  reduceMotion: boolean;
}) {
  return (
    <div className="pb-0.5">
      <EpisodeTreeRow
        episode={episode}
        duplicate={duplicate}
        expanded={expanded}
        active={active}
        onToggle={() => actions.onToggleEpisode(episode.id)}
        onSelect={() => actions.onSelectEpisode(episode.id)}
        onDelete={() => actions.onDeleteEpisode(episode.id)}
        onParse={
          canParse ? () => actions.onParseEpisode?.(episode.id) : undefined
        }
        onGenerateStoryboard={
          canGenerateStoryboard
            ? () => actions.onGenerateStoryboard?.(episode.id)
            : undefined
        }
      />
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={reduceMotion ? false : { height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="ml-3 border-l border-primary/15 py-1 pl-2 pr-1">
              {loading ? (
                <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" />
                  加载中...
                </div>
              ) : scenes.length === 0 ? (
                <p className="px-3 py-2 text-xs text-muted-foreground">
                  暂无场次
                </p>
              ) : (
                <EpisodeTreeScenes
                  episodeId={episode.id}
                  scenes={scenes}
                  selectedSceneId={selectedSceneId}
                  actions={actions}
                  onDraggingChange={onDraggingChange}
                  reduceMotion={reduceMotion}
                />
              )}
              <Button
                variant="ghost"
                size="xs"
                onClick={() => actions.onAddScene(episode.id)}
                className="w-full justify-start"
              >
                <Plus />
                添加场次
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});
