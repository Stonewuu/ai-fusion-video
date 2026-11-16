import { useLayoutEffect, useMemo, useRef } from "react";

export interface EpisodeTreeActions {
  onToggleEpisode: (id: number) => void;
  onSelectEpisode: (id: number) => void;
  onSelectScene: (sceneId: number, episodeId: number) => void;
  onAddEpisode: () => void;
  onDeleteEpisode: (id: number) => void;
  onAddScene: (episodeId: number) => void;
  onDeleteScene: (sceneId: number, episodeId: number) => void;
  onReorderScenes?: (
    episodeId: number,
    oldIndex: number,
    newIndex: number,
  ) => void;
  onParseEpisode?: (episodeId: number) => void;
  onGenerateStoryboard?: (episodeId: number) => void;
}

export function useEpisodeTreeActions(
  callbacks: EpisodeTreeActions,
): EpisodeTreeActions {
  const current = useRef(callbacks);
  useLayoutEffect(() => {
    current.current = callbacks;
  });
  // 行组件复用固定的回调，执行时仍使用页面最新的版本和场次数据。
  return useMemo(
    () => ({
      onToggleEpisode: (id) => current.current.onToggleEpisode(id),
      onSelectEpisode: (id) => current.current.onSelectEpisode(id),
      onSelectScene: (sceneId, episodeId) =>
        current.current.onSelectScene(sceneId, episodeId),
      onAddEpisode: () => current.current.onAddEpisode(),
      onDeleteEpisode: (id) => current.current.onDeleteEpisode(id),
      onAddScene: (id) => current.current.onAddScene(id),
      onDeleteScene: (sceneId, episodeId) =>
        current.current.onDeleteScene(sceneId, episodeId),
      onReorderScenes: (id, from, to) =>
        current.current.onReorderScenes?.(id, from, to),
      onParseEpisode: (id) => current.current.onParseEpisode?.(id),
      onGenerateStoryboard: (id) => current.current.onGenerateStoryboard?.(id),
    }),
    [],
  );
}
