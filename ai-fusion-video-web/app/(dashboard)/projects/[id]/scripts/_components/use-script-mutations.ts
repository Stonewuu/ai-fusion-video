"use client";

import { useRef, type Dispatch, type SetStateAction } from "react";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { scriptApi, type Script, type ScriptEpisode, type SceneItem } from "@/lib/api/script";
import { toastApiError } from "@/lib/api/toast-api-error";
import { toast } from "sonner";

interface Options {
  script: Script | null;
  episodes: ScriptEpisode[];
  episodeScenes: Record<number, SceneItem[]>;
  setEpisodeScenes: Dispatch<SetStateAction<Record<number, SceneItem[]>>>;
  setExpandedEpisodes: Dispatch<SetStateAction<Set<number>>>;
  setActiveEpisodeId: Dispatch<SetStateAction<number | null>>;
  setSelectedSceneId: Dispatch<SetStateAction<number | null>>;
  refreshScript: () => Promise<void>;
  scrollToScene: (id: number) => void;
}

export function useScriptMutations(options: Options) {
  const { script, episodes, episodeScenes, refreshScript } = options;
  const { confirm } = useConfirm();
  const requests = useCommandRequestKey();
  const pending = useRef(false);

  async function run<T>(payload: unknown, action: (requestId: string) => Promise<T>, done?: (value: T) => void) {
    if (pending.current) { toast.info("正在保存，请稍候"); return; }
    pending.current = true;
    try {
      const result = await action(requests.forPayload(payload));
      requests.clear();
      await refreshScript();
      done?.(result);
    } catch (error) {
      toastApiError(error, "剧本操作未完成，请检查版本后重试");
    } finally {
      pending.current = false;
    }
  }

  const handleAddEpisode = async () => {
    if (!script) return;
    const episodeNumber = Math.max(0, ...episodes.map(episode => episode.episodeNumber)) + 1;
    const data = { scriptId: script.id, parentVersion: script.version, episodeNumber, title: `第 ${episodeNumber} 集`, sortOrder: episodes.length };
    await run({ operation: "createEpisode", ...data }, id => scriptApi.createEpisode(data, id), episode => {
      options.setExpandedEpisodes(previous => new Set(previous).add(episode.id));
      options.setActiveEpisodeId(episode.id);
    });
  };

  const handleDeleteEpisode = async (id: number) => {
    const episode = episodes.find(item => item.id === id);
    if (!episode) return;
    const version = episode.version;
    const accepted = await confirm({ title: "删除剧本分集", description: `将删除「${episode.title}」及其 ${episode.totalScenes} 个场次，已有引用会标记为失效。`, variant: "destructive", confirmText: "删除" });
    if (!accepted) return;
    await run({ operation: "deleteEpisode", id, version }, request => scriptApi.deleteEpisode(id, version, request), () => {
      options.setExpandedEpisodes(previous => { const next = new Set(previous); next.delete(id); return next; });
      options.setEpisodeScenes(previous => { const next = { ...previous }; delete next[id]; return next; });
      options.setSelectedSceneId(null);
    });
  };

  async function createScene(episodeId: number, atIndex?: number) {
    const episode = episodes.find(item => item.id === episodeId);
    if (!episode) return;
    const data = { episodeId, parentVersion: episode.version, sceneHeading: "新场次", ...(atIndex === undefined ? {} : { sortOrder: atIndex }) };
    await run({ operation: "createScene", ...data }, request => scriptApi.createScene(data, request), scene => {
      options.setSelectedSceneId(scene.id);
      requestAnimationFrame(() => options.scrollToScene(scene.id));
    });
  }

  const handleDeleteScene = async (id: number, episodeId: number) => {
    const scene = episodeScenes[episodeId]?.find(item => item.id === id);
    if (!scene) return;
    const version = scene.version;
    const accepted = await confirm({ title: "删除场次", description: `将删除场次「${scene.sceneHeading || scene.sceneNumber}」，已有引用会标记为失效。`, variant: "destructive", confirmText: "删除" });
    if (!accepted) return;
    await run({ operation: "deleteScene", id, version }, request => scriptApi.deleteScene(id, version, request), () => options.setSelectedSceneId(null));
  };

  const handleSceneUpdated = (episodeId: number, updated: SceneItem) => {
    options.setEpisodeScenes(previous => ({ ...previous, [episodeId]: (previous[episodeId] || []).map(scene => scene.id === updated.id ? updated : scene) }));
    void refreshScript();
  };

  const handleReorderScenes = async (episodeId: number, oldIndex: number, newIndex: number) => {
    const episode = episodes.find(item => item.id === episodeId);
    if (!episode) return;
    const scenes = [...(episodeScenes[episodeId] || [])];
    if (oldIndex < 0 || newIndex < 0 || oldIndex >= scenes.length || newIndex >= scenes.length || oldIndex === newIndex) return;
    const [moved] = scenes.splice(oldIndex, 1);
    scenes.splice(newIndex, 0, moved);
    const sceneIds = scenes.map(scene => scene.id);
    await run({ operation: "reorder", episodeId, version: episode.version, sceneIds }, request => scriptApi.reorderScenes(episodeId, episode.version, sceneIds, request));
  };

  return { handleAddEpisode, handleDeleteEpisode, handleAddScene: (id: number) => createScene(id), handleInsertScene: createScene,
    handleDeleteScene, handleSceneUpdated, handleReorderScenes };
}
