import { projectApi } from "@/lib/api/project";
import { scriptApi } from "@/lib/api/script";
import { storyboardApi } from "@/lib/api/storyboard";
import { assetApi } from "@/lib/api/asset";
import { canvasTypeNames } from "./labels";
import type { ResourceType } from "@/lib/api/resource-events";
import type { CanvasResourceRef } from "./types";

export interface CanvasLibraryEntry {
  ref: CanvasResourceRef;
  title: string;
  parent: string | null;
  subtitle: string;
  assetType?: string;
  ordinal?: number;
  relations?: { key: string; label: string }[];
}
function entry(
  type: ResourceType,
  data: { id: number; version: number },
  title: string,
  parent: string | null = null,
): CanvasLibraryEntry {
  return {
    ref: { type, id: data.id, version: data.version },
    title,
    parent,
    subtitle: canvasTypeNames[type],
  };
}
export async function loadCanvasLibrary(
  projectId: number | null,
): Promise<CanvasLibraryEntry[]> {
  const assets = await assetApi.listWithItems(projectId);
  const rows: CanvasLibraryEntry[] = [];
  for (const asset of assets) {
    if (projectId === null && asset.projectId != null) continue;
    rows.push({ ...entry("asset", asset, asset.name), assetType: asset.type });
    asset.items.forEach((item) =>
      rows.push({
        ...entry(
          "assetItem",
          item,
          item.name || "未命名变体",
          `asset:${asset.id}`,
        ),
        assetType: asset.type,
      }),
    );
  }
  if (projectId === null) return rows;
  const [project, script, storyboard] = await Promise.all([
    projectApi.get(projectId),
    scriptApi.getByProject(projectId),
    storyboardApi.getByProject(projectId),
  ]);
  rows.unshift(entry("project", project, project.name));
  if (script) {
    rows.push(entry("script", script, script.title));
    const episodes = await scriptApi.listEpisodes(script.id);
    const children = await Promise.all(
      episodes.map((episode) => scriptApi.listScenes(episode.id)),
    );
    episodes.forEach((episode, index) => {
      rows.push({
        ...entry(
          "scriptEpisode",
          episode,
          `第 ${episode.episodeNumber} 集 · ${episode.title || "未命名"}`,
          `script:${script.id}`,
        ),
        ordinal: episode.episodeNumber,
      });
      children[index].forEach((scene) =>
        rows.push({
          ...entry(
            "scriptScene",
            scene,
            scene.sceneHeading || scene.sceneNumber || `场次 ${scene.id}`,
            `scriptEpisode:${episode.id}`,
          ),
          relations: [
            ...(scene.characterAssetIds ?? []).map((id) => ({
              key: `asset:${id}`,
              label: "角色",
            })),
            ...(scene.propAssetIds ?? []).map((id) => ({
              key: `asset:${id}`,
              label: "道具",
            })),
            ...(scene.sceneAssetId
              ? [{ key: `asset:${scene.sceneAssetId}`, label: "场景" }]
              : []),
          ],
        }),
      );
    });
  }
  if (storyboard) {
    rows.push(entry("storyboard", storyboard, storyboard.title || "分镜"));
    const [episodes, scenes, shots] = await Promise.all([
      storyboardApi.listEpisodes(storyboard.id),
      storyboardApi.listScenesByStoryboard(storyboard.id),
      storyboardApi.listItems(storyboard.id),
    ]);
    episodes.forEach((episode) =>
      rows.push({
        ...entry(
          "storyboardEpisode",
          episode,
          `第 ${episode.episodeNumber} 集 · ${episode.title || "未命名"}`,
          `storyboard:${storyboard.id}`,
        ),
        ordinal: episode.episodeNumber ?? undefined,
        relations: episode.scriptEpisodeId
          ? [
              {
                key: `scriptEpisode:${episode.scriptEpisodeId}`,
                label: "来自剧本集",
              },
            ]
          : [],
      }),
    );
    scenes.forEach((scene) =>
      rows.push(
        entry(
          "storyboardScene",
          scene,
          scene.sceneHeading || `场次 ${scene.id}`,
          `storyboardEpisode:${scene.episodeId}`,
        ),
      ),
    );
    shots.forEach((shot) =>
      rows.push({
        ...entry(
          "shot",
          shot,
          `${shot.shotNumber || "镜头"} · ${(shot.content || "未填写画面").slice(0, 40)}`,
          shot.storyboardSceneId
            ? `storyboardScene:${shot.storyboardSceneId}`
            : `storyboardEpisode:${shot.storyboardEpisodeId}`,
        ),
        relations: [
          ...(shot.characterIds ?? []).map((id) => ({
            key: `assetItem:${id}`,
            label: "角色",
          })),
          ...(shot.propIds ?? []).map((id) => ({
            key: `assetItem:${id}`,
            label: "道具",
          })),
          ...(shot.sceneAssetItemId
            ? [{ key: `assetItem:${shot.sceneAssetItemId}`, label: "场景" }]
            : []),
        ],
      }),
    );
  }
  return rows;
}
