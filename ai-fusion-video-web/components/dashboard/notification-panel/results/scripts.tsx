"use client";

import {
  BookOpen,
  Clapperboard,
  Clock,
  Film,
  ListTree,
  MapPin,
  MessageSquare,
} from "lucide-react";
import { cn } from "@/lib/utils";

export function ScriptInfoResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const title = obj.title as string | undefined;
  const totalEpisodes = obj.totalEpisodes as number | undefined;
  const genre = obj.genre as string | undefined;
  const parsingStatus = obj.parsingStatus as number | undefined;
  const episodes = obj.episodes as Array<Record<string, unknown>> | undefined;

  const statusMap: Record<number, string> = {
    0: "未解析",
    1: "解析中",
    2: "已完成",
    3: "解析失败",
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground flex items-center gap-1.5">
          <BookOpen className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
          {title || "未命名剧本"}
        </span>
        {genre && (
          <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 text-[10px]">
            {genre}
          </span>
        )}
        {parsingStatus !== undefined && (
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-[10px]",
              parsingStatus === 2
                ? "bg-green-500/10 text-green-500"
                : "bg-yellow-500/10 text-yellow-500"
            )}
          >
            {statusMap[parsingStatus] || `状态${parsingStatus}`}
          </span>
        )}
      </div>
      <div className="flex gap-3 text-[10px] text-muted-foreground">
        <span>
          总集数: <span className="font-medium text-foreground">{totalEpisodes ?? 0}</span>
        </span>
        {obj.scriptId != null && <span>剧本ID: {String(obj.scriptId)}</span>}
      </div>
      {episodes && Array.isArray(episodes) && episodes.length > 0 && (
        <div className="space-y-0.5">
          <p className="text-[10px] text-muted-foreground/70">分集概览:</p>
          <ul className="space-y-0.5">
            {episodes.slice(0, 10).map((episode, index) => (
              <li
                key={episode.episodeId ? String(episode.episodeId) : index}
                className="flex items-center gap-2 text-xs text-muted-foreground/90"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/40 shrink-0" />
                <span className="font-medium text-foreground">
                  第{String(episode.episodeNumber || index + 1)}集
                </span>
                {episode.title != null && (
                  <span className="text-muted-foreground/70 truncate">
                    {String(episode.title)}
                  </span>
                )}
                {episode.totalScenes !== undefined && (
                  <span className="text-[10px] text-muted-foreground/50">
                    {String(episode.totalScenes)}场
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function ScriptStructureResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const title = obj.title as string | undefined;
  const totalEpisodes = obj.totalEpisodes as number | undefined;
  const episodes = obj.episodes as Array<Record<string, unknown>> | undefined;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground flex items-center gap-1.5">
          <ListTree className="h-3.5 w-3.5 text-blue-400 shrink-0" />
          {title || "剧本结构"}
        </span>
        <span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px]">
          {totalEpisodes ?? 0} 集
        </span>
      </div>
      {episodes && Array.isArray(episodes) && episodes.length > 0 && (
        <div className="space-y-1.5">
          {episodes.slice(0, 8).map((episode, index) => {
            const scenes = episode.scenes as Array<Record<string, unknown>> | undefined;
            return (
              <div key={episode.episodeId ? String(episode.episodeId) : index} className="space-y-0.5">
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/60 shrink-0" />
                  <span className="font-medium text-foreground">
                    第{String(episode.episodeNumber || index + 1)}集
                    {episode.title ? ` — ${String(episode.title)}` : ""}
                  </span>
                  {episode.totalScenes !== undefined && (
                    <span className="text-[10px] text-muted-foreground/50">
                      {String(episode.totalScenes)}场
                    </span>
                  )}
                </div>
                {scenes && scenes.length > 0 && (
                  <ul className="ml-4 space-y-0">
                    {scenes.slice(0, 6).map((scene, sceneIndex) => (
                      <li key={sceneIndex} className="text-[10px] text-muted-foreground/70 truncate">
                        · {String(scene.sceneHeading || `场次${sceneIndex + 1}`)}
                      </li>
                    ))}
                    {scenes.length > 6 && (
                      <li className="text-[10px] text-muted-foreground/50">
                        …还有 {scenes.length - 6} 个场次
                      </li>
                    )}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function EpisodeDetailResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const title = obj.title as string | undefined;
  const episodeNumber = obj.episodeNumber as number | undefined;
  const synopsis = obj.synopsis as string | undefined;
  const totalScenes = obj.totalScenes as number | undefined;
  const scenes = obj.scenes as Array<Record<string, unknown>> | undefined;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground flex items-center gap-1.5">
          <Film className="h-3.5 w-3.5 text-orange-400 shrink-0" />
          第{episodeNumber ?? "?"}集 {title || ""}
        </span>
        {totalScenes !== undefined && (
          <span className="px-1.5 py-0.5 rounded bg-orange-500/10 text-orange-400 text-[10px]">
            {totalScenes} 个场次
          </span>
        )}
        {obj.episode_version !== undefined && (
          <span className="text-[10px] text-muted-foreground/50">
            v{String(obj.episode_version)}
          </span>
        )}
      </div>
      {synopsis && (
        <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
          {synopsis.length > 150 ? `${synopsis.slice(0, 150)}…` : synopsis}
        </p>
      )}
      {scenes && Array.isArray(scenes) && scenes.length > 0 && (
        <ul className="space-y-0.5">
          {scenes.slice(0, 10).map((scene, index) => (
            <li
              key={scene.sceneItemId ? String(scene.sceneItemId) : index}
              className="flex items-center gap-2 text-xs text-muted-foreground/90"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-orange-400/40 shrink-0" />
              <span className="text-foreground truncate">
                {String(scene.sceneHeading || `场次${scene.sceneNumber || index + 1}`)}
              </span>
              {scene.characters != null && (
                <span className="text-[10px] text-muted-foreground/50 truncate">
                  {Array.isArray(scene.characters)
                    ? (scene.characters as string[]).slice(0, 3).join("、")
                    : String(scene.characters).slice(0, 30)}
                </span>
              )}
            </li>
          ))}
          {scenes.length > 10 && (
            <li className="text-[10px] text-muted-foreground/50 pl-3">
              …还有 {scenes.length - 10} 个场次
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

export function SceneDetailResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const heading = obj.sceneHeading as string | undefined;
  const location = obj.location as string | undefined;
  const timeOfDay = obj.timeOfDay as string | undefined;
  const intExt = obj.intExt as string | undefined;
  const description = obj.sceneDescription as string | undefined;
  const characters = obj.characters as unknown;
  const dialogues = obj.dialogues as unknown;

  const characterList = Array.isArray(characters) ? (characters as string[]) : [];
  const dialogueCount = Array.isArray(dialogues)
    ? (dialogues as unknown[]).length
    : 0;

  return (
    <div className="space-y-1.5">
      <span className="text-xs font-medium text-foreground flex items-center gap-1.5">
        <Clapperboard className="h-3.5 w-3.5 text-amber-400 shrink-0" />
        {heading || "场次详情"}
      </span>
      <div className="flex flex-wrap gap-1">
        {location && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-500 text-[10px]">
            <MapPin className="h-2.5 w-2.5" />
            {location}
          </span>
        )}
        {timeOfDay && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 text-[10px]">
            <Clock className="h-2.5 w-2.5" />
            {timeOfDay}
          </span>
        )}
        {intExt && (
          <span className="px-1.5 py-0.5 rounded bg-slate-500/10 text-slate-400 text-[10px]">
            {intExt}
          </span>
        )}
        {dialogueCount > 0 && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 text-[10px]">
            <MessageSquare className="h-2.5 w-2.5" />
            {dialogueCount} 条对白
          </span>
        )}
      </div>
      {characterList.length > 0 && (
        <div className="flex items-center gap-1 flex-wrap">
          <span className="text-[10px] text-muted-foreground/70">出场角色:</span>
          {characterList.slice(0, 6).map((character, index) => (
            <span
              key={index}
              className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px] text-foreground/80"
            >
              {String(character)}
            </span>
          ))}
          {characterList.length > 6 && (
            <span className="text-[10px] text-muted-foreground/50">
              +{characterList.length - 6}
            </span>
          )}
        </div>
      )}
      {description && (
        <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
          {description.length > 200 ? `${description.slice(0, 200)}…` : description}
        </p>
      )}
    </div>
  );
}