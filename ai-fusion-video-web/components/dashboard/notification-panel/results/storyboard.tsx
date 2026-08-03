"use client";

import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { StoryboardShotList } from "./storyboard-shot-list";

export function StoryboardResult({
  data,
  toolName,
  expanded = false,
}: {
  data: unknown;
  toolName: string;
  expanded?: boolean;
}) {
  const obj = data as Record<string, unknown>;
  const isSave = toolName.startsWith("save_") || toolName === "insert_storyboard_item";

  if (isSave) {
    const status = obj.status as string | undefined;
    const message = obj.message as string | undefined;
    const sceneId = (obj.storyboardSceneId ?? obj.sceneId) as number | undefined;
    const sceneNumber = obj.sceneNumber as string | undefined;
    const shotCount = obj.shotCount as number | undefined;
    const episodeId = (obj.storyboardEpisodeId ?? obj.episodeId) as number | undefined;
    const sceneCount = obj.sceneCount as number | undefined;

    return (
      <div className="space-y-1">
        <p className="inline-flex items-center gap-1 text-xs text-muted-foreground">
          {status === "error" ? (
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-yellow-500" />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-green-500" />
          )}
          {message || (status === "error" ? "保存失败" : "保存成功")}
        </p>
        {sceneNumber || sceneId !== undefined ? (
          <p className="text-[10px] text-muted-foreground/60">
            {sceneNumber ? `场次 ${sceneNumber}` : `场次 ID ${sceneId}`}
            {shotCount !== undefined && ` · ${shotCount} 个镜头`}
          </p>
        ) : null}
        {episodeId !== undefined ? (
          <p className="text-[10px] text-muted-foreground/60">
            集 ID: {episodeId}
            {sceneCount !== undefined && ` · ${sceneCount} 个场次`}
          </p>
        ) : null}
      </div>
    );
  }

  const storyboards = obj.storyboards as Array<Record<string, unknown>> | undefined;
  if (storyboards) {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground">
          共 <span className="font-medium text-foreground">{storyboards.length}</span> 个分镜脚本
        </p>
        {storyboards.slice(0, 3).map((storyboard, index) => (
          <div
            key={index}
            className="flex items-center gap-2 text-xs text-muted-foreground/90"
          >
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-blue-400/60" />
            <span className="truncate">
              {(storyboard.title as string) || `分镜 #${storyboard.id || index + 1}`}
              {storyboard.totalItems !== undefined && ` · ${storyboard.totalItems} 项`}
            </span>
          </div>
        ))}
      </div>
    );
  }

  const sceneName = obj.sceneName as string | undefined;
  const title = obj.title as string | undefined;
  const description = obj.description as string | undefined;
  const totalItems = obj.totalItems as number | undefined;
  const items = obj.items as Array<Record<string, unknown>> | undefined;
  const storyboardId = obj.storyboardId ?? obj.id;
  const sceneId = obj.storyboardSceneId ?? obj.sceneId;
  const shotItems = Array.isArray(items) ? items : [];
  const displayTitle = title || sceneName;

  return (
    <div className="space-y-1.5">
      <div className="space-y-0.5">
        {displayTitle ? (
          <p className="text-xs font-medium text-foreground">{displayTitle}</p>
        ) : null}
        <p className="text-[10px] text-muted-foreground/60">
          {storyboardId !== undefined && `ID: ${storyboardId}`}
          {storyboardId === undefined && sceneId !== undefined && `场次 ID: ${sceneId}`}
          {totalItems !== undefined && ` · 共 ${totalItems} 个镜头`}
          {shotItems.length > 0 && (expanded ? " · 增量加载" : ` · 预览 ${Math.min(shotItems.length, 3)} 项`)}
        </p>
        {description ? (
          <p className="line-clamp-1 text-[10px] leading-4 text-muted-foreground/70">
            {description}
          </p>
        ) : null}
      </div>

      {shotItems.length > 0 ? (
        <StoryboardShotList
          key={expanded ? "expanded" : "collapsed"}
          items={shotItems}
          expanded={expanded}
        />
      ) : (
        <p className="text-[11px] text-muted-foreground/70">暂无镜头数据</p>
      )}
    </div>
  );
}
