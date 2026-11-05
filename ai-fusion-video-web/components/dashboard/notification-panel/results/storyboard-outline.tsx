"use client";

type OutlineEntry = Record<string, unknown>;

export function StoryboardOutlineResult({ data }: { data: OutlineEntry }) {
  const episodes = data.episodes as OutlineEntry[];

  return (
    <div className="flex flex-col gap-3 text-xs">
      <div className="flex flex-col gap-1">
        <p className="font-medium text-foreground">{data.title as string}</p>
        <p className="text-muted-foreground">
          {episodes.length} 集 · 共 {Number(data.totalItems).toLocaleString("zh-CN")} 个镜头 · 集与场次目录
        </p>
      </div>
      {episodes.map((episode) => {
        const scenes = episode.scenes as OutlineEntry[];
        return (
          <div key={String(episode.id)} className="flex flex-col gap-1">
            <p className="font-medium text-foreground">
              {episode.title as string} <span className="font-normal text-muted-foreground">· {scenes.length} 场 · {Number(episode.itemCount)} 个镜头</span>
            </p>
            {scenes.map((scene) => (
              <p key={String(scene.id)} className="truncate text-muted-foreground" title={scene.sceneHeading as string}>
                {scene.sceneNumber as string} {scene.sceneHeading as string} · {Number(scene.itemCount)} 个镜头
              </p>
            ))}
          </div>
        );
      })}
      {episodes.length === 0 && <p className="text-muted-foreground">暂无分集</p>}
      {Number(data.unassignedItems) > 0 && <p className="text-muted-foreground">{Number(data.unassignedItems)} 个镜头尚未归入分集</p>}
    </div>
  );
}
