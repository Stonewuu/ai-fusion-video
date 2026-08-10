"use client";

import { cn } from "@/lib/utils";
import type { Script, ScriptEpisode } from "@/lib/api/script";

export function ScriptOverview({
  script,
  episodes,
  projectName,
}: {
  script: Script;
  episodes: ScriptEpisode[];
  projectName?: string;
}) {
  // 解析角色
  let characters: Array<{ name: string; description?: string; importance?: string }> = [];
  if (script.charactersJson) {
    try {
      characters =
        typeof script.charactersJson === "string"
          ? JSON.parse(script.charactersJson)
          : script.charactersJson;
    } catch {
      /* 忽略 */
    }
  }

  return (
    <div className="p-4 space-y-5">
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          剧本信息
        </h4>
        <div className="space-y-2">
          {[
            { label: "项目", value: projectName },
            { label: "类型", value: script.genre },
            { label: "集数", value: String(episodes.length) },
            {
              label: "总场次",
              value: String(episodes.reduce((s, e) => s + e.totalScenes, 0)),
            },
          ].map(({ label, value }) => (
            <div key={label} className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">{label}</span>
              <span className="font-medium truncate ml-2 max-w-[120px]">
                {value || "—"}
              </span>
            </div>
          ))}
        </div>
      </div>
      {script.storySynopsis && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            故事梗概
          </h4>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {script.storySynopsis}
          </p>
        </div>
      )}
      {characters.length > 0 && (
        <div className="border-t border-border/20 pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            人物表
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {characters.map((c, i) => (
              <span
                key={i}
                className={cn(
                  "text-[10px] px-2 py-0.5 rounded-lg border",
                  c.importance === "主角"
                    ? "bg-primary/10 border-primary/20 text-primary"
                    : c.importance === "配角"
                    ? "bg-blue-500/10 border-blue-500/20 text-blue-400"
                    : "bg-muted/50 border-border/30 text-muted-foreground"
                )}
              >
                {c.name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
