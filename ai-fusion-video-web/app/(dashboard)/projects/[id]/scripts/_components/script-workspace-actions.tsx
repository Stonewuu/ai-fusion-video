"use client";

import { useState, type ReactNode } from "react";
import { AlertCircle, Import, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScriptImportDialog } from "@/components/script/script-import-dialog";
import type { Script, ScriptEpisode } from "@/lib/api/script";
import { duplicateTitles } from "@/lib/scripts/source-import";
import { ScriptParseAction } from "@/components/script/script-parse-action";

interface Props {
  script: Script;
  episodes: ScriptEpisode[];
  onImported: () => void;
  onAddEpisode: () => void;
  directoryControl: ReactNode;
  detailsControl: ReactNode;
}

export function ScriptWorkspaceActions({
  script,
  episodes,
  onImported,
  onAddEpisode,
  directoryControl,
  detailsControl,
}: Props) {
  const [importOpen, setImportOpen] = useState(false);
  const duplicates = duplicateTitles(episodes.map((episode) => episode.title));

  return (
    <>
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border/20 p-3">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {directoryControl}
          <div className="flex min-w-0 flex-col gap-1">
            <span className="text-sm font-semibold">
              剧本 · {episodes.length} 集
            </span>
            {duplicates.size > 0 && (
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <AlertCircle className="size-3.5" />
                存在重名分集，可在分集详情中修改标题
              </span>
            )}
          </div>
        </div>
        <div className="order-3 flex w-full flex-wrap items-center gap-2 sm:order-none sm:w-auto">
          <Button variant="ghost" onClick={onAddEpisode}>
            <Plus />
            新建分集
          </Button>
          <Button
            variant={episodes.length ? "outline" : "default"}
            onClick={() => setImportOpen(true)}
          >
            <Import />
            导入原文
          </Button>
          {episodes.length > 0 && (
            <ScriptParseAction script={script} onComplete={onImported} />
          )}
        </div>
        {detailsControl}
      </div>
      <ScriptImportDialog
        open={importOpen}
        scriptId={script.id}
        scriptVersion={script.version}
        hasLegacySource={!!script.rawContentLength}
        onClose={() => setImportOpen(false)}
        onImported={onImported}
      />
    </>
  );
}
