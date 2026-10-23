"use client";

import type { Storyboard, StoryboardItem } from "@/lib/api/storyboard";
import { Info } from "lucide-react";
import { BatchFrameGenerateControl } from "./batch-frame-generate";
import { BatchFrameGenerateHandler } from "./reference-shared";

export function StoryboardOverview({
  storyboard,
  items,
  projectName,
  onBatchGenerateFrames,
}: {
  storyboard: Storyboard;
  items: StoryboardItem[];
  projectName?: string;
  onBatchGenerateFrames?: BatchFrameGenerateHandler;
}) {
  const totalDuration = items.reduce(
    (sum, item) => sum + (item.duration || 0),
    0,
  );
  const withImage = items.filter(
    (i) =>
      i.firstFrameImageUrl ||
      i.lastFrameImageUrl ||
      i.imageUrl ||
      i.generatedImageUrl ||
      i.referenceImageUrl,
  ).length;

  return (
    <div className="p-4 space-y-5">
      <div>
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Info className="h-3 w-3" /> 分镜概览
        </h4>
        <p className="text-sm font-semibold mb-1">
          {projectName || "未命名项目"}
        </p>
        {storyboard.description && (
          <p className="text-xs text-muted-foreground leading-relaxed">
            {storyboard.description}
          </p>
        )}
      </div>

      <BatchFrameGenerateControl
        items={[]}
        currentEpisodeId={null}
        currentSceneId={null}
        onConfirm={onBatchGenerateFrames}
      />

      {/* 统计 */}
      <div className="border-t border-border/20 pt-4">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
          统计
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-primary">{items.length}</p>
            <p className="text-[10px] text-muted-foreground">总镜头数</p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-cyan-400">
              {totalDuration > 0 ? `${totalDuration}s` : "—"}
            </p>
            <p className="text-[10px] text-muted-foreground">总时长</p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-amber-400">{withImage}</p>
            <p className="text-[10px] text-muted-foreground">有画面</p>
          </div>
          <div className="text-center p-2.5 rounded-lg bg-muted/20">
            <p className="text-lg font-bold text-violet-400">
              {items.length - withImage}
            </p>
            <p className="text-[10px] text-muted-foreground">无画面</p>
          </div>
        </div>
      </div>
    </div>
  );
}
