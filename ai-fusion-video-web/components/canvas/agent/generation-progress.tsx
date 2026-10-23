"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { GenerationExecutionDialog } from "@/components/dashboard/generation/generation-execution-dialog";
import { generationStateLabels } from "@/lib/generation-execution-view";
import type { CanvasAgentExecution } from "@/lib/api/canvas-agents";

export function CanvasAgentGenerationProgress({
  generations,
  onChanged,
}: {
  generations: CanvasAgentExecution["generations"];
  onChanged: () => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  if (!generations.length) return null;
  return (
    <section
      className="flex flex-col gap-3 border-t border-border/20 pt-4"
      aria-label="智能体生成任务"
    >
      <h3 className="text-sm font-medium">生成任务</h3>
      {generations.map((generation, index) => (
        <div key={generation.id} className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-sm">
              {generation.kind === "IMAGE" ? "图片" : "视频"} {index + 1} ·{" "}
              {generationStateLabels[generation.state]}
            </p>
            <p className="truncate text-xs text-muted-foreground">
              {generation.modelName}
            </p>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSelected(generation.id)}
            aria-label={`查看生成任务 ${index + 1}`}
          >
            查看详情
          </Button>
        </div>
      ))}
      {selected && (
        <GenerationExecutionDialog
          id={selected}
          onClose={() => setSelected(null)}
          onChanged={onChanged}
        />
      )}
    </section>
  );
}
