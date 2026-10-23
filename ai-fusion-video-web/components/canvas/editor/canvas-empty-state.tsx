"use client";

import { Cable, FileText, Plus, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export function CanvasEmptyState({
  editable,
  onAdd,
}: {
  editable: boolean;
  onAdd: () => void;
}) {
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-6">
      <div className="flex max-w-md flex-col items-center gap-4 text-center">
        <h2 className="text-lg font-medium">
          {editable ? "从一个想法开始" : "此画布暂无内容"}
        </h2>
        {editable && (
          <>
            <p className="text-sm leading-relaxed text-muted-foreground">
              先写下想法，或从左侧拖入剧本、分镜和素材。
            </p>
            <ol className="flex flex-wrap justify-center gap-4 text-xs" aria-label="创作流程">
              <li className="flex items-center gap-1.5 text-sky-700 dark:text-sky-300"><FileText className="size-4" aria-hidden /><span><strong>1</strong> 添加内容</span></li>
              <li className="flex items-center gap-1.5 text-violet-600 dark:text-violet-300"><Cable className="size-4" aria-hidden /><span><strong>2</strong> 连接创作步骤</span></li>
              <li className="flex items-center gap-1.5 text-emerald-700 dark:text-emerald-300"><Sparkles className="size-4" aria-hidden /><span><strong>3</strong> 运行并挑选结果</span></li>
            </ol>
            <Button className="pointer-events-auto" onClick={onAdd}>
              <Plus />
              添加第一个节点
            </Button>
            <p className="text-xs text-muted-foreground">
              右键空白处，也能随时添加。
            </p>
          </>
        )}
      </div>
    </div>
  );
}
