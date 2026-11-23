"use client";

import { Plus } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";

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
              添加文字或素材，开始生成图片和视频。
            </p>
            <CanvasButton variant="primary" className="pointer-events-auto" onClick={onAdd}>
              <Plus />
              添加第一个节点
            </CanvasButton>
          </>
        )}
      </div>
    </div>
  );
}
