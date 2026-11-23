"use client";

import { useState } from "react";
import { toast } from "sonner";
import { FileCheck2, FileInput } from "lucide-react";
import { CanvasButton } from "../ui/canvas-button";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasNodeActivity } from "@/lib/api/canvas-runs";
import { textGenerationWriteback } from "@/lib/canvas/generation";
import type { CanvasNode } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";

/** 生成文本完成后，把结果写回提供写作素材的文本节点正文。 */
export function TextGenerationWriteback({
  node,
  activity,
  disabled,
}: {
  node: CanvasNode;
  activity: CanvasNodeActivity | undefined;
  disabled: boolean;
}) {
  const store = useCanvasStoreApi();
  const graph = useCanvas((state) => state.snapshot!.graph);
  const catalogNodes = useCanvas((state) => state.catalog!.nodes);
  const [saving, setSaving] = useState(false);
  const writeback = textGenerationWriteback(graph, node.id, activity);
  if (!writeback) {
    return null;
  }
  const { target, text } = writeback;
  const limit = catalogNodes.find((item) => item.kind === target.kind)
    ?.configFields.content?.maxLength;
  const length = [...text].length;
  const tooLong = !!limit && length > limit;
  const write = async () => {
    setSaving(true);
    try {
      await store.getState().edit("写入生成文本", [
        {
          type: "canvas.node.patch",
          nodeId: target.id,
          expectedConfigVersion: target.configVersion,
          configPatch: { content: text },
        },
      ]);
      toast.success(`已写入“${target.title}”`);
    } catch (failure) {
      toast.error(getApiErrorMessage(failure));
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-[12px] border border-primary/20 bg-primary/[0.05] py-1.5 pl-3 pr-1.5">
      <FileCheck2 className="size-3.5 shrink-0 text-primary" aria-hidden />
      <p className="min-w-0 flex-1 truncate text-xs font-medium text-foreground/80">
        {tooLong
          ? `结果超过 ${limit!.toLocaleString()} 字，请缩短后再写入`
          : `已生成 ${length.toLocaleString()} 字`}
      </p>
      <CanvasButton
        variant="primary"
        size="sm"
        className="min-w-0"
        title={`用生成结果替换“${target.title}”的正文`}
        disabled={disabled || saving || tooLong}
        onClick={() => void write()}
      >
        <FileInput />
        <span className="max-w-40 truncate">写入“{target.title}”</span>
      </CanvasButton>
    </div>
  );
}
