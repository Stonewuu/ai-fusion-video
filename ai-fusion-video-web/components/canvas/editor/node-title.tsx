"use client";

import { useState } from "react";
import { toast } from "sonner";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasNode } from "@/lib/canvas/types";
import { useCanvasStoreApi } from "../canvas-provider";

/** 节点标题：双击进入改名，回车或失焦保存，Esc 取消。 */
export function CanvasNodeTitle({
  node,
  canEdit,
}: {
  node: CanvasNode;
  canEdit: boolean;
}) {
  const store = useCanvasStoreApi();
  const [draft, setDraft] = useState<string | null>(null);
  const commit = async () => {
    if (draft === null) {
      return;
    }
    const next = draft.trim();
    setDraft(null);
    if (!next || next === node.title) {
      return;
    }
    try {
      await store.getState().edit("重命名节点", [
        {
          type: "canvas.node.patch",
          nodeId: node.id,
          expectedConfigVersion: node.configVersion,
          configPatch: {},
          title: next,
        },
      ]);
    } catch (failure) {
      toast.error(getApiErrorMessage(failure));
    }
  };
  if (draft !== null) {
    return (
      <input
        autoFocus
        aria-label="节点名称"
        value={draft}
        maxLength={200}
        className="nodrag nopan h-6 min-w-0 flex-1 rounded-sm border border-primary bg-background px-1 text-xs font-medium text-foreground outline-none ring-2 ring-primary/20"
        onFocus={(event) => event.currentTarget.select()}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => void commit()}
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          event.stopPropagation();
          if (event.key === "Enter") {
            event.preventDefault();
            void commit();
          } else if (event.key === "Escape") {
            event.preventDefault();
            setDraft(null);
          }
        }}
      />
    );
  }
  return (
    <p
      className="min-w-0 truncate border border-transparent px-1 font-medium text-foreground/80"
      title={canEdit ? `${node.title}（双击改名）` : node.title}
      onDoubleClick={(event) => {
        if (!canEdit) {
          return;
        }
        event.stopPropagation();
        setDraft(node.title);
      }}
    >
      {node.title}
    </p>
  );
}
