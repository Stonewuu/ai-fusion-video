"use client";

import { Plus, Undo2, Redo2, MousePointer2, Hand } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCanvas } from "../canvas-provider";
import type { CanvasAlignment } from "@/lib/canvas/layout-actions";
import type { CanvasRelationScope } from "@/lib/canvas/relations";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import {
  CanvasToolbarAction,
  CanvasViewActions,
  CanvasViewSettings,
} from "./toolbar-view-settings";

const pointerOptions = [
  {
    value: "select",
    label: "选择",
    title: "选择工具（V）：拖动节点或框选",
    icon: MousePointer2,
  },
  {
    value: "pan",
    label: "移动",
    title: "移动工具（H）：拖动画布；空格可临时切换",
    icon: Hand,
  },
] as const;

export function CanvasToolbar(props: {
  minimap: boolean;
  onMinimap: () => void;
  relations: CanvasRelationScope;
  onRelations: (scope: CanvasRelationScope) => void;
  onAdd: () => void;
  onFit: () => void;
  onArrange: (mode: CanvasAlignment) => void;
  onSearch: () => void;
  mode: "select" | "pan";
  onMode: (value: "select" | "pan") => void;
}) {
  const editable = useCanvas(
    (state) =>
      state.snapshot!.permissions.canEdit &&
      !state.snapshot!.document.archived &&
      !state.pending,
  );
  const canUndo = useCanvas((state) => state.undoStack.length > 0);
  const canRedo = useCanvas((state) => state.redoStack.length > 0);
  const undo = useCanvas((state) => state.undo);
  const redo = useCanvas((state) => state.redo);
  return (
    <div
      className="@container pointer-events-none w-full"
      role="group"
      aria-label="画布工具栏"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="pointer-events-auto flex min-w-0 flex-wrap items-center gap-2 rounded-2xl border border-border/40 bg-popover/90 p-1 shadow-sm backdrop-blur-xl">
        <CanvasToolbarAction
          icon={Plus}
          label="添加节点"
          title="添加节点（N）"
          onClick={props.onAdd}
          disabled={!editable}
          primary
        />
        <CanvasSegmentedControl
          label="鼠标工具"
          value={props.mode}
          options={pointerOptions}
          onChange={props.onMode}
          compact
        />
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="icon"
            title="撤销画布编辑（Ctrl/⌘ Z）"
            disabled={!editable || !canUndo}
            onClick={() => void undo().catch(() => undefined)}
          >
            <Undo2 />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            title="重做画布编辑（Ctrl/⌘ Shift Z）"
            disabled={!editable || !canRedo}
            onClick={() => void redo().catch(() => undefined)}
          >
            <Redo2 />
          </Button>
        </div>
        <CanvasViewActions
          {...props}
          editable={!!editable}
          onArrange={() => props.onArrange("arrange")}
        />
        </div>
        <CanvasViewSettings
          {...props}
          editable={!!editable}
          onArrange={() => props.onArrange("arrange")}
        />
      </div>
    </div>
  );
}
