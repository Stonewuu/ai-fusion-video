"use client";

import { Plus, Undo2, Redo2, MousePointer2, Hand } from "lucide-react";
import { useCanvas } from "../canvas-provider";
import type { CanvasRelationAvailability, CanvasRelationScope } from "@/lib/canvas/relations";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasSegmented } from "../ui/canvas-segmented";
import { CanvasBarDivider, CanvasFloatingBar } from "../ui/canvas-surface";
import {
  CanvasToolbarAction,
  CanvasViewActions,
  CanvasViewSettings,
} from "./toolbar-view-settings";

const pointerOptions = [
  {
    value: "select",
    label: "选择",
    hint: "选择工具（V）：拖动节点或框选",
    icon: MousePointer2,
  },
  {
    value: "pan",
    label: "移动",
    hint: "移动工具（H）：拖动画布；空格可临时切换",
    icon: Hand,
  },
] as const;

export function CanvasToolbar(props: {
  minimap: boolean;
  onMinimap: () => void;
  relations: CanvasRelationScope;
  onRelations: (scope: CanvasRelationScope) => void;
  relationAvailability: CanvasRelationAvailability;
  relationLoading: boolean;
  relationError: string | null;
  onReloadRelations: () => void;
  onAdd: () => void;
  onFit: () => void;
  onArrange: () => void;
  arrangeError: string | null;
  arrangeUnavailable: string | null;
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
  const canArrange = editable && !props.arrangeUnavailable;
  const arrangeTitle = props.arrangeUnavailable ?? props.arrangeError ?? "整理选中的节点，未选择时整理全部";
  return (
    <div
      className="@container pointer-events-none w-full"
      role="group"
      aria-label="画布工具栏"
    >
      <div className="flex items-start justify-between gap-2">
        <CanvasFloatingBar className="min-w-0 flex-wrap">
          <CanvasToolbarAction
            icon={Plus}
            label="添加节点"
            shortcut="N"
            onClick={props.onAdd}
            disabled={!editable}
            primary
          />
          <CanvasBarDivider />
          <CanvasSegmented
            label="鼠标工具"
            value={props.mode}
            options={pointerOptions}
            onChange={props.onMode}
            iconOnly
          />
          <CanvasBarDivider />
          <CanvasIconButton
            label="撤销"
            shortcut="Ctrl/⌘ Z"
            disabled={!editable || !canUndo}
            onClick={() => void undo().catch(() => undefined)}
          >
            <Undo2 />
          </CanvasIconButton>
          <CanvasIconButton
            label="重做"
            shortcut="Ctrl/⌘ Shift Z"
            disabled={!editable || !canRedo}
            onClick={() => void redo().catch(() => undefined)}
          >
            <Redo2 />
          </CanvasIconButton>
          <CanvasBarDivider />
          <CanvasViewActions
            {...props}
            editable={!!canArrange}
            arrangeTitle={arrangeTitle}
          />
        </CanvasFloatingBar>
        <CanvasViewSettings
          {...props}
          editable={!!canArrange}
          arrangeTitle={arrangeTitle}
        />
      </div>
    </div>
  );
}
