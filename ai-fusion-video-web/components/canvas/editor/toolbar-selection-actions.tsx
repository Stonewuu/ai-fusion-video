"use client";

import { Fragment } from "react";
import { useStore as useFlowStore, ViewportPortal } from "@xyflow/react";
import { shallow } from "zustand/shallow";
import type { CanvasSelectionAction } from "./build-context-actions";
import { CanvasSelectionActionButton } from "./selection-action-button";
import { CanvasBarDivider } from "../ui/canvas-surface";

/**
 * 选中工具条渲染在画布视口层内，吸附在选区正上方并水平居中：平移时与节点同步移动，
 * 按缩放倍数反向缩放保持原始尺寸；超出画布的部分由画布容器裁切。
 */
export function CanvasSelectionActions({
  nodeIds,
  count,
  edgeCount,
  actions,
}: {
  nodeIds: string[];
  count: number;
  edgeCount: number;
  actions: CanvasSelectionAction[];
}) {
  const bounds = useFlowStore((state) => {
    const nodes = nodeIds
      .map((id) => state.nodeLookup.get(id))
      .filter((node) => node && !node.hidden);
    if (!nodes.length) {
      return null;
    }
    const left = Math.min(
      ...nodes.map((node) => node!.internals.positionAbsolute.x),
    );
    const top = Math.min(
      ...nodes.map((node) => node!.internals.positionAbsolute.y),
    );
    const right = Math.max(
      ...nodes.map(
        (node) =>
          node!.internals.positionAbsolute.x + (node!.measured.width ?? 0),
      ),
    );
    return {
      center: (left + right) / 2,
      top,
      zoom: state.transform[2],
      width: state.width,
    };
  }, shallow);
  if (!bounds || !actions.length) {
    return null;
  }
  return (
    <ViewportPortal>
      <div
        role="toolbar"
        aria-label="所选节点操作"
        data-canvas-selection-toolbar
        className="nodrag nopan nowheel pointer-events-auto absolute flex w-max origin-bottom items-center gap-1 overflow-x-auto rounded-[14px] border border-foreground/[0.08] bg-popover/95 p-1 shadow-[0_2px_6px_rgba(0,0,0,0.05),0_12px_28px_-6px_rgba(0,0,0,0.12)] backdrop-blur-xl dark:border-white/[0.11] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_12px_30px_-6px_rgba(0,0,0,0.52)]"
        style={{
          left: bounds.center,
          top: bounds.top,
          maxWidth: Math.max(0, bounds.width - 16),
          // 选中节点会被提升到 z-index 1000，工具条需位于其上。
          zIndex: 2000,
          transform: `translate(-50%, calc(-100% - ${12 / bounds.zoom}px)) scale(${1 / bounds.zoom})`,
        }}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <span className="inline-flex h-8 shrink-0 items-center rounded-[10px] bg-foreground/[0.055] px-2.5 text-xs font-medium text-foreground/70 dark:bg-white/[0.07]">
          {count ? "已选 " + count + " 项" : ""}
          {count && edgeCount ? " · " : ""}
          {edgeCount ? edgeCount + " 条连线" : ""}
        </span>
        <CanvasBarDivider />
        {actions.map((action, index) => (
          <Fragment key={action.id}>
            {action.destructive && index > 0 && <CanvasBarDivider />}
            <CanvasSelectionActionButton action={action} />
          </Fragment>
        ))}
      </div>
    </ViewportPortal>
  );
}
