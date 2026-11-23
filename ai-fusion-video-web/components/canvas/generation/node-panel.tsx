"use client";

import {
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import {
  useReactFlow,
  useStore as useFlowStore,
  ViewportPortal,
} from "@xyflow/react";
import { shallow } from "zustand/shallow";
import { useReducedMotion } from "framer-motion";
import type { CanvasNode } from "@/lib/canvas/types";

/** 节点上方依次是画布工具栏（56px）和选中工具条（56px），打开面板时为它们留位。 */
const RESERVED_TOP = 112;
/** 画布底边安全留白（避开阴影截断与底部居中水印）。 */
const RESERVED_BOTTOM = 32;

/**
 * 节点快捷面板外壳（生成表单、文本面板）：渲染在画布视口层内，与节点处于同一变换层，
 * 平移时无需重新渲染即可同步移动；按缩放倍数反向缩放保持屏幕原始尺寸，超出画布的部分由画布容器裁切。
 */
export function CanvasNodePanel({
  node,
  area,
  onClose,
  children,
}: {
  node: CanvasNode;
  area: RefObject<HTMLDivElement | null>;
  onClose: () => void;
  children: (maxHeight: number) => ReactNode;
}) {
  const flow = useReactFlow();
  const reduced = useReducedMotion();
  const opened = useRef("");
  const panel = useRef<HTMLDivElement>(null);
  const [panelHeight, setPanelHeight] = useState(0);
  const [size, setSize] = useState<{ width: number; height: number } | null>(
    null,
  );
  const bounds = useFlowStore((state) => {
    const internal = state.nodeLookup.get(node.id);
    if (
      !internal ||
      internal.hidden ||
      !internal.measured.width ||
      !internal.measured.height
    ) {
      return null;
    }
    return {
      x: internal.internals.positionAbsolute.x,
      y: internal.internals.positionAbsolute.y,
      width: internal.measured.width,
      height: internal.measured.height,
      tx: state.transform[0],
      ty: state.transform[1],
      zoom: state.transform[2],
    };
  }, shallow);
  useLayoutEffect(() => {
    const element = area.current!;
    const measure = () => {
      const next = { width: element.clientWidth, height: element.clientHeight };
      if (next.width === 0 || next.height === 0) {
        setSize(null);
        return;
      }
      setSize((current) =>
        current &&
          current.width === next.width &&
          current.height === next.height
          ? current
          : next,
      );
    };
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    measure();
    return () => {
      observer.disconnect();
    };
  }, [area]);
  const positioned = !!bounds && !!size;
  useLayoutEffect(() => {
    if (!positioned || !panel.current) {
      return;
    }
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        if (panel.current) {
          setPanelHeight(panel.current.offsetHeight);
        }
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(panel.current);
    measure();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [positioned, node.id]);
  useLayoutEffect(() => {
    if (!bounds || !size || !panelHeight) {
      return;
    }
    // 仅在首次打开当前节点面板或画布视口尺寸变化时居中，避免面板内展开/收起时反复跳动镜头。
    const key = `${node.id}:${Math.round(size.width)}:${Math.round(size.height)}`;
    if (opened.current === key) {
      return;
    }
    opened.current = key;
    const space = Math.max(
      80,
      size.height - panelHeight - RESERVED_TOP - RESERVED_BOTTOM - 12,
    );
    const zoom = Math.min(
      1,
      space / bounds.height,
      (size.width - 32) / bounds.width,
    );
    const maxTop = Math.max(
      RESERVED_TOP,
      size.height - bounds.height * zoom - panelHeight - RESERVED_BOTTOM - 12,
    );
    const top = Math.max(
      RESERVED_TOP,
      Math.min(
        maxTop,
        (size.height - bounds.height * zoom - panelHeight - 12) / 2,
      ),
    );
    // 推迟一帧定位，避免与新建节点触发的 focus 动画竞态覆盖面板留位。
    const frame = requestAnimationFrame(() => {
      void flow.setViewport(
        {
          x: size.width / 2 - (bounds.x + bounds.width / 2) * zoom,
          y: top - bounds.y * zoom,
          zoom,
        },
        { duration: reduced ? 0 : 200 },
      );
    });
    return () => {
      cancelAnimationFrame(frame);
    };
  }, [node.id, bounds, size, panelHeight, flow, reduced]);
  if (!bounds || !size) {
    return null;
  }
  const width = Math.min(560, size.width - 24);
  const idealMaxHeight = Math.max(
    220,
    Math.min(480, size.height - RESERVED_TOP - 140),
  );
  const panelTop = (bounds.y + bounds.height) * bounds.zoom + bounds.ty + 12;
  const availableBelow = Math.floor(size.height - panelTop - RESERVED_BOTTOM);
  const maxHeight =
    availableBelow >= 220
      ? Math.min(idealMaxHeight, availableBelow)
      : idealMaxHeight;
  return (
    <ViewportPortal>
      <div
        ref={panel}
        className="nodrag nopan nowheel pointer-events-auto absolute origin-top"
        style={{
          left: bounds.x + bounds.width / 2,
          top: bounds.y + bounds.height,
          width,
          zIndex: 2000,
          transform: `translate(-50%, ${12 / bounds.zoom}px) scale(${1 / bounds.zoom})`,
        }}
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => event.stopPropagation()}
        onDoubleClick={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
        onCopy={(event) => event.stopPropagation()}
        onPaste={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          event.stopPropagation();
          if (
            event.key === "Escape" &&
            !event.defaultPrevented &&
            !event.nativeEvent.isComposing &&
            event.target instanceof Node &&
            panel.current?.contains(event.target)
          ) {
            onClose();
          }
        }}
      >
        <div className="animate-in fade-in zoom-in-95 duration-200 ease-out motion-reduce:animate-none">
          {children(maxHeight)}
        </div>
      </div>
    </ViewportPortal>
  );
}
