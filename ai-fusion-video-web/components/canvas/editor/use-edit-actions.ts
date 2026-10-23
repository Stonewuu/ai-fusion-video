"use client";
import type { KeyboardEventHandler, RefObject } from "react";
import { useReactFlow } from "@xyflow/react";
import type { CanvasGeometry } from "@/lib/canvas/auto-layout";
import { toast } from "sonner";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { copyNodes } from "@/lib/canvas/graph";
import {
  arrangeCanvasNodes,
  deleteCanvasSelection,
  groupCanvasNodes,
  isEditableTarget,
  type CanvasAlignment,
} from "@/lib/canvas/layout-actions";
import type { CanvasStore } from "@/lib/store/canvas-store";
import type { CanvasOperation, CanvasSnapshot } from "@/lib/canvas/types";
export function useCanvasEditActions({
  store,
  snapshot,
  editable,
  hidden,
  region,
  perform,
  onEscape,
  onAdd,
  onInspect,
  onMode,
  onSearch,
  onFit,
  onHelp,
}: {
  store: CanvasStore;
  snapshot: CanvasSnapshot;
  editable: boolean;
  hidden: Set<string>;
  region: RefObject<HTMLDivElement | null>;
  perform: (label: string, operations: CanvasOperation[]) => void;
  onEscape: () => void;
  onAdd: () => void;
  onInspect: (id: string) => void;
  onMode: (mode: "select" | "pan") => void;
  onSearch: () => void;
  onFit: () => void;
  onHelp: () => void;
}) {
  const flow = useReactFlow();
  const selectedNodes = () =>
    store
      .getState()
      .snapshot!.graph.nodes.filter((node) =>
        store.getState().selection.includes(node.id),
      );
  const remove = () =>
    perform(
      "移除画布视图",
      deleteCanvasSelection(
        store.getState().snapshot!.graph,
        store.getState().selection,
        store.getState().edgeSelection,
      ),
    );
  const copy = () =>
    perform(
      "复制节点引用",
      copyNodes(store.getState().snapshot!.graph, store.getState().selection),
    );
  const group = () => {
    try {
      perform(
        "分组",
        groupCanvasNodes(
          store.getState().snapshot!.graph,
          store.getState().selection,
        ),
      );
    } catch (error) {
      toast.error(getApiErrorMessage(error));
    }
  };
  const arrange = (mode: CanvasAlignment) => {
    try {
      const geometry: CanvasGeometry = {};
      for (const node of flow.getNodes()) {
        const internal = flow.getInternalNode(node.id);
        if (!node.measured?.width || !node.measured.height) continue;
        const handleMap = (
          handles:
            | { id?: string | null; y: number; height: number }[]
            | null
            | undefined,
        ) =>
          Object.fromEntries(
            (handles ?? [])
              .filter((handle) => !!handle.id)
              .map((handle) => [handle.id!, handle.y + handle.height / 2]),
          );
        geometry[node.id] = {
          width: node.measured.width,
          height: node.measured.height,
          inputs: handleMap(internal?.internals.handleBounds?.target),
          outputs: handleMap(internal?.internals.handleBounds?.source),
        };
      }
      perform(
        "排列节点",
        arrangeCanvasNodes(
          store.getState().snapshot!.graph,
          store.getState().selection,
          mode,
          geometry,
        ),
      );
    } catch (error) {
      toast.error(getApiErrorMessage(error));
    }
  };
  const lock = () => {
    const list = selectedNodes();
    const locked = !list.every((node) => node.layout.locked);
    perform(
      locked ? "锁定位置" : "解锁位置",
      list.map((node) => ({
        type: "canvas.node.layout",
        nodeId: node.id,
        expectedLayoutVersion: node.layoutVersion,
        layout: { ...node.layout, locked },
      })),
    );
  };
  const onKeyDown: KeyboardEventHandler<HTMLDivElement> = (event) => {
    if (
      event.defaultPrevented ||
      event.nativeEvent.isComposing ||
      !(event.target instanceof Element) ||
      !region.current?.contains(event.target) ||
      isEditableTarget(event.target) ||
      event.target.closest("button,a,[role='menu'],[role='listbox'],summary")
    )
      return;
    const modifier = event.ctrlKey || event.metaKey;
    if (!modifier && !event.altKey) {
      const key = event.key.toLowerCase();
      const shortcut =
        key === "v"
          ? () => onMode("select")
          : key === "h"
            ? () => onMode("pan")
            : key === "n" && editable
              ? onAdd
              : key === "/"
                ? onSearch
                : key === "f"
                  ? onFit
                  : key === "?"
                    ? onHelp
                    : key === "escape"
                      ? onEscape
                      : null;
      if (shortcut) {
        event.preventDefault();
        event.stopPropagation();
        shortcut();
        return;
      }
      if (event.key === "Enter") {
        const id =
          event.target.closest<HTMLElement>(".react-flow__node")?.dataset.id;
        if (id) {
          event.preventDefault();
          event.stopPropagation();
          onInspect(id);
          return;
        }
      }
    }
    if (modifier && event.key.toLowerCase() === "a") {
      event.preventDefault();
      store
        .getState()
        .select(
          snapshot.graph.nodes
            .filter((node) => !hidden.has(node.id))
            .map((node) => node.id),
        );
      return;
    }
    if (!editable) return;
    if (
      event.key.startsWith("Arrow") &&
      (event.target === region.current ||
        (event.target instanceof Element &&
          !!event.target.closest(".react-flow__node")))
    ) {
      event.preventDefault();
      event.stopPropagation();
      const distance = event.shiftKey ? 32 : 4;
      const x =
        event.key === "ArrowLeft"
          ? -distance
          : event.key === "ArrowRight"
            ? distance
            : 0;
      const y =
        event.key === "ArrowUp"
          ? -distance
          : event.key === "ArrowDown"
            ? distance
            : 0;
      perform(
        "键盘移动节点",
        selectedNodes()
          .filter((node) => !node.layout.locked)
          .map((node) => ({
            type: "canvas.node.layout",
            nodeId: node.id,
            expectedLayoutVersion: node.layoutVersion,
            layout: {
              ...node.layout,
              x: node.layout.x + x,
              y: node.layout.y + y,
            },
          })),
      );
    } else if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      remove();
    } else if (modifier && event.key.toLowerCase() === "z") {
      event.preventDefault();
      void (
        event.shiftKey ? store.getState().redo() : store.getState().undo()
      ).catch((error) => toast.error(getApiErrorMessage(error)));
    } else if (modifier && event.key.toLowerCase() === "d") {
      event.preventDefault();
      copy();
    } else if (modifier && event.key.toLowerCase() === "g") {
      event.preventDefault();
      group();
    }
  };
  return { remove, copy, group, arrange, lock, onKeyDown };
}
