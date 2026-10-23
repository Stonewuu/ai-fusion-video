import {
  AlignHorizontalJustifyStart,
  AlignVerticalJustifyStart,
  ArrowDown,
  ArrowRight,
  Columns2,
  Copy,
  Expand,
  FoldVertical,
  Group,
  Link2,
  LockKeyhole,
  Plus,
  SlidersHorizontal,
  Trash2,
  Ungroup,
  UnlockKeyhole,
  Wand2,
  type LucideIcon,
} from "lucide-react";
import type { CanvasCapability, CanvasNode, CanvasOperation } from "@/lib/canvas/types";
import type { CanvasAlignment } from "@/lib/canvas/layout-actions";
import { canvasPortLabels } from "@/lib/canvas/labels";

export interface CanvasSelectionAction {
  id: string;
  label: string;
  icon: LucideIcon;
  action?: (trigger: HTMLElement) => void;
  opensOverlay?: boolean;
  children?: CanvasSelectionAction[];
  disabled?: boolean;
  destructive?: boolean;
}
export function buildCanvasSelectionActions({
  nodes,
  edgeCount,
  editable,
  copy,
  lock,
  perform,
  remove,
  group,
  arrange,
  compare,
  annotate,
  capability,
  addAfter,
}: {
  nodes: CanvasNode[];
  edgeCount: number;
  editable: boolean;
  copy: () => void;
  lock: () => void;
  remove: () => void;
  group: () => void;
  arrange: (mode: CanvasAlignment) => void;
  compare: () => void;
  annotate: () => void;
  capability: CanvasCapability | undefined;
  addAfter: (nodeId: string, portId: string, trigger: HTMLElement) => void;
  perform: (label: string, operations: CanvasOperation[]) => void;
}): CanvasSelectionAction[] {
  if (!nodes.length && !edgeCount) return [];
  const result: CanvasSelectionAction[] = [];
  if (nodes.length === 1 && capability?.outputs.length) {
    const node = nodes[0];
    const outputs = capability.outputs.map((port) => ({
      id: `add-after:${port.id}`,
      label: canvasPortLabels[port.id] ?? port.label,
      icon: ArrowRight,
      opensOverlay: true,
      action: (trigger: HTMLElement) => addAfter(node.id, port.id, trigger),
    }));
    result.push({
      id: "add-after",
      label: "添加后续",
      icon: Plus,
      disabled: !editable,
      opensOverlay: true,
      ...(outputs.length === 1
        ? { action: outputs[0].action }
        : { children: outputs }),
    });
  }
  if (nodes.length >= 2 && nodes.length <= 4)
    result.push({
      id: "compare",
      label: "并排对比",
      icon: Columns2,
      action: compare,
    });
  if (nodes.length === 2)
    result.push({
      id: "annotate",
      label: "添加说明连线",
      icon: Link2,
      action: annotate,
      disabled: !editable,
    });
  if (nodes.length) {
    result.push({
      id: "copy",
      label: "复制",
      icon: Copy,
      action: copy,
      disabled: !editable,
    });
    result.push({
      id: "group",
      label: "分组",
      icon: Group,
      action: group,
      disabled: !editable,
    });
    result.push({
      id: "arrange",
      label: "排列",
      icon: SlidersHorizontal,
      disabled: !editable,
      children: [
        {
          id: "auto",
          label: "自动整理",
          icon: Wand2,
          action: () => arrange("arrange"),
        },
        {
          id: "left",
          label: "左对齐",
          icon: AlignHorizontalJustifyStart,
          action: () => arrange("left"),
        },
        {
          id: "top",
          label: "顶部对齐",
          icon: AlignVerticalJustifyStart,
          action: () => arrange("top"),
        },
        {
          id: "horizontal",
          label: "横向排列",
          icon: ArrowRight,
          action: () => arrange("horizontal"),
        },
        {
          id: "vertical",
          label: "纵向排列",
          icon: ArrowDown,
          action: () => arrange("vertical"),
        },
        {
          id: "lock",
          label: nodes.every((node) => node.layout.locked)
            ? "解锁位置"
            : "锁定位置",
          icon: nodes.every((node) => node.layout.locked)
            ? UnlockKeyhole
            : LockKeyhole,
          action: lock,
        },
      ],
    });
    if (nodes.length === 1 && nodes[0].kind === "group") {
      const node = nodes[0];
      result.push({
        id: "collapse",
        label: node.layout.collapsed ? "展开分组" : "折叠分组",
        icon: node.layout.collapsed ? Expand : FoldVertical,
        disabled: !editable,
        action: () =>
          perform("切换分组显示", [
            {
              type: "canvas.node.layout",
              nodeId: node.id,
              expectedLayoutVersion: node.layoutVersion,
              layout: { ...node.layout, collapsed: !node.layout.collapsed },
            },
          ]),
      });
      result.push({
        id: "ungroup",
        label: "解散分组",
        icon: Ungroup,
        action: remove,
        disabled: !editable,
      });
    }
  }
  result.push({
    id: "remove",
    label: nodes.length ? "移除" : "移除连线",
    icon: Trash2,
    action: remove,
    disabled: !editable,
    destructive: true,
  });
  return result;
}
