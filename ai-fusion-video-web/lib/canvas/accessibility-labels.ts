import type { AriaLabelConfig } from "@xyflow/react";

export const canvasAriaLabels: Partial<AriaLabelConfig> = {
  "node.a11yDescription.default": "按回车或空格选中节点，用方向键移动。按 Delete 移除节点，按 Escape 取消选择。",
  "node.a11yDescription.keyboardDisabled": "按回车或空格选中节点。",
  "node.a11yDescription.ariaLiveMessage": ({ x, y }) => `节点已移动到横向 ${Math.round(x)}、纵向 ${Math.round(y)}。`,
  "edge.a11yDescription.default": "按回车或空格选中连线，按 Delete 移除连线，按 Escape 取消选择。",
  "controls.ariaLabel": "缩放控制",
  "controls.zoomIn.ariaLabel": "放大画布",
  "controls.zoomOut.ariaLabel": "缩小画布",
  "controls.fitView.ariaLabel": "查看全部节点",
  "controls.interactive.ariaLabel": "切换画布编辑状态",
  "minimap.ariaLabel": "画布小地图",
  "handle.ariaLabel": "节点连接端口",
};
