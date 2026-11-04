"use client";

import { memo } from "react";
import { BezierEdge, type Edge, type EdgeProps } from "@xyflow/react";

export type CanvasDisplayEdge = Edge<{ warning?: string | null }>;

export const CanvasEdgeRenderer = memo(function CanvasEdgeRenderer(props: EdgeProps<CanvasDisplayEdge>) {
  return (
    <g>
      {props.data?.warning && <title>{props.data.warning}</title>}
      <BezierEdge {...props} />
    </g>
  );
});
