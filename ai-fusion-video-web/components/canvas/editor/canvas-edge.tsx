"use client";

import { memo } from "react";
import { BaseEdge, BezierEdge, type Edge, type EdgeProps } from "@xyflow/react";
import { roundedRoutePath, routeMatchesEndpoints } from "@/lib/canvas/layout/edge-path";
import type { EdgeRoute } from "@/lib/canvas/layout/types";

export type CanvasDisplayEdge = Edge<{ warning?: string | null; route?: EdgeRoute }>;

export const CanvasEdgeRenderer = memo(function CanvasEdgeRenderer(props: EdgeProps<CanvasDisplayEdge>) {
  const route = props.data?.route;
  const settled = route && routeMatchesEndpoints(route,
    { x: props.sourceX, y: props.sourceY }, { x: props.targetX, y: props.targetY });
  const middle = route?.label ?? route?.points[Math.floor(route.points.length / 2)];
  return (
    <g>
      {props.data?.warning && <title>{props.data.warning}</title>}
      {settled ? <BaseEdge
        id={props.id}
        path={roundedRoutePath(route.points)}
        style={props.style}
        markerStart={props.markerStart}
        markerEnd={props.markerEnd}
        interactionWidth={props.interactionWidth}
        label={props.label}
        labelX={middle!.x}
        labelY={middle!.y}
        labelStyle={props.labelStyle}
        labelShowBg={props.labelShowBg}
        labelBgStyle={props.labelBgStyle}
        labelBgPadding={props.labelBgPadding}
        labelBgBorderRadius={props.labelBgBorderRadius}
      /> : <BezierEdge {...props} />}
    </g>
  );
});
