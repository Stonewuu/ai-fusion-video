import type { CanvasGraph } from "../types";

export interface Point {
  x: number;
  y: number;
}

export interface Box extends Point {
  width: number;
  height: number;
}

export interface CanvasNodeGeometry {
  width: number;
  height: number;
  inputs: Record<string, number>;
  outputs: Record<string, number>;
}

export type CanvasGeometry = Record<string, CanvasNodeGeometry>;

export interface LayoutRelation {
  id: string;
  source: string;
  target: string;
  category: "structure" | "assets";
  label: string;
}

export interface LayoutLink {
  id: string;
  source: string;
  target: string;
  sourcePort: string;
  targetPort: string;
  kind: "execution" | "structure" | "reference";
  label: string;
}

export interface LayoutRequest {
  graph: CanvasGraph;
  geometry: CanvasGeometry;
  selection: string[];
  relations: LayoutRelation[];
  hidden: string[];
  labels?: Record<string, string>;
}

export interface LayoutUnit extends Box {
  id: string;
  movable: boolean;
  inputs: Record<string, number>;
  outputs: Record<string, number>;
}

export interface LayoutScope {
  units: LayoutUnit[];
  links: LayoutLink[];
  obstacles: Box[];
  bounds: Box | null;
}

export interface EdgeRoute {
  points: Point[];
  label: Point | null;
}

export interface LayoutResult {
  positions: Record<string, Point>;
  routes: Record<string, EdgeRoute>;
}

export const NODE_GAP = 48;
export const LAYER_GAP = 96;
export const ROUTE_MARGIN = 12;

export class LayoutConstraintError extends Error {}

export function overlaps(a: Box, b: Box, gap = 0) {
  return a.x < b.x + b.width + gap && a.x + a.width + gap > b.x
    && a.y < b.y + b.height + gap && a.y + a.height + gap > b.y;
}

export function boundsOf(boxes: Box[]): Box {
  const x = Math.min(...boxes.map((box) => box.x));
  const y = Math.min(...boxes.map((box) => box.y));
  return {
    x, y,
    width: Math.max(...boxes.map((box) => box.x + box.width)) - x,
    height: Math.max(...boxes.map((box) => box.y + box.height)) - y,
  };
}
