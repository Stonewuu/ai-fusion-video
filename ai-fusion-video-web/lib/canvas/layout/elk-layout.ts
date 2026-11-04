import type { ELK, ElkNode } from "elkjs/lib/elk-api.js";
import { LAYER_GAP, NODE_GAP, type LayoutScope, type Point } from "./types";

const portId = (id: string, output: boolean, port: string) => JSON.stringify([id, output, port]);

export async function layeredPlacement(scope: LayoutScope, engine: Pick<ELK, "layout">) {
  const graph: ElkNode = {
    id: "canvas-layout",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      // 最终路径由避让器计算，ELK 只安排节点，不重复执行折线路由和标签排版。
      "elk.edgeRouting": "POLYLINE",
      "elk.randomSeed": "1",
      "elk.padding": "[top=0,left=0,bottom=0,right=0]",
      "elk.spacing.nodeNode": String(NODE_GAP),
      "elk.spacing.componentComponent": "80",
      "elk.spacing.edgeNode": "20",
      "elk.spacing.edgeEdge": "12",
      "elk.layered.spacing.nodeNodeBetweenLayers": String(LAYER_GAP),
      "elk.layered.spacing.edgeNodeBetweenLayers": "24",
      "elk.layered.layering.strategy": "NETWORK_SIMPLEX",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.crossingMinimization.greedySwitch.type": "OFF",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.layered.thoroughness": "3",
      "elk.separateConnectedComponents": "true",
      "elk.aspectRatio": "1.6",
    },
    children: scope.units.map((node) => ({
      id: node.id,
      width: node.width,
      height: node.height,
      layoutOptions: { "elk.portConstraints": "FIXED_POS" },
      ports: [false, true].flatMap((output) => Object.entries(output ? node.outputs : node.inputs)
        .map(([port, y]) => ({
          id: portId(node.id, output, port),
          x: output ? node.width : 0,
          y,
          width: 0,
          height: 0,
          layoutOptions: { "elk.port.side": output ? "EAST" : "WEST" },
        }))),
    })),
    edges: scope.links.map((link) => ({
      id: link.id,
      sources: [portId(link.source, true, link.sourcePort)],
      targets: [portId(link.target, false, link.targetPort)],
      layoutOptions: {
        "elk.layered.priority.direction": String(link.kind === "execution" ? 1000 : link.kind === "structure" ? 100 : 1),
        "elk.layered.priority.straightness": String(link.kind === "execution" ? 100 : 1),
      },
    })),
  };
  const result = await engine.layout(graph);
  const positions = new Map<string, Point>();
  for (const node of result.children!) {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) {
      throw new Error("布局引擎没有返回有效位置");
    }
    positions.set(node.id, { x: node.x!, y: node.y! });
  }
  return positions;
}
