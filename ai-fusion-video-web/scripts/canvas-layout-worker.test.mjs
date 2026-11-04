import test from "node:test";
import assert from "node:assert/strict";
import { loadBrowserWorkerModule } from "./helpers/browser-worker-runtime.mjs";

const { runLayoutWorker } = loadBrowserWorkerModule("lib/canvas/layout/worker-client.ts");

function request() {
  const nodes = ["a", "b"].map((id) => ({
    id, kind: "text", title: id, schemaVersion: 1, config: {}, resource: null, configVersion: 0, layoutVersion: 1,
    layout: { x: 0, y: 0, width: null, height: null, parentId: null, locked: false, collapsed: false },
  }));
  const edge = {
    id: "a-b", kind: "execution", sourceNodeId: "a", targetNodeId: "b",
    sourcePort: "out", targetPort: "in", inputIndex: 0, output: null, version: 0,
  };
  const geometry = Object.fromEntries(nodes.map((node) => [node.id, {
    width: 300, height: 200,
    inputs: { in: { x: -4, y: 100 }, "annotation-in": { x: -4, y: 100 } },
    outputs: { out: { x: 304, y: 100 }, "annotation-out": { x: 304, y: 100 } },
  }]));
  return { graph: { schemaVersion: 1, projectId: null, nodes, edges: [edge] }, geometry, selection: [], relations: [], hidden: [] };
}

test("布局 Worker 在 self 环境下完成分层布局，ELK 消息不覆盖应用消息", async () => {
  const input = request();
  const result = await runLayoutWorker(input, new AbortController().signal);
  assert.deepEqual(Object.keys(result.positions).sort(), ["a", "b"]);
  assert.ok(result.positions.a.x + 300 <= result.positions.b.x);
  const placed = {
    ...input,
    graph: { ...input.graph, nodes: input.graph.nodes.map((node) => ({
      ...node, layout: { ...node.layout, ...result.positions[node.id] },
    })) },
  };
  const repeated = await runLayoutWorker(placed, new AbortController().signal);
  assert.deepEqual(repeated.positions, result.positions);
});

test("布局 Worker 将约束错误返回调用方，不伪造成功结果", async () => {
  const input = request();
  input.graph.edges.push({
    ...input.graph.edges[0], id: "b-a", sourceNodeId: "b", targetNodeId: "a",
  });
  await assert.rejects(runLayoutWorker(input, new AbortController().signal), /执行连线形成循环/);
});
