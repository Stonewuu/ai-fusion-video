import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
const cache = new Map();
export function load(relative) {
  const file = path.resolve(root, relative);
  if (cache.has(file)) {
    return cache.get(file).exports;
  }
  const module = { exports: {} };
  cache.set(file, module);
  const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
  }).outputText;
  const localRequire = (name) => {
    if (name.startsWith("@/")) {
      return load(`${name.slice(2)}.ts`);
    }
    if (name.startsWith(".")) {
      return load(`${path.resolve(path.dirname(file), name)}.ts`);
    }
    return require(name);
  };
  vm.runInThisContext(`(function(require,module,exports){${compiled}\n})`, { filename: file })(localRequire, module, module.exports);
  return module.exports;
}
const { autoArrangeCanvas, autoLayoutOperations } = load("lib/canvas/auto-layout.ts");
const { layoutWorldBoxes } = load("lib/canvas/layout/model.ts");
const { overlaps, boundsOf } = load("lib/canvas/layout/types.ts");

function node(id, layout = {}, kind = "text") {
  return {
    id, kind, title: id, schemaVersion: 1, config: {}, resource: null, configVersion: 0, layoutVersion: 7,
    layout: { x: 0, y: 0, width: null, height: null, parentId: null, locked: false, collapsed: false, ...layout },
  };
}
function edge(source, target, kind = "execution", id = `${source}-${target}`) {
  return { id, kind, sourceNodeId: source, targetNodeId: target, sourcePort: "out", targetPort: "in", inputIndex: 0, output: null, version: 0 };
}
function request(nodes, edges = [], options = {}) {
  const geometry = Object.fromEntries(nodes.map((node) => [node.id, {
    width: node.layout.width ?? 300, height: node.layout.height ?? 200,
    inputs: { in: 100, "annotation-in": 100 }, outputs: { out: 100, "annotation-out": 100 },
  }]));
  return { graph: { schemaVersion: 1, projectId: null, nodes, edges }, selection: [], relations: [], hidden: [], geometry, ...options };
}
function apply(input, result) {
  const operations = autoLayoutOperations(input, result);
  return { ...input, graph: { ...input.graph, nodes: input.graph.nodes.map((node) => {
    const operation = operations.find((operation) => operation.nodeId === node.id);
    return operation ? { ...node, layout: operation.layout } : node;
  }) } };
}
function verify(input, result) {
  const boxes = layoutWorldBoxes(input, result.positions).boxes;
  const top = input.graph.nodes.filter((node) => node.layout.parentId === null);
  for (let a = 0; a < top.length; a++) {
    for (let b = a + 1; b < top.length; b++) {
      assert.equal(overlaps(boxes.get(top[a].id), boxes.get(top[b].id)), false, `${top[a].id} 与 ${top[b].id} 不应重叠`);
    }
  }
  for (const edge of input.graph.edges.filter((edge) => edge.kind === "execution")) {
    assert.ok(boxes.get(edge.sourceNodeId).x + boxes.get(edge.sourceNodeId).width <= boxes.get(edge.targetNodeId).x, edge.id);
  }
  for (const [id, route] of Object.entries(result.routes)) {
    for (let index = 1; index < route.points.length; index++) {
      const a = route.points[index - 1];
      const b = route.points[index];
      assert.ok(a.x === b.x || a.y === b.y, `${id} 必须是正交路径`);
      for (const node of input.graph.nodes.filter((node) => node.kind !== "group")) {
        const box = boxes.get(node.id);
        const intersects = a.x === b.x
          ? a.x > box.x + 0.01 && a.x < box.x + box.width - 0.01 && Math.min(a.y, b.y) < box.y + box.height && Math.max(a.y, b.y) > box.y
          : a.y > box.y + 0.01 && a.y < box.y + box.height - 0.01 && Math.min(a.x, b.x) < box.x + box.width && Math.max(a.x, b.x) > box.x;
        assert.equal(intersects, false, `${id} 不应穿过 ${node.id}`);
      }
    }
  }
  return boxes;
}

test("分支汇合按执行顺序排列且连线避开节点，重复整理位置不变", async () => {
  const input = request(["a", "b", "c", "d", "x", "y"].map((id) => node(id)), [edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d"), edge("x", "y")]);
  const result = await autoArrangeCanvas(input);
  verify(input, result);
  const bounds = boundsOf([...layoutWorldBoxes(input, result.positions).boxes.values()]);
  assert.ok(bounds.height < 1000, "两条链路不应逐条摊成长列");
  const second = await autoArrangeCanvas(apply(input, result));
  assert.deepEqual(second.positions, result.positions);
  assert.equal(autoLayoutOperations(apply(input, result), second).length, 0);
});

test("测量尺寸及端口参与布局，直线链路端口对齐", async () => {
  const input = request([node("a"), node("b"), node("c")], [edge("a", "b"), edge("b", "c")]);
  input.geometry.a = { width: 300, height: 160, inputs: {}, outputs: { out: 140 } };
  input.geometry.b = { width: 620, height: 400, inputs: { in: 365 }, outputs: { out: 365 } };
  input.geometry.c = { width: 300, height: 210, inputs: { in: 185 }, outputs: {} };
  input.labels = { "a-b": "检查输入" };
  const result = await autoArrangeCanvas(input);
  verify(input, result);
  assert.equal(result.positions.a.y + 140, result.positions.b.y + 365);
  assert.equal(result.positions.b.y + 365, result.positions.c.y + 185);
  const label = result.routes["a-b"].label;
  assert.ok(label, "输入校验提示同样参与连线标签避让");
  assert.equal([...layoutWorldBoxes(input, result.positions).boxes.values()]
    .some((box) => overlaps({ x: label.x - 32, y: label.y - 12, width: 64, height: 24 }, box)), false);
});

test("局部整理保留未选中和锁定节点，并利用边界连线安排选区", async () => {
  const input = request([node("a", { x: 0, y: 100, locked: true }), node("b", { x: 400, y: 600 }), node("c", { x: 1400, y: 100 }), node("obstacle", { x: 500, y: 100 })], [edge("a", "b"), edge("b", "c")], { selection: ["a", "b"] });
  const original = structuredClone(input);
  const result = await autoArrangeCanvas(input);
  verify(input, result);
  assert.deepEqual(Object.keys(result.positions), ["b"]);
  assert.deepEqual(input, original);
  assert.ok(autoLayoutOperations(input, result).every((operation) => operation.expectedLayoutVersion === 7 && operation.nodeId === "b"));
});

test("分组整体移动时保留内部坐标，局部整理遵守分组边界，锁定后代不能被整体移动", async () => {
  const group = node("g", { x: 1000, y: 700, width: 1200, height: 900 }, "group");
  const a = node("a", { x: 40, y: 80, parentId: "g" });
  const b = node("b", { x: 40, y: 380, parentId: "g" });
  const input = request([group, a, b], [edge("a", "b")], { selection: ["a", "b"] });
  const result = await autoArrangeCanvas(input);
  const next = apply(input, result);
  assert.deepEqual(next.graph.nodes[0], group);
  for (const node of next.graph.nodes.slice(1)) {
    assert.equal(node.layout.parentId, "g");
    assert.ok(node.layout.x >= 24 && node.layout.x + 300 <= 1176);
    assert.ok(node.layout.y >= 64 && node.layout.y + 200 <= 876);
  }
  const whole = await autoArrangeCanvas({ ...input, selection: ["g"] });
  assert.deepEqual(Object.keys(whole.positions), ["g"]);
  const locked = { ...input, graph: { ...input.graph, nodes: [group, { ...a, layout: { ...a.layout, locked: true } }, b] }, selection: ["g"] };
  await assert.rejects(autoArrangeCanvas(locked), /锁定/);
});

test("执行回路和放不下的固定选区明确失败，不输出部分结果", async () => {
  await assert.rejects(autoArrangeCanvas(request([node("a"), node("b")], [edge("a", "b"), edge("b", "a")])), /循环/);
  const input = request([node("a", { x: 0 }), node("b", { x: 400 }), node("c", { x: 500 })], [edge("a", "b"), edge("b", "c")], { selection: ["b"] });
  await assert.rejects(autoArrangeCanvas(input), /空间/);
});

test("项目关系在没有执行边时仍形成层次，循环引用不逆转执行顺序", async () => {
  const input = request(["script", "episode", "scene", "board", "boardEpisode", "boardScene", "shot1", "shot2", "role", "roleItem", "sceneAsset", "sceneItem"].map((id) => node(id)));
  const structure = [["script", "episode"], ["episode", "scene"], ["board", "boardEpisode"], ["boardEpisode", "boardScene"], ["boardScene", "shot1"], ["boardScene", "shot2"], ["role", "roleItem"], ["sceneAsset", "sceneItem"]];
  const assets = [["episode", "boardEpisode"], ["role", "scene"], ["sceneAsset", "scene"], ["roleItem", "shot1"], ["roleItem", "shot2"], ["sceneItem", "shot1"], ["sceneItem", "shot2"]];
  input.relations = [...structure.map(([source, target]) => ({ source, target, category: "structure", label: "包含" })), ...assets.map(([source, target]) => ({ source, target, category: "assets", label: "素材" }))].map((relation) => ({ ...relation, id: `${relation.source}-${relation.target}` }));
  const result = await autoArrangeCanvas(input);
  verify(input, result);
  assert.equal(Object.keys(result.routes).length, 15);
  for (const [source, target] of structure) {
    assert.ok(result.positions[source].x < result.positions[target].x);
  }
  const loop = request(["a", "b", "c"].map((id) => node(id)), [edge("a", "b"), edge("b", "c"), edge("c", "a", "annotation")]);
  verify(loop, await autoArrangeCanvas(loop));
});

test("独立节点紧凑排布，100 节点 400 连线完成全部路径且不重叠", { timeout: 20000 }, async (context) => {
  const loose = request(Array.from({ length: 16 }, (_, index) => node(`loose-${index}`)));
  const looseResult = await autoArrangeCanvas(loose);
  const looseBounds = boundsOf([...verify(loose, looseResult).values()]);
  assert.ok(looseBounds.height < 2000 && looseBounds.width < 3000);
  const nodes = Array.from({ length: 100 }, (_, index) => node(`n${String(index).padStart(3, "0")}`, { x: index % 10 * 400, y: Math.floor(index / 10) * 300 }, ["text", "image", "video"][index % 3]));
  const edges = [];
  for (let column = 0; column < 9; column++) {
    for (let row = 0; row < 10; row++) {
      for (let branch = 0; branch < 4; branch++) {
        edges.push(edge(nodes[column * 10 + row].id, nodes[(column + 1) * 10 + (row + branch) % 10].id));
      }
    }
  }
  for (let index = 0; index < 40; index++) {
    edges.push(edge(nodes[index].id, nodes[index + 20].id));
  }
  const input = request(nodes, edges);
  const started = performance.now();
  const result = await autoArrangeCanvas(input);
  const elapsed = performance.now() - started;
  verify(input, result);
  assert.equal(Object.keys(result.positions).length, 100);
  assert.equal(Object.keys(result.routes).length, 400);
  context.diagnostic(`100 节点 / 400 连线：${elapsed.toFixed(0)} ms`);
});

test("多素材、多端口的 100 节点工作流保持可浏览比例，数字标签不遮挡节点，重复整理稳定", { timeout: 15000 }, async (context) => {
  // 保留原测试画布的连接拓扑，去除项目、文件和素材标识；保护大量素材输入时的耗时及布局比例。
  const fixture = JSON.parse(fs.readFileSync(path.join(root, "scripts/fixtures/canvas-layout-media.json"), "utf8"));
  const nodes = fixture.nodes.map((item, index) => node(`media-${index}`, {
    x: index % 10 * 400 + 0.375, y: Math.floor(index / 10) * 300 + 0.375,
  }, item.kind));
  const input = request(nodes, fixture.edges.map(([source, sourcePort, target, targetPort, inputIndex], index) => ({
    ...edge(nodes[source].id, nodes[target].id, "execution", `media-edge-${index}`), sourcePort, targetPort, inputIndex,
  })));
  fixture.nodes.forEach((item, index) => {
    const height = (["image", "video"].includes(item.kind) ? 320 : 160) + (item.inputs.length + item.outputs.length) * 32;
    input.geometry[nodes[index].id] = {
      width: 300, height,
      inputs: { "annotation-in": height / 2, ...Object.fromEntries(item.inputs.map((port, portIndex) => [port, height - (item.inputs.length + item.outputs.length - portIndex) * 32 + 16])) },
      outputs: { "annotation-out": height / 2, ...Object.fromEntries(item.outputs.map((port, portIndex) => [port, height - (item.outputs.length - portIndex) * 32 + 16])) },
    };
  });
  const started = performance.now();
  const result = await autoArrangeCanvas(input);
  context.diagnostic(`多素材工作流：${(performance.now() - started).toFixed(0)} ms`);
  const boxes = [...verify(input, result).values()];
  const bounds = boundsOf(boxes);
  assert.ok(bounds.height < 6000 && bounds.width < 10000, "大量根节点不能把画布排成长列");
  const edges = new Map(input.graph.edges.map((edge) => [edge.id, edge]));
  for (const [id, route] of Object.entries(result.routes)) {
    if (edges.get(id).inputIndex > 0) {
      assert.ok(route.label, "集合输入的序号必须保留");
      const width = String(edges.get(id).inputIndex + 1).length * 12 + 16;
      assert.equal(boxes.some((box) => overlaps({ x: route.label.x - width / 2, y: route.label.y - 12, width, height: 24 }, box)), false);
    }
  }
  const updated = apply(input, result);
  const repeated = await autoArrangeCanvas(updated);
  assert.deepEqual(repeated.positions, result.positions);
  assert.equal(autoLayoutOperations(updated, repeated).length, 0);
});
