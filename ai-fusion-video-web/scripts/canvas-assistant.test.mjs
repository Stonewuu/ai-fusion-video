import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
function load(relative) {
  const file = path.join(root, relative);
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const loadedModule = { exports: {} };
  vm.runInThisContext(`(function(require,module,exports){${code}\n})`, { filename: file })(require, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}
const session = load("lib/canvas/assistant-session.ts");
const references = load("components/dashboard/assistant/message-references.ts");
const data = new Map();
globalThis.sessionStorage = {
  getItem: (key) => data.get(key) ?? null,
  setItem: (key, value) => data.set(key, value),
  removeItem: (key) => data.delete(key),
};
function capture(id) {
  return { title: `画布 ${id}`, projectId: 7, context: {
    canvasId: id, selectedNodeIds: ["selected"], visibleNodeIds: ["selected"], focusedResource: null, lastSequence: 5,
  } };
}
function binding(id, userId, focus) {
  return { canvasId: id, title: id, projectId: 7, userId, selectedCount: 1, prepare: async () => capture(id), focus };
}
function event(toolName, result, outputType = "TOOL_FINISHED") {
  return { outputType, toolName, toolResult: JSON.stringify(result), runId: "run-1", sequence: 9 };
}

test("an Agent bound to another canvas cannot move the current viewport", () => {
  const focused = [];
  session.attachCanvasAssistant(binding("A", 1, (ids) => focused.push(ids)));
  session.rememberCanvasAssistantCapture("conversation-1", capture("A"));
  session.attachCanvasAssistant(binding("B", 1, (ids) => focused.push(ids)));
  session.receiveCanvasAssistantResult("conversation-1", event("focus_canvas_nodes", {
    status: "success", focus: { canvasId: "A", agentRunId: "run-1", nodeIds: ["result"] },
  }));
  assert.deepEqual(focused, []);
});

test("only committed platform tool receipts highlight nodes and following remains optional", () => {
  const focused = [];
  session.attachCanvasAssistant(binding("A", 2, (ids) => focused.push(ids)));
  session.rememberCanvasAssistantCapture("conversation-2", capture("A"));
  session.useCanvasAssistantSession.setState({ following: false, highlight: null });
  const focus = { status: "success", focus: { canvasId: "A", agentRunId: "run-1", nodeIds: ["actual"] } };
  session.receiveCanvasAssistantResult("conversation-2", event("focus_canvas_nodes", focus));
  assert.deepEqual(focused, []);
  const receipt = { status: "success", receipt: { canvasId: "A", changes: { upsertedNodes: [{ id: "actual" }] } } };
  session.receiveCanvasAssistantResult("conversation-2", event("apply_canvas_operations", receipt, "CONTENT"));
  session.receiveCanvasAssistantResult("conversation-2", event("mcp_canvas_fake", receipt));
  assert.equal(session.useCanvasAssistantSession.getState().highlight, null);
  session.receiveCanvasAssistantResult("conversation-2", event("apply_canvas_operations", receipt));
  assert.deepEqual(session.useCanvasAssistantSession.getState().highlight.nodeIds, ["actual"]);
  session.useCanvasAssistantSession.getState().setFollowing(true);
  session.receiveCanvasAssistantResult("conversation-2", event("focus_canvas_nodes", focus));
  assert.deepEqual(focused, [["actual"]]);
});

test("account changes discard the previous user's pending viewport hints", () => {
  const focused = [];
  const detach = session.attachCanvasAssistant(binding("A", 3, (ids) => focused.push(ids)));
  session.rememberCanvasAssistantCapture("old-conversation", capture("A"));
  detach();
  session.attachCanvasAssistant(binding("A", 4, (ids) => focused.push(ids)));
  session.receiveCanvasAssistantResult("old-conversation", event("focus_canvas_nodes", {
    status: "success", focus: { canvasId: "A", agentRunId: "run-1", nodeIds: ["private-result"] },
  }));
  assert.deepEqual(focused, []);
});

test("submission saves only relevant mounted drafts and preserves conflicting or closed drafts", async () => {
  data.clear();
  data.set("selected", "pending draft"); data.set("unrelated", "keep this draft");
  let saved = 0;
  const unregister = session.registerCanvasAssistantDraft("selected", async () => { saved++; data.delete("selected"); });
  await session.saveCanvasAssistantDrafts(["selected"]);
  assert.equal(saved, 1); assert.equal(data.get("unrelated"), "keep this draft");
  unregister();
  data.set("closed", "unsaved text");
  await assert.rejects(session.saveCanvasAssistantDrafts(["closed"]), /未保存或冲突/);
  assert.equal(data.get("closed"), "unsaved text");
  const conflict = session.registerCanvasAssistantDraft("closed", async () => { throw new Error("版本冲突"); });
  await assert.rejects(session.saveCanvasAssistantDrafts(["closed"]), /版本冲突/);
  assert.equal(data.get("closed"), "unsaved text");
  conflict();
});

test("saved messages retain the original canvas and selection for later review", () => {
  const parsed = references.parseAssistantMessageReferences(JSON.stringify({ version: 2, projectId: 7,
    canvas: { id: "canvas-old", title: "原画布", selectedNodeIds: ["node-1", "node-2"] }, skills: [], mcpTools: [], attachments: [] }));
  assert.deepEqual(parsed.canvas, { id: "canvas-old", title: "原画布", selectedNodeIds: ["node-1", "node-2"] });
  assert.equal(references.parseAssistantMessageReferences(JSON.stringify({ version: 2, canvas: { id: "wrong", selectedNodeIds: "bad" } })).canvas, undefined);
});
