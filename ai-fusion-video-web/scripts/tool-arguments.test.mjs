import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
const shared = "components/dashboard/shared/";
const cache = new Map();
const mocks = new Map([
  ["components/ui/button.tsx", { Button: ({ children, variant, size, ...props }) => React.createElement("button", props, children) }],
  [shared + "tool-call-json.tsx", { ToolCallJson: ({ code, label }) => React.createElement("pre", { "aria-label": label }, code) }],
  [shared + "tool-result-viewport.tsx", { ToolResultViewport: ({ children }) => children(false) }],
  [shared + "tool-approval-review.tsx", { ToolApprovalReviewContent: () => React.createElement("div", null, "审批摘要") }],
  ["components/dashboard/notification-panel/results.tsx", { ToolResultDisplay: () => null }],
]);
function load(relative) {
  const file = path.resolve(root, relative);
  const normalized = path.relative(root, file).split(path.sep).join("/");
  if (mocks.has(normalized)) {
    return mocks.get(normalized);
  }
  if (cache.has(file)) {
    return cache.get(file).exports;
  }
  const loadedModule = { exports: {} };
  cache.set(file, loadedModule);
  const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText;
  function localRequire(name) {
    if (!name.startsWith(".") && !name.startsWith("@/")) {
      return require(name);
    }
    const base = name.startsWith("@/") ? path.join(root, name.slice(2)) : path.resolve(path.dirname(file), name);
    const resolved = [base + ".ts", base + ".tsx"].find((item) => fs.existsSync(item) || mocks.has(path.relative(root, item).split(path.sep).join("/")));
    if (!resolved) {
      throw new Error(`Unresolved test import ${name}`);
    }
    return load(resolved);
  }
  vm.runInThisContext(`(function(require,module,exports){${compiled}\n})`, { filename: file })(localRequire, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}
load(shared + "tool-call-argument-model.test.ts");
const { ToolCallArguments } = load(shared + "tool-call-arguments.tsx");
const { ToolCallCard } = load(shared + "tool-call-card.tsx");
const broken = '{"operations":[{"node":{"config":{"content":"雨夜';

test("invalid tool arguments explain the next step without exposing the malformed protocol", () => {
  const html = renderToStaticMarkup(React.createElement(ToolCallArguments, {
    argumentsText: broken, toolName: "apply_canvas_operations",
  }));
  assert.match(html, /role="alert"/);
  assert.match(html, /操作信息不完整/);
  assert.doesNotMatch(html, /雨夜|原始调用参数|JSON|此工具无需调用参数/);
});

test("a canvas operation cannot be approved without its readable impact review", () => {
  const html = renderToStaticMarkup(React.createElement(ToolCallCard, {
    items: [{ toolCallId: "opaque", toolName: "apply_canvas_operations", argumentsText: '{"operations":[]}', status: "awaiting_approval" }],
    approval: { submitting: false, showActions: true, showCountdown: false,
      expiresAt: new Date(Date.now() + 3600000).toISOString(), onApprove() {}, onReject() {} },
  }));
  assert.match(html, /不能确认执行/);
  assert.match(html, /<button[^>]*disabled=""[^>]*>允许执行<\/button>/);
  assert.match(html, /<button(?![^>]*disabled)[^>]*>拒绝<\/button>/);
  assert.doesNotMatch(html, /operations|查看 JSON|调用参数/);
});

test("a readable canvas review keeps approval available without a protocol dump", () => {
  const html = renderToStaticMarkup(React.createElement(ToolCallCard, {
    items: [{ toolCallId: "reviewed", toolName: "apply_canvas_operations", argumentsText: '{"operations":[]}', status: "awaiting_approval" }],
    approval: { review: { title: "确认修改" }, submitting: false, showActions: true, showCountdown: false,
      expiresAt: new Date(Date.now() + 3600000).toISOString(), onApprove() {}, onReject() {} },
  }));
  assert.match(html, /审批摘要/);
  assert.match(html, /<button(?![^>]*disabled)[^>]*>允许执行<\/button>/);
  assert.doesNotMatch(html, /operations|查看 JSON|调用参数/);
});

const { CanvasToolResult } = load("components/dashboard/notification-panel/results/canvas.tsx");
test("canvas receipts show the real changed nodes but never model instructions or internal IDs", () => {
  const html = renderToStaticMarkup(React.createElement(CanvasToolResult, { toolName: "apply_canvas_operations", data: {
    status: "success", affectedNodes: [{ title: "灯塔旁白", id: "node-private-id" }], deletedNodeIds: [],
    commandId: "receipt-private-id", canvasRevision: 7, receipt: { privateProtocol: true },
    nextAction: "调用 await_canvas_run，随后读取内部响应", idMappings: { node: "node-private-id" },
  } }));
  assert.match(html, /已更新 1 个节点|灯塔旁白/);
  assert.doesNotMatch(html, /private-id|commandId|nextAction|await_canvas_run|canvasRevision|privateProtocol|idMappings/);
});

test("a restored invalid approval cannot be accepted but remains rejectable", () => {
  const html = renderToStaticMarkup(React.createElement(ToolCallCard, {
    items: [{ toolCallId: "incomplete", toolName: "apply_canvas_operations", argumentsText: broken, status: "awaiting_approval" }],
    approval: { review: { title: "确认修改" }, submitting: false, showActions: true, showCountdown: false,
      expiresAt: new Date(Date.now() + 3600000).toISOString(), onApprove() {}, onReject() {} },
  }));
  assert.match(html, /<button[^>]*disabled=""[^>]*>允许执行<\/button>/);
  assert.match(html, /<button(?![^>]*disabled)[^>]*>拒绝<\/button>/);
  assert.doesNotMatch(html, /审批摘要/);
});

test("specialized business result cards show saved content rather than database identities", () => {
  const { SaveEpisodeResult, CreateResult } = load("components/dashboard/notification-panel/results/resources.tsx");
  const episode = renderToStaticMarkup(React.createElement(SaveEpisodeResult, { data: { episodeNumber: 2, title: "灯塔来信", scriptEpisodeId: 987654, episode_version: 14 } }));
  assert.match(episode, /灯塔来信|保存成功/);
  assert.doesNotMatch(episode, /987654|分集ID|版本:|v14/);
  const asset = renderToStaticMarkup(React.createElement(CreateResult, { toolName: "create_asset", data: { status: "success", name: "阿青", assetId: 876543 } }));
  assert.match(asset, /阿青|创建成功/);
  assert.doesNotMatch(asset, /876543|资产ID/);
});

test("cancelled historical proposals never ask the user to reject them again", () => {
  const html = renderToStaticMarkup(React.createElement(ToolCallCard, {
    items: [{ toolCallId: "cancelled", toolName: "apply_canvas_operations", argumentsText: broken, status: "cancelled" }],
  }));
  assert.match(html, /已取消/);
  assert.doesNotMatch(html, /请拒绝|不能确认执行|允许执行|operations/);
});
