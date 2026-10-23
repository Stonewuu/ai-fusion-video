import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "..");
const flush = () => new Promise((resolve) => setImmediate(resolve));

function load(relative, mocks = {}) {
  const source = ts.transpileModule(fs.readFileSync(path.join(root, relative), "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const loadedModule = { exports: {} };
  const require = (name) => mocks[name] ?? load(
    path.posix.join(path.posix.dirname(relative), `${name}.ts`), mocks,
  );
  vm.runInNewContext(`(function(require,module,exports){${source}\n})`, {
    TextDecoder, AbortController, URLSearchParams, Error, setTimeout, clearTimeout, console,
  }, { filename: relative })(require, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}

const event = (sequence, outputType = "CONTENT") => ({
  schemaVersion: 1, runId: "run", sequence, outputType, content: `内容${sequence}`,
});
const frame = (value) => `id: run:${value.sequence}\r\ndata: ${JSON.stringify(value)}\r\n\r\n`;

function transport(response) {
  class ApiResultError extends Error {
    constructor(code, message) { super(message); this.code = code; }
  }
  return load("lib/api/ai-pipeline.ts", {
    "./client": { getApiBaseUrl: () => "http://localhost", http: {} },
    "./api-error": { ApiResultError, readApiResponseError: async () => "服务拒绝请求" },
    "./ai-assistant": { authenticatedFetch: async () => response },
  });
}

function receive(api, onEvent = () => {}) {
  return new Promise((resolve, reject) => {
    api.pipelineStream({}, { onEvent, onComplete: resolve, onError: reject });
  });
}

test("SSE decodes CRLF and Chinese UTF-8 split at every byte without premature events", async () => {
  const bytes = new TextEncoder().encode(frame(event(1)) + frame(event(2, "DONE")));
  const received = [];
  const stream = new ReadableStream({ start(controller) {
    for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
    controller.close();
  } });
  await receive(transport(new Response(stream)), (value) => received.push(value));
  assert.deepEqual(received.map((value) => value.content), ["内容1", "内容2"]);
});

test("first event is visible before the next event and terminal cancels an otherwise open connection", async () => {
  let writer;
  let cancelled = false;
  const received = [];
  const stream = new ReadableStream({ start(controller) { writer = controller; }, cancel() { cancelled = true; } });
  const done = receive(transport(new Response(stream)), (value) => received.push(value));
  writer.enqueue(new TextEncoder().encode(frame(event(1))));
  await flush();
  assert.equal(received.length, 1);
  assert.equal(cancelled, false);
  writer.enqueue(new TextEncoder().encode(frame(event(2, "DONE"))));
  await done;
  assert.equal(cancelled, true);
});

test("partial terminal frame at EOF cannot complete the business run", async () => {
  const incomplete = frame(event(2, "DONE")).slice(0, -4);
  await assert.rejects(receive(transport(new Response(frame(event(1)) + incomplete))), /完整事件/);
});

test("a rejected consumer event closes the response and does not call completion", async () => {
  let cancelled = false;
  const stream = new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode(frame(event(1))));
  }, cancel() { cancelled = true; } });
  await assert.rejects(receive(transport(new Response(stream)), () => { throw new Error("无法应用事件"); }), /无法应用事件/);
  assert.equal(cancelled, true);
});

test("HTTP 200 platform error envelopes are recognized as rejected starts", async () => {
  const response = new Response(JSON.stringify({ code: 500, msg: "配置未完成", data: null }), {
    headers: { "Content-Type": "application/json" },
  });
  await assert.rejects(receive(transport(response)), (error) => error.code === 500 && error.requestRejected === true);
});

function coordinatorHarness(t, response) {
  let callbacks;
  let loaded = 0;
  const runtime = {
    conversation: { conversationId: "c", status: "running" }, status: "running", statusConfirmed: true,
    knownRunId: "run", pipeline: { runId: "run", lastSequence: 0, timeline: [], reasoningText: "" },
    messagesLoaded: false,
  };
  let state = {
    mode: "docked", selectedConversationId: "c", connectionGeneration: 0, connection: null,
    conversations: [runtime.conversation], conversationStates: { c: runtime },
    loadMessagesIfNeeded: async () => { loaded++; },
  };
  const runtimeMocks = {
    hasLiveTranscript: (value) => value.pipeline.timeline.length > 0,
    hasTerminal: (value) => value.outputType === "DONE",
    pendingPipelineForNextRun: () => ({ lastSequence: 0, timeline: [], reasoningText: "" }),
    reduceAssistantEvent: (pipeline, value) => {
      if (value.invalid) throw new Error("状态不一致");
      return { ...pipeline, timeline: [...pipeline.timeline, value.content] };
    },
    statusFromPipeline: (value) => value === "COMPLETED" ? "completed" : "running",
    statusIsRunning: (value) => value === "running",
    terminalStatusForEvent: () => "completed",
  };
  const apiMocks = {
    getPipelineStatus: async () => response,
    pipelineStream: (_, value) => { callbacks = value; return new AbortController(); },
    reconnectPipelineStream: (_, __, value) => { callbacks = value; return new AbortController(); },
  };
  const { createAssistantConnectionCoordinator } = load("lib/store/assistant-connection-coordinator.ts", {
    "@/lib/api/api-error": { getApiErrorStatus: (error) => error.code ?? null },
    "@/lib/api/ai-pipeline": apiMocks,
    "@/lib/canvas/assistant-session": { receiveCanvasAssistantResult: () => {} },
    "./assistant-runtime": runtimeMocks,
  });
  const coordinator = createAssistantConnectionCoordinator((value) => {
    state = { ...state, ...(typeof value === "function" ? value(state) : value) };
  }, () => state);
  t.after(() => coordinator.reset());
  return { coordinator, state: () => state, callbacks: () => callbacks, loaded: () => loaded };
}

test("a reducer error does not poison deduplication or acknowledge the rejected sequence", (t) => {
  const h = coordinatorHarness(t);
  h.coordinator.startConnection("c", {});
  h.callbacks().onEvent(event(1));
  assert.throws(() => h.callbacks().onEvent({ ...event(2), invalid: true }), /状态不一致/);
  assert.equal(h.state().conversationStates.c.pipeline.lastSequence, 1);
  h.callbacks().onEvent(event(2));
  assert.equal(h.state().conversationStates.c.pipeline.lastSequence, 2);
});

test("completion while disconnected reloads the full transcript even when partial text is visible", async (t) => {
  const h = coordinatorHarness(t, { runId: "run", status: "COMPLETED", lastSequence: 5 });
  Object.assign(h.state().conversationStates.c, {
    statusConfirmed: false,
    pipeline: { runId: "run", lastSequence: 1, timeline: ["半段"], reasoningText: "" },
  });
  h.coordinator.ensureConnection();
  await flush();
  assert.equal(h.state().conversationStates.c.status, "completed");
  assert.ok(h.loaded() >= 2, "加载初始历史后仍须补全结束时的历史");
});

test("an ambiguous transport failure before the first event stays recoverable without resending", (t) => {
  const h = coordinatorHarness(t);
  h.coordinator.startConnection("c", {});
  h.callbacks().onError(new Error("连接中断"));
  assert.equal(h.state().conversationStates.c.status, "running");
  assert.equal(h.state().conversationStates.c.statusConfirmed, false);
});

test("an explicit platform rejection cannot leave a new conversation permanently running", (t) => {
  const h = coordinatorHarness(t);
  h.coordinator.startConnection("c", {});
  h.callbacks().onError(Object.assign(new Error("配置未完成"), { code: 500, requestRejected: true }));
  assert.equal(h.state().conversationStates.c.status, "failed");
  assert.equal(h.state().connection, null);
});
