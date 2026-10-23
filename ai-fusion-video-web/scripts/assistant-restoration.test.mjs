import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
const flush = () => new Promise((resolve) => setImmediate(resolve));
const key = (id) => `fusion-assistant:${id}:v1`;
const conversation = (id) => ({ id: 1, conversationId: id, category: "assistant", title: id, status: "completed" });

function harness(t, realCoordinator = false) {
  const data = new Map();
  const requests = [];
  const messageRequests = [];
  let deferredMessages = false;
  const cache = new Map();
  const reconnects = [];
  const metadata = new Map();
  const metadataRequests = [];
  const noop = () => {};
  const mocks = {
    "@/lib/canvas/assistant-session": { rememberCanvasAssistantCapture: noop },
    "@/lib/api/ai-assistant": {
      listConversations: (params) => new Promise((resolve, reject) => requests.push({ params, resolve, reject })),
      getConversation: async (id) => {
        metadataRequests.push(id);
        const value = metadata.get(id);
        if (value instanceof Error) throw value;
        if (!value) throw Object.assign(new Error("对话不存在"), { status: 404 });
        return value;
      },
      listMessages: async () => deferredMessages
        ? new Promise((resolve) => messageRequests.push({ resolve }))
        : [],
      deleteConversation: async () => {},
      deleteConversationByConversationId: async () => {},
    },
    "@/lib/api/ai-pipeline": {
      cancelPipeline: async () => {},
      reconnectPipelineStream: (runId, sequence) => {
        reconnects.push({ runId, sequence });
        return new AbortController();
      },
    },
    "@/lib/api/api-error": { getApiErrorStatus: (error) => error.status ?? null },
    "@/components/dashboard/assistant/geometry": {
      clampDockWidth: (value) => value,
      clampLauncherPosition: (value) => value,
      clampRect: (value) => value,
      getDefaultLauncherPosition: () => ({ x: 0, y: 0 }),
      getDefaultNormalRect: () => ({ x: 0, y: 0, width: 520, height: 600 }),
    },
    "./assistant-connection-coordinator": {
      createAssistantConnectionCoordinator: () => ({
        reset: noop, scheduleStatusPolling: noop, ensureConnection: noop, invalidateConnection: noop,
      }),
    },
    "./assistant-runtime": {
      makeRuntime: (item, drafts, runIds, mode) => ({
        conversation: item, messages: [], pipeline: { lastSequence: 0, timeline: [], runId: runIds[item.conversationId] },
        draft: drafts[item.conversationId] ?? "", knownRunId: runIds[item.conversationId],
        toolExecutionMode: mode, status: item.status, messagesLoaded: false, messagesLoading: false,
      }),
      mergeMessages: (_, incoming) => incoming,
      timelineForMessages: () => [],
      statusIsRunning: (status) => ["running", "pending", "RUNNING", "WAITING_CONFIRMATION", "WAITING_EXTERNAL", "CANCEL_REQUESTED"].includes(status),
      uniqueConversations: (first, second) => [...new Map([...first, ...second].map((item) => [item.conversationId, item])).values()],
    },
  };
  if (realCoordinator) delete mocks["./assistant-connection-coordinator"];
  const context = vm.createContext({
    window: { innerWidth: 1280, innerHeight: 900 },
    localStorage: { getItem: (name) => data.get(name) ?? null, setItem: (name, value) => data.set(name, value) },
    setTimeout, clearTimeout, console, AbortController, crypto,
  });
  function load(relative) {
    if (cache.has(relative)) return cache.get(relative);
    const filename = path.join(root, relative);
    const code = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const loadedModule = { exports: {} };
    cache.set(relative, loadedModule.exports);
    const scopedRequire = (name) => mocks[name] ?? (name.startsWith(".")
      ? load(path.posix.join(path.posix.dirname(relative), `${name}.ts`))
      : require(name));
    vm.runInContext(`(function(require,module,exports){${code}\n})`, context, { filename })(scopedRequire, loadedModule, loadedModule.exports);
    return loadedModule.exports;
  }
  const store = load("lib/store/assistant-store.ts").useAssistantStore;
  const persistence = load("lib/store/assistant-persistence.ts");
  t.after(() => { store.getState().resetForUser(); persistence.clearAssistantPersistTimer(); });
  const seed = (userId, selected = "saved") => data.set(key(userId), JSON.stringify({
    schemaVersion: 1, geometryVersion: 2, mode: "docked", selectedConversationId: selected,
    drafts: { saved: "未发送的内容", older: "较早会话的草稿" },
    runIds: { saved: "durable-run", older: "older-run" },
    lastSequences: { saved: 42, older: 8 }, toolExecutionModes: { saved: "ALWAYS_ASK", older: "DEFAULT" },
  }));
  return { store, requests, data, seed, persistence, reconnects, metadata, metadataRequests, messageRequests,
    deferMessages: () => { deferredMessages = true; }, saved: (id = 1) => JSON.parse(data.get(key(id))) };
}

test("history requested while running cannot replace a newer completed live transcript", async (t) => {
  const h = harness(t);
  h.seed(1);
  h.store.getState().initializeForUser(1);
  h.requests[0].resolve({ list: [conversation("saved")], total: 1 });
  await flush();
  h.deferMessages();
  h.store.setState((state) => ({ conversationStates: { ...state.conversationStates,
    saved: { ...state.conversationStates.saved, status: "running", messagesLoaded: false, messagesLoading: false,
      knownRunId: "run", pipeline: { runId: "run", lastSequence: 1, timeline: ["半段"] } },
  } }));
  const pending = h.store.getState().loadMessagesIfNeeded("saved");
  h.store.setState((state) => ({ conversationStates: { ...state.conversationStates,
    saved: { ...state.conversationStates.saved, status: "completed",
      pipeline: { runId: "run", lastSequence: 5, timeline: ["完整正文"] } },
  } }));
  h.messageRequests[0].resolve([]);
  await flush();
  assert.equal(h.messageRequests.length, 2);
  assert.equal(h.store.getState().conversationStates.saved.messagesLoaded, false);
  assert.deepEqual(h.store.getState().conversationStates.saved.pipeline.timeline, ["完整正文"]);
  h.messageRequests[1].resolve([{ id: 10, conversationId: "saved", role: "assistant", content: "完整正文", messageOrder: 2 }]);
  await pending;
  assert.equal(h.store.getState().conversationStates.saved.messagesLoaded, true);
  assert.equal(h.store.getState().conversationStates.saved.messages[0].content, "完整正文");
});

test("mount cleanup before hydration preserves selected conversation, draft and durable run", async (t) => {
  const h = harness(t);
  h.seed(1);
  h.store.getState().initializeForUser(1);
  h.store.getState().resetForUser();
  assert.equal(h.saved().selectedConversationId, "saved");
  assert.equal(h.saved().drafts.saved, "未发送的内容");
  h.store.getState().initializeForUser(1);
  h.requests[1].resolve({ list: [conversation("saved")], total: 1 });
  await flush();
  assert.equal(h.store.getState().selectedConversationId, "saved");
  assert.equal(h.store.getState().conversationStates.saved.knownRunId, "durable-run");
  assert.equal(h.store.getState().conversationStates.saved.toolExecutionMode, "ALWAYS_ASK");
  h.requests[0].resolve({ list: [], total: 0 });
  await flush();
  assert.equal(h.store.getState().selectedConversationId, "saved", "stale first mount must not erase the restored state");
});

test("geometry persistence during hydration and early cleanup retain previous conversation data", (t) => {
  const h = harness(t);
  h.seed(1);
  h.store.getState().initializeForUser(1);
  h.store.getState().setDraft(null, "恢复期间的新草稿");
  h.store.getState().updateDockWidth(600, 1200, true);
  assert.equal(h.saved().dockWidth, 600);
  assert.equal(h.saved().selectedConversationId, "saved");
  assert.equal(h.saved().drafts.__new__, "恢复期间的新草稿");
  assert.equal(h.saved().drafts.older, "较早会话的草稿");
  h.store.getState().resetForUser();
  assert.equal(h.saved().runIds.saved, "durable-run");
});

test("choosing a new conversation during hydration is not overwritten by the late list", async (t) => {
  const h = harness(t);
  h.seed(1);
  h.store.getState().initializeForUser(1);
  h.store.getState().startNewConversation();
  h.store.getState().setDraft(null, "这是新对话");
  h.requests[0].resolve({ list: [conversation("saved")], total: 1 });
  await flush();
  assert.equal(h.store.getState().selectedConversationId, null);
  assert.equal(h.store.getState().newDraft, "这是新对话");
});

test("saving the first page preserves unloaded drafts but explicitly deleted conversations are removed", async (t) => {
  const h = harness(t);
  h.seed(1);
  h.store.getState().initializeForUser(1);
  h.requests[0].resolve({ list: [conversation("saved")], total: 2 });
  await flush();
  h.persistence.commitAssistantPersist(h.store.getState);
  assert.equal(h.saved().drafts.older, "较早会话的草稿");
  h.store.getState().setDraft("saved", "");
  h.persistence.commitAssistantPersist(h.store.getState);
  assert.equal(h.saved().drafts.saved, undefined, "clearing a draft must not restore its saved contents");
  await h.store.getState().deleteConversation("saved", 1);
  h.persistence.commitAssistantPersist(h.store.getState);
  assert.equal(h.saved().runIds.saved, undefined);
  assert.equal(h.saved().toolExecutionModes.saved, undefined);
  assert.equal(h.saved().drafts.older, "较早会话的草稿");
});

test("switching accounts and returning ignores responses from the old account session", async (t) => {
  const h = harness(t);
  h.seed(1);
  h.seed(2, "other");
  h.store.getState().initializeForUser(1);
  h.store.getState().initializeForUser(2);
  h.store.getState().initializeForUser(1);
  h.requests[2].resolve({ list: [conversation("saved")], total: 1 });
  await flush();
  h.requests[0].resolve({ list: [conversation("stale")], total: 1 });
  h.requests[1].resolve({ list: [conversation("other")], total: 1 });
  await flush();
  assert.equal(h.store.getState().selectedConversationId, "saved");
  assert.equal(h.store.getState().conversationStates.stale, undefined);
  assert.equal(h.store.getState().conversationStates.other, undefined);
  assert.equal(h.saved(2).selectedConversationId, "other");
});

test("cancelling a suspended live request reconnects its durable journal without another model start", async (t) => {
  const h = harness(t, true);
  h.seed(1);
  h.store.getState().initializeForUser(1);
  h.requests[0].resolve({ list: [conversation("saved")], total: 1 });
  await flush();
  const controller = new AbortController();
  const runtime = h.store.getState().conversationStates.saved;
  h.store.setState({
    connection: { conversationId: "saved", runId: "durable-run", controller, connectionGeneration: 7, connectionMode: "start" },
    connectionGeneration: 7,
    conversationStates: { saved: { ...runtime, status: "WAITING_CONFIRMATION", statusConfirmed: true,
      pipeline: { ...runtime.pipeline, status: "waiting_confirmation", runId: "durable-run", lastSequence: 20 } } },
  });
  await h.store.getState().stopGeneration();
  assert.equal(controller.signal.aborted, true, "the suspended request must stop owning the active connection slot");
  assert.equal(h.reconnects.length, 1);
  assert.equal(h.reconnects[0].runId, "durable-run");
  assert.equal(h.reconnects[0].sequence, 20, "continue from the existing journal cursor instead of starting a new run");
  assert.equal(h.store.getState().connection.connectionMode, "reconnect");
});

test("an older selected conversation restores directly without shifting the list pagination", async (t) => {
  const h = harness(t);
  h.seed(1, "older");
  h.metadata.set("older", conversation("older"));
  h.store.getState().initializeForUser(1);
  h.requests[0].resolve({ list: Array.from({ length: 20 }, (_, index) => conversation(`recent-${index}`)), total: 35 });
  await flush();
  assert.equal(h.store.getState().selectedConversationId, "older");
  assert.equal(h.store.getState().conversationStates.older.draft, "较早会话的草稿");
  assert.equal(h.store.getState().conversationStates.older.knownRunId, "older-run");
  assert.deepEqual(h.metadataRequests, ["older"]);
  h.store.getState().loadMoreConversations();
  assert.equal(h.requests[1].params.pageNo, 2);
  h.requests[1].resolve({ list: [conversation("older"), ...Array.from({ length: 14 }, (_, index) => conversation(`earlier-${index}`))], total: 35 });
  await flush();
  assert.equal(h.store.getState().conversations.length, 35);
  assert.equal(h.store.getState().hasMoreConversations, false);
});

test("a missing or inaccessible selection clears without exposing it in the assistant list", async (t) => {
  const h = harness(t);
  h.seed(1, "removed");
  h.store.getState().initializeForUser(1);
  h.requests[0].resolve({ list: [conversation("saved")], total: 1 });
  await flush();
  h.persistence.commitAssistantPersist(h.store.getState);
  assert.equal(h.store.getState().selectedConversationId, null);
  assert.equal(h.store.getState().conversationsError, undefined);
  assert.equal(h.saved().selectedConversationId, null);
});

test("a transport error keeps the saved selection and an explicit reload can recover it", async (t) => {
  const h = harness(t);
  h.seed(1, "older");
  h.metadata.set("older", Object.assign(new Error("离线"), { status: 503 }));
  h.store.getState().initializeForUser(1);
  h.requests[0].resolve({ list: [conversation("saved")], total: 2 });
  await flush();
  assert.ok(h.store.getState().conversationsError);
  h.persistence.commitAssistantPersist(h.store.getState);
  assert.equal(h.saved().selectedConversationId, "older");
  h.metadata.set("older", conversation("older"));
  h.store.getState().initializeForUser(1);
  h.requests[1].resolve({ list: [conversation("saved")], total: 2 });
  await flush();
  assert.equal(h.store.getState().selectedConversationId, "older");
  assert.equal(h.store.getState().conversationsError, undefined);
});
