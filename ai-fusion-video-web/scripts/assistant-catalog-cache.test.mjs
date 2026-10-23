import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import ts from "typescript";

function harness() {
  const requests = [];
  let now = 100_000;
  const request = (kind, agentType) => new Promise((resolve, reject) => {
    requests.push({ kind, agentType, resolve, reject });
  });
  const mocks = {
    "@/lib/api/ai-model": { aiModelApi: { listByType: (type) => request(`models:${type}`) } },
    "@/lib/api/project": { projectApi: { list: () => request("projects") } },
    "@/lib/api/ai-assistant": { getAssistantReferenceOptions: (agentType) => request("references", agentType) },
  };
  const file = path.resolve(import.meta.dirname, "../components/dashboard/assistant/catalog-cache.ts");
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const loaded = { exports: {} };
  vm.runInNewContext(`(function(require,module,exports){${code}\n})`, { Date: { now: () => now } }, { filename: file })(
    (name) => {
      assert.ok(mocks[name], `Unexpected dependency: ${name}`);
      return mocks[name];
    }, loaded, loaded.exports,
  );
  return { cache: loaded.exports, requests, advance: (time) => { now += time; } };
}

test("same account shares pending requests and cached values only until expiry", async () => {
  const { cache, requests, advance } = harness();
  const first = cache.loadAssistantModels(1);
  assert.equal(cache.loadAssistantModels(1), first);
  assert.equal(requests.length, 1);
  assert.equal(cache.hasCachedAssistantModels(1), false);
  const models = [{ id: 11 }];
  requests[0].resolve(models);
  assert.equal(await first, models);
  assert.equal(await cache.loadAssistantModels(1), models);
  assert.equal(cache.hasCachedAssistantModels(1), true);
  assert.equal(cache.hasCachedAssistantModels(2), false);
  advance(30_001);
  assert.equal(cache.hasCachedAssistantModels(1), false);
  const refreshed = cache.loadAssistantModels(1);
  assert.equal(requests.length, 2);
  requests[1].resolve([{ id: 12 }]);
  assert.equal((await refreshed)[0].id, 12);
});

test("account changes cannot reuse another account's model, project or capability list", async () => {
  const { cache, requests } = harness();
  for (const load of [
    (id) => cache.loadAssistantModels(id),
    (id) => cache.loadAssistantProjects(id),
    (id) => cache.loadAssistantReferenceOptions(id, "canvas_assistant"),
  ]) {
    const first = load(1);
    const firstData = [{ owner: 1 }];
    requests.at(-1).resolve(firstData);
    assert.equal(await first, firstData);
    const beforeSwitch = requests.length;
    const second = load(2);
    assert.equal(requests.length, beforeSwitch + 1);
    const secondData = [{ owner: 2 }];
    requests.at(-1).resolve(secondData);
    assert.equal(await second, secondData);
  }
});

test("late results from before A to B to A cannot poison the new A cache", async () => {
  const { cache, requests } = harness();
  const oldA = cache.loadAssistantProjects(1);
  const second = cache.loadAssistantProjects(2);
  const newA = cache.loadAssistantProjects(1);
  assert.equal(requests.length, 3);
  const newData = [{ id: 13, name: "current A" }];
  requests[2].resolve(newData);
  await newA;
  requests[0].resolve([{ id: 11, name: "stale A" }]);
  requests[1].resolve([{ id: 22, name: "B" }]);
  await Promise.all([oldA, second]);
  assert.equal(await cache.loadAssistantProjects(1), newData);
  assert.equal(requests.length, 3);
});

test("canvas and ordinary assistants keep separate MCP catalogs for the same account", async () => {
  const { cache, requests } = harness();
  const ordinary = cache.loadAssistantReferenceOptions(1, "ai_assistant_agent");
  const canvas = cache.loadAssistantReferenceOptions(1, "canvas_assistant");
  assert.deepEqual(requests.map((item) => item.agentType), ["ai_assistant_agent", "canvas_assistant"]);
  const ordinaryData = { skills: [], mcpTools: [{ toolName: "ordinary" }] };
  const canvasData = { skills: [], mcpTools: [{ toolName: "canvas" }] };
  requests[0].resolve(ordinaryData);
  requests[1].resolve(canvasData);
  await Promise.all([ordinary, canvas]);
  assert.equal(await cache.loadAssistantReferenceOptions(1, "canvas_assistant"), canvasData);
  assert.equal(await cache.loadAssistantReferenceOptions(1, "ai_assistant_agent"), ordinaryData);
  assert.equal(requests.length, 2);
});

test("failed requests can be retried and absent identity never starts a request", async () => {
  const { cache, requests } = harness();
  for (const id of [0, -1, NaN]) assert.throws(() => cache.loadAssistantModels(id), /登录身份/);
  assert.equal(requests.length, 0);
  const failed = cache.loadAssistantModels(1);
  requests[0].reject(new Error("offline"));
  await assert.rejects(failed, /offline/);
  const retried = cache.loadAssistantModels(1);
  requests[1].resolve([]);
  assert.equal((await retried).length, 0);
  assert.equal(cache.hasCachedAssistantModels(1), true);
});
