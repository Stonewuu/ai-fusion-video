import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import ts from "typescript";

const filename = path.resolve(import.meta.dirname, "../lib/api/session-refresh-coordinator.ts");
const compiled = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`, { filename })(loaded, loaded.exports);
const { createSessionRefresher, RefreshRejectedError } = loaded.exports;
const initial = { userId: 7, accessToken: "expired", refreshToken: "single-use" };
const rotated = { userId: 7, accessToken: "new-access", refreshToken: "new-refresh" };

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((accept, decline) => {
    resolve = accept;
    reject = decline;
  });
  return { promise, resolve, reject };
}

function fixture() {
  let stored = initial;
  let requests = 0;
  let publishes = 0;
  let waiting = 0;
  const response = deferred();
  const otherTab = deferred();
  const refresh = createSessionRefresher({
    read: () => stored,
    exchange: async () => {
      requests += 1;
      return response.promise;
    },
    publish: async (_previous, next) => {
      stored = next;
      publishes += 1;
    },
    waitForRotation: async () => {
      waiting += 1;
      return otherTab.promise;
    },
  });
  return {
    refresh, response, otherTab,
    set: (value) => {
      stored = value;
    },
    state: () => ({ stored, requests, publishes, waiting }),
  };
}

test("concurrent HTTP and SSE consumers share one replacement session", async () => {
  const f = fixture();
  const consumers = Array.from({ length: 8 }, () => f.refresh(initial.accessToken, initial.userId));
  f.response.resolve(rotated);
  assert.deepEqual(await Promise.all(consumers), Array(8).fill(rotated));
  assert.deepEqual(f.state(), { stored: rotated, requests: 1, publishes: 1, waiting: 0 });
});

test("late rejection of an old access token reuses the already rotated session", async () => {
  const f = fixture();
  f.set(rotated);
  assert.equal(await f.refresh(initial.accessToken, initial.userId), rotated);
  assert.equal(f.state().requests, 0);
});

test("a refresh finishing after logout cannot recreate the logged out session", async () => {
  const f = fixture();
  const pending = f.refresh(initial.accessToken, initial.userId);
  f.set(null);
  f.response.resolve(rotated);
  await assert.rejects(pending, /会话已变化/);
  assert.equal(f.state().stored, null);
  assert.equal(f.state().publishes, 0);
});

test("a refresh finishing after switching users cannot overwrite the new account", async () => {
  const f = fixture();
  const pending = f.refresh(initial.accessToken, initial.userId);
  const other = { userId: 8, accessToken: "other-access", refreshToken: "other-refresh" };
  f.set(other);
  f.response.resolve(rotated);
  await assert.rejects(pending, /会话已变化/);
  await assert.rejects(f.refresh(initial.accessToken, initial.userId), /会话已变化/);
  assert.equal(f.state().stored, other);
  assert.equal(f.state().publishes, 0);
});

test("another tab consuming the credential waits for its published replacement", async () => {
  const f = fixture();
  const pending = f.refresh(initial.accessToken, initial.userId);
  f.response.reject(new RefreshRejectedError(401, "consumed"));
  f.set(rotated);
  f.otherTab.resolve(rotated);
  assert.equal(await pending, rotated);
  assert.deepEqual(f.state(), { stored: rotated, requests: 1, publishes: 0, waiting: 1 });
});

test("an expired refresh without another tab's replacement stays a failure", async () => {
  const f = fixture();
  const pending = f.refresh(initial.accessToken, initial.userId);
  f.response.reject(new RefreshRejectedError(401, "expired"));
  await Promise.resolve();
  f.otherTab.reject(new Error("expired session"));
  await assert.rejects(pending, /expired session/);
  assert.equal(f.state().publishes, 0);
});

test("network failures do not wait for or invent a replacement session", async () => {
  const f = fixture();
  const pending = f.refresh(initial.accessToken, initial.userId);
  f.response.reject(new Error("offline"));
  await assert.rejects(pending, /offline/);
  assert.deepEqual(f.state(), { stored: initial, requests: 1, publishes: 0, waiting: 0 });
});

test("a response for a different account is rejected without changing storage", async () => {
  const f = fixture();
  const pending = f.refresh(initial.accessToken, initial.userId);
  f.response.resolve({ ...rotated, userId: 8 });
  await assert.rejects(pending, /会话已变化/);
  assert.equal(f.state().stored, initial);
  assert.equal(f.state().publishes, 0);
});
