import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

function fixture() {
  const calls = [],
    timers = new Map();
  let timerId = 0;
  const request = (kind, id, retryJobId, signal) =>
    new Promise((resolve, reject) =>
      calls.push({ kind, id, retryJobId, signal, resolve, reject }),
    );
  const require = (name) =>
    name.endsWith("media-preview")
      ? {
          mediaPreviewApi: {
            ensure: (id, retry, signal) => request("ensure", id, retry, signal),
            get: (id, signal) => request("get", id, null, signal),
          },
        }
      : {
          getApiErrorMessage: (error) => error.message,
          getApiErrorStatus: (error) => error.status ?? null,
        };
  const compiled = ts.transpileModule(
    fs.readFileSync(
      new URL("../lib/media-preview-cache.ts", import.meta.url),
      "utf8",
    ),
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.CommonJS,
      },
    },
  ).outputText;
  const loaded = { exports: {} };
  vm.runInThisContext(
    `(function(require,module,exports,setTimeout,clearTimeout){${compiled}\n})`,
  )(
    require,
    loaded,
    loaded.exports,
    (callback) => {
      const id = ++timerId;
      timers.set(id, callback);
      return id;
    },
    (id) => timers.delete(id),
  );
  return { ...loaded.exports, calls, timers };
}
const ready = {
  sourceId: "original",
  sourceVersion: 2,
  profile: "canvas_640_v1",
  jobId: "job",
  status: "SUCCEEDED",
  previewId: "preview",
  width: 640,
  height: 480,
  sampleAtMs: 100,
  canRetry: false,
  error: null,
};
const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

test("same scoped media shares one admission and stops polling when its preview is ready", async () => {
  const f = fixture();
  const a = f.subscribePreview("1:team:original", "original", () => {});
  const b = f.subscribePreview("1:team:original", "original", () => {});
  assert.equal(f.calls.length, 1);
  f.calls[0].resolve(ready);
  await flush();
  assert.equal(f.previewSnapshot("1:team:original").view.previewId, "preview");
  assert.equal(f.timers.size, 0);
  a();
  assert.equal(f.calls[0].signal.aborted, false);
  b();
  assert.equal(f.calls[0].signal.aborted, true);
});
test("leaving a viewport cancels observation and a stale callback cannot leak into another account or scope", async () => {
  const f = fixture();
  const old = f.subscribePreview("1:old:original", "original", () => {});
  old();
  const current = f.subscribePreview("2:new:original", "original", () => {});
  f.calls[0].resolve(ready);
  await flush();
  assert.equal(f.previewSnapshot("2:new:original").view, null);
  f.calls[1].resolve(ready);
  await flush();
  assert.equal(f.previewSnapshot("2:new:original").view.previewId, "preview");
  current();
});
test("queued work polls its original job and a failed preview requires an explicit retry", async () => {
  const f = fixture();
  const close = f.subscribePreview("scope", "original", () => {});
  f.calls[0].resolve({
    ...ready,
    status: "QUEUED",
    previewId: null,
    width: null,
    height: null,
  });
  await flush();
  assert.equal(f.timers.size, 1);
  const [timer, callback] = [...f.timers][0];
  f.timers.delete(timer);
  callback();
  assert.equal(f.calls[1].kind, "get");
  f.calls[1].resolve({
    ...ready,
    status: "FAILED",
    previewId: null,
    canRetry: true,
    error: "解码失败",
  });
  await flush();
  assert.equal(f.timers.size, 0);
  assert.equal(f.calls.length, 2);
  f.retryPreview("scope");
  assert.equal(f.calls[2].kind, "ensure");
  assert.equal(f.calls[2].retryJobId, "job");
  f.calls[2].resolve({ ...ready, jobId: "retry-job" });
  await flush();
  assert.equal(f.previewSnapshot("scope").error, null);
  close();
});
test("the cache never displays a wrong source or silently returns the full original", async () => {
  for (const response of [
    { ...ready, sourceId: "another-private-source" },
    { ...ready, previewId: "original" },
  ]) {
    const f = fixture();
    const close = f.subscribePreview("scope", "original", () => {});
    f.calls[0].resolve(response);
    await flush();
    assert.equal(f.previewSnapshot("scope").view, null);
    assert.match(f.previewSnapshot("scope").error, /不一致/);
    close();
  }
});
test("duplicate cleanup cannot remove a new subscription using the same key", async () => {
  const f = fixture();
  const old = f.subscribePreview("scope", "original", () => {});
  old();
  const current = f.subscribePreview("scope", "original", () => {});
  old();
  f.calls[1].resolve(ready);
  await flush();
  assert.equal(f.previewSnapshot("scope").view.previewId, "preview");
  current();
});

test("temporary disconnection resumes by reading the accepted job instead of generating another preview", async () => {
  const f = fixture();
  const close = f.subscribePreview("scope", "original", () => {});
  f.calls[0].reject(new Error("network disconnected"));
  await flush();
  assert.equal(f.timers.size, 1);
  const [timer, callback] = [...f.timers][0];
  f.timers.delete(timer);
  callback();
  assert.equal(f.calls[1].kind, "get");
  f.calls[1].resolve(ready);
  await flush();
  assert.equal(f.previewSnapshot("scope").view.previewId, "preview");
  assert.equal(f.timers.size, 0);
  close();
});
