import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "..");
function harness() {
  const window = {};
  const modules = new Map();
  const localStorage = { getItem: () => null };
  function load(file) {
    if (modules.has(file)) {
      return modules.get(file).exports;
    }
    const loadedModule = { exports: {} };
    modules.set(file, loadedModule);
    const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    }).outputText;
    const localRequire = name => name.startsWith("@/") ? load(path.join(root, name.slice(2) + ".ts"))
      : name.startsWith(".") ? load(path.resolve(path.dirname(file), name + ".ts")) : require(name);
    vm.runInNewContext(compiled, { module: loadedModule, exports: loadedModule.exports, require: localRequire, window, localStorage, URL });
    return loadedModule.exports;
  }
  const client = load(path.join(root, "lib/api/client.ts"));
  return { window, client };
}

test("a client imported before runtime configuration uses the backend configured before the request", async () => {
  const { window, client } = harness();
  assert.equal(client.getApiBaseUrl(), "");
  window.__FUSION_RUNTIME_CONFIG__ = { apiBaseUrl: "https://api.example.test///" };
  let requestedBase;
  const result = await client.http.get("/api/canvases", {
    adapter: async config => {
      requestedBase = config.baseURL;
      return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { ok: true } } };
    },
  });
  assert.equal(requestedBase, "https://api.example.test");
  assert.equal(result.ok, true);
  assert.equal(client.resolveMediaUrl("/api/media/file"), "https://api.example.test/api/media/file");
});

test("explicit same-origin deployment keeps relative requests and external media URLs intact", async () => {
  const { window, client } = harness();
  window.__FUSION_RUNTIME_CONFIG__ = { apiBaseUrl: "" };
  await client.http.get("/api/system/init/status", {
    adapter: async config => {
      assert.equal(config.baseURL, "");
      assert.equal(config.headers.Authorization, undefined);
      return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: {} } };
    },
  });
  assert.equal(client.resolveMediaUrl("/media/image.png"), "/media/image.png");
  assert.equal(client.resolveMediaUrl("https://cdn.example.test/image.png"), "https://cdn.example.test/image.png");
});
