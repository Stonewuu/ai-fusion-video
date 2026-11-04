import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isMainThread, parentPort, workerData, Worker as ThreadWorker } from "node:worker_threads";
import ts from "typescript";

const require = createRequire(import.meta.url);
const root = path.resolve(import.meta.dirname, "../..");
const cache = new Map();

// 用独立线程执行应用的 Worker 入口，覆盖 self 环境及父子线程通信，不替代浏览器构建验收。
class BrowserWorker {
  onmessage = null;
  onerror = null;

  constructor(url) {
    this.thread = new ThreadWorker(new URL(import.meta.url), { workerData: { entry: fileURLToPath(url) } });
    this.thread.on("message", (data) => this.onmessage?.({ data }));
    this.thread.on("error", (error) => this.onerror?.(error));
  }

  postMessage(message) {
    this.thread.postMessage(message);
  }

  terminate() {
    return this.thread.terminate();
  }
}

export function loadBrowserWorkerModule(relative) {
  const file = path.resolve(root, relative);
  if (cache.has(file)) {
    return cache.get(file).exports;
  }
  const loadedModule = { exports: {} };
  cache.set(file, loadedModule);
  const compiled = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
  }).outputText.replaceAll("import.meta.url", JSON.stringify(pathToFileURL(file).href));
  const localRequire = (name) => {
    if (name.startsWith("@/")) {
      return loadBrowserWorkerModule(`${name.slice(2)}.ts`);
    }
    if (name.startsWith(".")) {
      return loadBrowserWorkerModule(`${path.resolve(path.dirname(file), name)}.ts`);
    }
    return require(name);
  };
  vm.runInThisContext(`(function(require,module,exports,Worker){${compiled}\n})`, { filename: file })(
    localRequire, loadedModule, loadedModule.exports, BrowserWorker,
  );
  return loadedModule.exports;
}

if (!isMainThread) {
  globalThis.self = globalThis;
  globalThis.postMessage = (message) => parentPort.postMessage(message);
  parentPort.on("message", (data) => globalThis.onmessage({ data }));
  loadBrowserWorkerModule(workerData.entry);
}
