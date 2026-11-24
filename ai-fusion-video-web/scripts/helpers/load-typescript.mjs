import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import ts from "typescript";

const root = path.resolve(import.meta.dirname, "../..");
const nodeRequire = createRequire(import.meta.url);

export function createTypeScriptLoader() {
  const cache = new Map();
  const load = (relative) => {
    if (cache.has(relative)) {
      return cache.get(relative).exports;
    }
    const source = ts.transpileModule(fs.readFileSync(path.join(root, relative), "utf8"), {
      compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
    }).outputText;
    const loadedModule = { exports: {} };
    cache.set(relative, loadedModule);
    const require = (name) => {
      if (!name.startsWith(".") && !name.startsWith("@/")) {
        return nodeRequire(name);
      }
      const target = name.startsWith("@/") ? name.slice(2) : path.posix.join(path.posix.dirname(relative), name);
      return load(`${target}.ts`);
    };
    vm.runInThisContext(`(function(require,module,exports){${source}\n})`, { filename: relative })(require, loadedModule, loadedModule.exports);
    return loadedModule.exports;
  };
  return load;
}
