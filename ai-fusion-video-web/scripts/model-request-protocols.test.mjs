import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(new URL("../app/(dashboard)/settings/ai-models/_components/model-request-protocols.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const loaded = { exports: {} };
vm.runInThisContext(`(function(module,exports){${compiled}\n})`)(loaded, loaded.exports);
const { getProtocolOptionsForModelType } = loaded.exports;
const providers = [
  { id: "metaso", restrictedProtocols: ["metaso_video"] },
  { id: "infistar", restrictedProtocols: ["infistar_video"] },
];
const values = (id, catalog = providers) => getProtocolOptionsForModelType(3, id, catalog).map(item => item.value);

test("普通渠道不显示赞助商专用协议，赞助渠道只能选择自己的专用协议", () => {
  assert.ok(!values(null).includes("metaso_video"));
  assert.ok(!values(null).includes("infistar_video"));
  assert.ok(values("metaso").includes("metaso_video"));
  assert.ok(!values("metaso").includes("infistar_video"));
  assert.ok(values("infistar").includes("infistar_video"));
  assert.ok(values("infistar").includes("openai"));
});

test("目录加载失败或渠道已隐藏时，不暴露专用协议选项", () => {
  assert.deepEqual(values(null, []), []);
  assert.ok(!values(null, [providers[1]]).includes("metaso_video"));
});
