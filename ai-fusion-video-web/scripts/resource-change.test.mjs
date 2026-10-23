import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

const file = new URL("../lib/store/resource-change-store.ts", import.meta.url);
const { outputText } = ts.transpileModule(await readFile(file, "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
});
const loaded = { exports: {} };
new Function("require", "module", "exports", outputText)(createRequire(file), loaded, loaded.exports);
const store = loaded.exports.useResourceChangeStore;
const revisions = () => ({ projects: 0, scripts: 0, assets: 0, storyboards: 0 });
const event = (sequence, resourceType) => ({ sequence, commandId: `c${sequence}`, change: { resourceType } });
const page = (cursor, events = [], extra = {}) => ({ cursor, head: cursor, scope: "owner", events, reset: false, hasMore: false, ...extra });
function reset() { store.setState({ cursor: 0, scope: "owner", revisions: revisions() }); }

test("committed resource changes invalidate all dependent native views once", () => {
  reset();
  const changes = page(8, [event(2, "assetItem"), event(4, "scriptEpisode"), event(8, "shot")]);
  store.getState().apply(changes);
  assert.deepEqual(store.getState().revisions, { projects: 0, scripts: 1, assets: 1, storyboards: 1 });
  store.getState().apply(changes);
  store.getState().apply(page(4, [event(4, "scriptEpisode")]));
  assert.equal(store.getState().cursor, 8);
  assert.deepEqual(store.getState().revisions, { projects: 0, scripts: 1, assets: 1, storyboards: 1 });
});

test("filtered inaccessible events still advance cursor without refreshing unrelated views", () => {
  reset();
  store.getState().apply(page(15));
  assert.equal(store.getState().cursor, 15);
  assert.deepEqual(store.getState().revisions, revisions());
});

test("permission-scope reset refreshes every view even when sequence did not change", () => {
  reset();
  store.getState().apply(page(15));
  store.getState().apply(page(15, [], { scope: "revoked-member", reset: true }));
  assert.equal(store.getState().scope, "revoked-member");
  assert.deepEqual(store.getState().revisions, { projects: 1, scripts: 1, assets: 1, storyboards: 1 });
});

test("invalid or mismatched pages cannot advance the acknowledged cursor", () => {
  for (const broken of [page(1, [event(2, "shot")]), page(4, [event(4, "shot"), event(3, "shot")]),
    page(4, [], { scope: "other" }), page(NaN), page(4, [event(2, "unregistered")])]) {
    reset();
    assert.throws(() => store.getState().apply(broken));
    assert.equal(store.getState().cursor, 0);
    assert.deepEqual(store.getState().revisions, revisions());
  }
});
