import assert from "node:assert/strict";
import test from "node:test";
import { getToolArgumentLabel } from "./tool-call-argument-metadata";
import {
  formatToolArgumentScalar,
  formatToolArgumentsJson,
  formatToolResultJson,
  listToolArgumentEntries,
  parseEmbeddedJsonArgument,
  parseToolArguments,
} from "./tool-call-argument-model";

test("parses query tool arguments into friendly entries", () => {
  const parsed = parseToolArguments('{"projectId":1,"detailLevel":"outline"}');
  assert.deepEqual(listToolArgumentEntries(parsed), [
    { key: "projectId", label: "项目 ID", value: 1 },
    { key: "detailLevel", label: "详情级别", value: "outline" },
  ]);
  assert.equal(formatToolArgumentScalar("detailLevel", "outline"), "大纲");
});

test("preserves nested batch parameters", () => {
  const parsed = parseToolArguments(JSON.stringify({
    projectId: 7,
    assets: [{
      type: "character",
      name: "林默",
      properties: { appearance: "黑发，身形挺拔" },
    }],
  }));
  assert.equal(Array.isArray(parsed.assets), true);
  assert.equal(formatToolArgumentScalar("type", "character"), "角色");
  assert.match(formatToolArgumentsJson(parsed), /"appearance": "黑发，身形挺拔"/);
});

test("formats media generation parameters", () => {
  const parsed = parseToolArguments(JSON.stringify({
    prompt: "电影感夜景",
    imageUrls: ["https://example.com/reference.png"],
    width: 1920,
    height: 1080,
  }));
  assert.equal(listToolArgumentEntries(parsed)[0]?.label, "生成提示词");
  assert.deepEqual(parsed.imageUrls, ["https://example.com/reference.png"]);
  assert.equal(formatToolArgumentScalar("width", 1920), "1920 px");
  assert.equal(formatToolArgumentScalar("duration", 5), "5 秒");
});

test("parses declared JSON string parameters as structured content", () => {
  assert.deepEqual(
    parseEmbeddedJsonArgument("charactersJson", '[{"name":"林默"}]'),
    [{ name: "林默" }],
  );
  assert.throws(
    () => parseEmbeddedJsonArgument("charactersJson", "not-json"),
    /charactersJson is not valid JSON/,
  );
});

test("accepts tools without arguments", () => {
  assert.deepEqual(parseToolArguments("{}"), {});
  assert.deepEqual(listToolArgumentEntries({}), []);
});

test("formats structured and plain-text tool results as valid JSON", () => {
  assert.equal(
    formatToolResultJson('{"status":"success","items":[1,2]}'),
    '{\n  "status": "success",\n  "items": [\n    1,\n    2\n  ]\n}',
  );
  assert.equal(formatToolResultJson(null), "null");
  assert.equal(formatToolResultJson("[FILE_PATH_REDACTED]"),
    '"[FILE_PATH_REDACTED]"');
  assert.equal(formatToolResultJson(""), '""');
});

test("renders lifecycle delete arguments with tool-specific Chinese semantics", () => {
  const parsed = parseToolArguments('{"resourceType":"episode","resourceId":140}');
  assert.deepEqual(
    listToolArgumentEntries(parsed, "delete_storyboard_child_resource"),
    [
      { key: "resourceType", label: "删除对象", value: "episode" },
      { key: "resourceId", label: "分镜集 ID", value: 140 },
    ],
  );
  assert.equal(
    formatToolArgumentScalar(
      "resourceType",
      "episode",
      "delete_storyboard_child_resource",
    ),
    "分镜集",
  );
  assert.equal(
    formatToolArgumentScalar("resourceType", "scene", "delete_script_child_resource"),
    "剧本场次",
  );
  assert.equal(
    formatToolArgumentScalar("resourceType", "item", "delete_asset_resource"),
    "子资产",
  );
});

test("formats lifecycle mutation enum and boolean arguments", () => {
  assert.equal(formatToolArgumentScalar("status", 2, "save_project"), "已完成");
  assert.equal(
    formatToolArgumentScalar("sourceType", 2, "update_asset_item"),
    "AI 生成",
  );
  assert.equal(
    formatToolArgumentScalar("status", 0, "update_storyboard_item"),
    "草稿",
  );
  assert.equal(formatToolArgumentScalar("overwriteMode", true), "覆盖已有数据");
});

test("rejects malformed JSON and non-object roots explicitly", () => {
  assert.throws(() => parseToolArguments("{invalid"), /not valid JSON/);
  assert.throws(() => parseToolArguments("[]"), /root must be a JSON object/);
  assert.throws(() => parseToolArguments(""), /must contain a JSON object/);
});

test("turns runtime MCP field names into readable labels", () => {
  assert.equal(getToolArgumentLabel("repository_name"), "Repository name");
  assert.equal(getToolArgumentLabel("includeArchivedItems"), "Include Archived Items");
});
