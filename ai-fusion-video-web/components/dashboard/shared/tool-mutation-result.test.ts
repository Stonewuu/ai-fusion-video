import assert from "node:assert/strict";
import test from "node:test";
import { buildToolMutationPresentation } from "./tool-mutation-result-model";

test("renders deleted storyboard resource with its concrete business type", () => {
  const presentation = buildToolMutationPresentation(
    {
      deletedResourceId: 140,
      deletedResourceType: "episode",
      status: "success",
    },
    "delete_storyboard_child_resource",
  );

  assert.equal(presentation.message, "分镜集已删除");
  assert.equal(presentation.label, "分镜集");
  assert.equal(presentation.id, 140);
});

test("renders nested project and storyboard mutation results", () => {
  const project = buildToolMutationPresentation(
    { operation: "created", project: { id: 8, name: "新项目" }, status: "success" },
    "save_project",
  );
  assert.equal(project.message, "项目创建成功");
  assert.equal(project.summary, "新项目");
  assert.equal(project.id, 8);

  const scene = buildToolMutationPresentation(
    {
      status: "success",
      storyboardScene: { id: 31, sceneHeading: "内景 客厅 夜", sceneNumber: "1-2" },
    },
    "update_storyboard_scene",
  );
  assert.equal(scene.message, "分镜场次更新成功");
  assert.equal(scene.summary, "1-2 · 内景 客厅 夜");
  assert.equal(scene.id, 31);
});
