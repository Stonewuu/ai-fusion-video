import assert from "node:assert/strict";
import test from "node:test";
import { getToolDisplayName } from "./ai-task-display";

test("declares Chinese display names for lifecycle tools", () => {
  const expectedNames = {
    delete_asset_resource: "删除资产资源",
    delete_project: "删除项目",
    delete_script_child_resource: "删除剧本子资源",
    delete_storyboard_child_resource: "删除分镜子资源",
    save_project: "保存项目",
    update_asset_item: "更新子资产",
    update_storyboard_item: "更新分镜镜头",
    update_storyboard_scene: "更新分镜场次",
  };

  for (const [toolName, displayName] of Object.entries(expectedNames)) {
    assert.equal(getToolDisplayName(toolName), displayName);
  }
});
