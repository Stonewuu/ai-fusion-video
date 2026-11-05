import test from "node:test";
import assert from "node:assert/strict";
import { generationTaskSummary } from "./generation-task-summary";

test("受理与执行中不能显示成生成成功", () => {
  assert.equal(generationTaskSummary("QUEUED", true).label, "已受理，等待生成");
  assert.equal(generationTaskSummary("RUNNING", false).tone, "pending");
  assert.equal(generationTaskSummary("SUCCEEDED", false).tone, "success");
});

test("结果未知与需要核实不能显示成失败或完成", () => {
  for (const state of ["UNKNOWN_SUBMISSION", "NEEDS_REVIEW"]) {
    assert.equal(generationTaskSummary(state, false).label, "生成结果需要核实");
  }
  assert.equal(generationTaskSummary("PARTIAL_FAILED", false).label, "部分完成");
});
