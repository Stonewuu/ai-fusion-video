import assert from "node:assert/strict";
import test from "node:test";
import {
  AUTH_REFRESH_FAILED_ERROR_CODE,
  AuthFlowError,
} from "./auth-flow-error";
import {
  getClientErrorMessage,
  isIgnoredClientError,
} from "./client-error-policy";

test("normalizes client error messages", () => {
  assert.equal(getClientErrorMessage(new Error("  渲染失败  ")), "渲染失败");
  assert.equal(getClientErrorMessage(null), "发生未知客户端异常");
});

test("ignores request cancellation errors", () => {
  const error = new Error("canceled") as Error & { code: string };
  error.name = "CanceledError";
  error.code = "ERR_CANCELED";
  assert.equal(isIgnoredClientError(error), true);
});

test("ignores automatic token refresh failures regardless of message", () => {
  const error = new AuthFlowError(
    AUTH_REFRESH_FAILED_ERROR_CODE,
    "刷新凭证已被撤销"
  );
  assert.equal(isIgnoredClientError(error), true);
});

test("ignores Next.js internal control flow digests", () => {
  const error = new Error("内部跳转") as Error & { digest: string };
  error.digest = "NEXT_REDIRECT;replace;/login;307";
  assert.equal(isIgnoredClientError(error), true);
});

test("does not ignore unexpected application errors", () => {
  assert.equal(isIgnoredClientError(new Error("组件渲染失败")), false);
});
