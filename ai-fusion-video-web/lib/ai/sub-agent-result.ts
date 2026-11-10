/** 将平台子任务回执转成用户可读的结果，内部运行标识保留在运行日志中。 */
export function formatSubAgentResult(raw: string): string {
  const text = raw.replace(/^Error:\s*/i, "").trim();
  if (!text.startsWith("{")) {
    return readableError(text);
  }
  let result: unknown;
  try {
    result = JSON.parse(text);
  } catch {
    return "子任务结果暂时无法读取，请刷新后重试。";
  }
  if (!result || typeof result !== "object" || !("executionMode" in result)
      || result.executionMode !== "PLATFORM_MANAGED_CHILD_RUN") {
    return raw;
  }
  if ("error" in result && typeof result.error === "string" && result.error) {
    return readableError(result.error);
  }
  const output = "result" in result ? result.result : null;
  if (typeof output === "string") {
    return output;
  }
  if (output && typeof output === "object" && "message" in output && typeof output.message === "string") {
    return output.message;
  }
  return "status" in result && result.status === "COMPLETED" ? "子任务已完成。" : "子任务未完成，请稍后继续。";
}

function readableError(text: string): string {
  if (/^(request failed|failed to fetch|fetch failed)$/i.test(text)) {
    return "模型请求中断，请稍后继续。";
  }
  return text;
}
