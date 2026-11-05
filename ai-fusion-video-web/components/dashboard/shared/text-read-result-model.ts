export interface TextReadPage {
  content: string;
  offset: number;
  endOffset: number;
  totalChars: number;
  hasMore: boolean;
  sourceLabel: string;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function count(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function readTextPage(data: unknown, toolName: string): TextReadPage | null {
  if (!record(data) || data.status !== "success" || typeof data.content !== "string"
    || !count(data.offset) || !count(data.endOffset) || !count(data.totalChars)
    || data.offset > data.endOffset || data.endOffset > data.totalChars
    || typeof data.hasMore !== "boolean" || data.hasMore !== (data.endOffset < data.totalChars)) {
    return null;
  }
  if (data.hasMore ? data.nextOffset !== data.endOffset || data.endOffset === data.offset : data.nextOffset !== null) {
    return null;
  }
  if (toolName === "read_script_content" && ((data.resourceType !== "script" && data.resourceType !== "episode")
    || (data.field !== "rawContent" && data.field !== "content"))) {
    return null;
  }
  return {
    content: data.content,
    offset: data.offset,
    endOffset: data.endOffset,
    totalChars: data.totalChars,
    hasMore: data.hasMore,
    sourceLabel: toolName === "read_tool_result" ? "工具结果"
      : data.resourceType === "episode" ? "分集原文" : data.field === "content" ? "剧本正文" : "剧本原文",
  };
}

export function textReadSummary(page: TextReadPage) {
  const number = (value: number) => value.toLocaleString("zh-CN");
  const length = page.endOffset - page.offset;
  return {
    range: length > 0 ? `第 ${number(page.offset + 1)}–${number(page.endOffset)} 字` : "本次没有返回正文",
    total: `共 ${number(page.totalChars)} 字`,
    detail: page.totalChars === 0 ? page.sourceLabel === "工具结果" ? "结果为空" : "原文为空"
      : length === 0 ? "已到内容末尾"
        : `本次读取 ${number(length)} 字 · ${page.hasMore ? "后面还有内容" : "已到内容末尾"}`,
  };
}

export function savedTextResult(data: unknown): number | null {
  if (!record(data) || data.readTool !== "read_tool_result" || typeof data.resultId !== "string"
    || !/^[0-9a-f]{64}$/.test(data.resultId) || !count(data.totalChars)) {
    return null;
  }
  return data.totalChars;
}

export function scriptContentSummaries(data: unknown): { label: string; totalChars: number }[] {
  if (!record(data) || !record(data.contentReads)) {
    return [];
  }
  return Object.entries(data.contentReads).flatMap(([field, value]) => {
    if (!record(value) || value.readTool !== "read_script_content" || !count(value.totalChars)
      || (field !== "rawContent" && field !== "content")) {
      return [];
    }
    return [{ label: field === "rawContent" ? "原文" : "剧本正文", totalChars: value.totalChars }];
  });
}
