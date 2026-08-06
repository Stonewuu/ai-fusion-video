import {
  getToolArgumentLabel,
  getToolArgumentValueLabel,
} from "./tool-call-argument-metadata";

export type ToolArgumentValue =
  | null
  | boolean
  | number
  | string
  | ToolArgumentValue[]
  | { [key: string]: ToolArgumentValue };

export type ToolArguments = Record<string, ToolArgumentValue>;

export interface ToolArgumentEntry {
  key: string;
  label: string;
  value: ToolArgumentValue;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertArgumentValue(value: unknown, path: string): asserts value is ToolArgumentValue {
  if (
    value === null
    || typeof value === "boolean"
    || typeof value === "number"
    || typeof value === "string"
  ) {
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertArgumentValue(item, `${path}[${index}]`));
    return;
  }
  if (isObject(value)) {
    Object.entries(value).forEach(([key, item]) => {
      if (!key.trim()) throw new Error(`Tool argument contains a blank key at ${path}`);
      assertArgumentValue(item, `${path}.${key}`);
    });
    return;
  }
  throw new Error(`Tool argument contains an unsupported value at ${path}`);
}

export function parseToolArguments(argumentsText: string): ToolArguments {
  if (!argumentsText.trim()) {
    throw new Error("Tool arguments must contain a JSON object");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(argumentsText);
  } catch (cause) {
    throw new Error("Tool arguments are not valid JSON", { cause });
  }
  if (!isObject(parsed)) {
    throw new Error("Tool arguments root must be a JSON object");
  }
  assertArgumentValue(parsed, "$arguments");
  return parsed;
}

export function listToolArgumentEntries(
  argumentsValue: ToolArguments,
  toolName?: string,
): ToolArgumentEntry[] {
  return Object.entries(argumentsValue).map(([key, value]) => ({
    key,
    label: getToolArgumentLabel(key, { arguments: argumentsValue, toolName }),
    value,
  }));
}

export function formatToolArgumentScalar(
  key: string,
  value: null | boolean | number | string,
  toolName?: string,
): string {
  if (value === null) return "未设置";
  if (typeof value === "boolean") {
    return getToolArgumentValueLabel(key, String(value), toolName)
      ?? (value ? "是" : "否");
  }
  if (typeof value === "number") {
    const declaredLabel = getToolArgumentValueLabel(key, value, toolName);
    if (declaredLabel) return declaredLabel;
    if (key === "duration") return `${value} 秒`;
    if (key === "width" || key === "height") return `${value} px`;
    return String(value);
  }
  if (value.length === 0) return "（空文本）";
  if (key === "scriptEpisodeIds") {
    return value.split(",").map((item) => item.trim()).join("、");
  }
  return getToolArgumentValueLabel(key, value, toolName) ?? value;
}

export function parseEmbeddedJsonArgument(
  key: string,
  value: string,
): ToolArgumentValue | undefined {
  if (key !== "charactersJson") return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch (cause) {
    throw new Error(`${key} is not valid JSON`, { cause });
  }
  assertArgumentValue(parsed, `$arguments.${key}`);
  return parsed;
}

export function formatToolArgumentsJson(argumentsValue: ToolArguments): string {
  return JSON.stringify(argumentsValue, null, 2);
}

export function formatToolResultJson(resultText: string | null): string {
  if (resultText === null) return "null";
  let parsed: unknown;
  try {
    parsed = JSON.parse(resultText);
  } catch {
    parsed = resultText;
  }
  assertArgumentValue(parsed, "$result");
  return JSON.stringify(parsed, null, 2);
}
