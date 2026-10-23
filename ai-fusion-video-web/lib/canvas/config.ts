import type { CanvasCapability, CanvasConfig } from "./types";
import { canvasFieldLabels } from "./labels";

export function validateCanvasConfig(
  capability: CanvasCapability,
  config: CanvasConfig,
  complete = true,
): string | null {
  for (const [key, field] of Object.entries(capability.configFields)) {
    const value = config[key];
    const label = canvasFieldLabels[key] ?? key;
    if (value === undefined) {
      if (complete && field.required) return `请填写${label}`;
      continue;
    }
    if (field.type === "integer" || field.type === "number") {
      if (
        typeof value !== "number" ||
        !Number.isFinite(value) ||
        (field.type === "integer" && !Number.isSafeInteger(value))
      )
        return `${label}必须是有效${field.type === "integer" ? "整数" : "数字"}`;
      if (
        (field.minimum !== null && value < field.minimum) ||
        (field.maximum !== null && value > field.maximum)
      )
        return `${label}超出允许范围`;
    } else if (field.type === "string_array") {
      if (!Array.isArray(value) || field.maximum !== null && value.length > field.maximum ||
        value.some((item) => typeof item !== "string" || !item.trim() || field.maxLength !== null && item.length > field.maxLength) ||
        new Set(value).size !== value.length) return `${label}选择无效，请检查数量和重复项`;
    } else if (
      field.type === "object" &&
      (typeof value !== "object" ||
        value === null ||
        Array.isArray(value) ||
        (field.maxLength !== null &&
          JSON.stringify(value).length > field.maxLength))
    )
      return `${label}配置无效`;
    else if (field.type === "boolean" && typeof value !== "boolean")
      return `${label}必须是开关值`;
    else if (
      field.type === "string" &&
      (typeof value !== "string" ||
        (field.maxLength !== null && value.length > field.maxLength))
    )
      return `${label}内容不符合长度要求`;
    if (field.enumValues.length && !field.enumValues.includes(String(value)))
      return `${label}不在允许选项中`;
  }
  return null;
}
