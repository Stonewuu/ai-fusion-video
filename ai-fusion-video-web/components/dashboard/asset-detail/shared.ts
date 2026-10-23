import type { Asset } from "@/lib/api/asset";
export const typeColorMap: Record<string, string> = {
  character: "text-blue-400 bg-blue-500/10",
  scene: "text-green-400 bg-green-500/10",
  prop: "text-amber-400 bg-amber-500/10",
};
export const typeLabelMap: Record<string, string> = {
  character: "角色",
  scene: "场景",
  prop: "道具",
};
/** 需要多行编辑的长文本字段 key */
export const LONG_TEXT_KEYS = new Set([
  "description",
  "relationship",
  "appearance",
  "personality",
]);
/** 复制已验证的原生 JSON 属性对象，避免表单修改响应对象 */
export function parseProps(
  raw: Record<string, unknown> | null,
): Record<string, unknown> {
  return raw === null ? {} : { ...raw };
}
export function propertyText(value: unknown): string {
  if (value === undefined || value === null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
}
export const assetTypeOptions = [
  { value: "character", label: "角色" },
  { value: "scene", label: "场景" },
  { value: "prop", label: "道具" },
];
export const itemTypeOptions = [
  { value: "initial", label: "初始图" },
  { value: "three_view", label: "三视图" },
  { value: "variant", label: "变体" },
];
// ========== Props 类型 ==========
export interface EditProps {
  isCreating?: false;
  asset: Asset;
  projectId?: never;
  onClose: () => void;
  onSaved: () => void;
  onDeleted: () => void;
  onCreated?: never;
}
export interface CreateProps {
  isCreating: true;
  asset?: never;
  projectId: number | null;
  onClose: () => void;
  onSaved: () => void;
  onDeleted?: never;
  onCreated: (created: Asset) => void;
}
export type Props = EditProps | CreateProps;
