import type { StorageConfigSaveReq } from "@/lib/api/storage";

/** 编辑表单中的空值用于展示；仅提交修改项，避免把数据库的 null 变成空串。 */
export function storageUpdatePayload(form: StorageConfigSaveReq, original: StorageConfigSaveReq): StorageConfigSaveReq {
  const changes = Object.fromEntries(Object.entries(form).filter(([key, value]) =>
    JSON.stringify(value) !== JSON.stringify(original[key as keyof StorageConfigSaveReq])));
  return { ...changes, id: form.id, name: form.name };
}
