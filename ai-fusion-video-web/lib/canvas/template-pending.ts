import type { TemplateImportRequest } from "@/lib/api/canvas-templates";

// 完整模板可达 16 MiB；IndexedDB 事务完成后才提交，避免 localStorage 容量限制丢失恢复请求。
function request<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open("fusion-canvas-template-imports", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("pending");
    open.onerror = () => reject(open.error);
    open.onblocked = () => reject(new Error("本地导入记录正在被其他窗口使用，请关闭旧窗口后重试。"));
    open.onsuccess = () => {
      const db = open.result;
      db.onversionchange = () => db.close();
      try {
        const tx = db.transaction("pending", mode);
        const result = action(tx.objectStore("pending"));
        tx.oncomplete = () => { db.close(); resolve(result.result); };
        tx.onabort = tx.onerror = () => { db.close(); reject(tx.error ?? new Error("无法保存本地导入记录，请检查浏览器存储空间。")); };
      } catch (failure) { db.close(); reject(failure); }
    };
  });
}
export async function readTemplatePending(key: string): Promise<TemplateImportRequest | null> {
  const value: unknown = await request("readonly", store => store.get(key));
  if (value === undefined) return null;
  if (!value || typeof value !== "object") throw new Error("本地模板导入记录损坏，请检查项目列表中的创建结果。");
  const data = value as Record<string, unknown>;
  if (typeof data.title !== "string" || typeof data.templateJson !== "string" || !data.bindings
    || ![data.canvasId, data.requestId].every(id => typeof id === "string" && /^[0-9a-f-]{36}$/.test(id))
    || typeof data.previewHash !== "string" || !/^[0-9a-f]{64}$/.test(data.previewHash))
    throw new Error("本地模板导入记录不完整，请检查项目列表中的创建结果。");
  return data as unknown as TemplateImportRequest;
}
export const saveTemplatePending = (key: string, value: TemplateImportRequest) => request("readwrite", store => store.put(value, key));
export const clearTemplatePending = (key: string) => request("readwrite", store => store.delete(key));
