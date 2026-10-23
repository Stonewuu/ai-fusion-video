"use client";

import { useMemo, useRef } from "react";

/** 同一表单内容失败后重试沿用请求标识；成功或明确关闭表单后开始新操作。 */
export function useCommandRequestKey() {
  const pending = useRef(new Map<string, string>());
  return useMemo(() => ({
    forPayload(payload: unknown) {
      const signature = JSON.stringify(payload, (_key, value) => {
        if (value && typeof value === "object" && !Array.isArray(value)) {
          return Object.fromEntries(Object.keys(value).sort().map((key) => [key, value[key]]));
        }
        return value;
      });
      if (signature === undefined) throw new Error("操作参数不能为空");
      let key = pending.current.get(signature);
      if (!key) {
        key = crypto.randomUUID();
        pending.current.set(signature, key);
      }
      return key;
    },
    clear() { pending.current.clear(); },
  }), []);
}
