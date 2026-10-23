"use client";

import { useEffect, useRef } from "react";
import { useResourceChangeStore, type ResourceArea } from "@/lib/store/resource-change-store";
import { toastApiError } from "@/lib/api/toast-api-error";

/** 仅重读已提交数据；业务编辑器负责保留尚未提交的草稿。 */
export function useResourceInvalidation(area: ResourceArea | readonly ResourceArea[], refresh: () => void | Promise<unknown>) {
  const revision = useResourceChangeStore((state) => typeof area === "string" ? state.revisions[area]
    : area.reduce((total, entry) => total + state.revisions[entry], 0));
  const seen = useRef(revision);
  const callback = useRef(refresh);
  useEffect(() => { callback.current = refresh; }, [refresh]);
  useEffect(() => {
    if (seen.current === revision) return;
    seen.current = revision;
    void Promise.resolve().then(() => callback.current())
      .catch((error) => toastApiError(error, "更新资源失败"));
  }, [revision]);
}
