"use client";

import { useCallback, useRef, useState, type SetStateAction } from "react";
import { EMPTY_GENERATION_FORM, type GenerationFormState } from "./generation-types";

export function useGenerationForm() {
  const current = useRef(EMPTY_GENERATION_FORM);
  const [form, setState] = useState(EMPTY_GENERATION_FORM);
  const setForm = useCallback((update: SetStateAction<GenerationFormState>) => {
    // 连续点击移除、添加时，在下一次渲染前也要使用上一操作更新后的素材列表。
    const next = typeof update === "function" ? update(current.current) : update;
    current.current = next;
    setState(next);
  }, []);
  const getForm = useCallback(() => current.current, []);
  return [form, setForm, getForm] as const;
}
